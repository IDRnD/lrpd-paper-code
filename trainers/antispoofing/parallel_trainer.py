import json
import multiprocessing as mp
import random
import sys
from functools import partial
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# adding pipeline root to sys.path
pipeline_root = Path(__file__).resolve().parents[2]
if str(pipeline_root) not in sys.path:
    print(f"Adding pipeline root in sys.path: {pipeline_root}")
    sys.path.append(str(pipeline_root))

from datautils.audio_processing.augmentors import *
from datautils.parallel_dataset import (ParallelDataset, parallel_collate_func, transforms)
from models.model_builders import AudioClassificationModel
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from utils.lr_scheduler import generate_lr_array
from utils.vis import plot_confusion_matrix


DEFAULT_SEED = 23


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_data_sp(source_root, parallel_roots, glob_str="**/*.wav", add_source=False):
    """
    assert that parallel_root wil contain folders of following structure:
    PARALLEL_ROOT/record_17/IPhone 12 Pro Max/JBL CLIP3/distance=60-loudness=15-recording_mode=default/RELATIVE_PATH_TO_WAV_FROM_SOURCE
    """
    data = defaultdict(list)
    source_root = Path(source_root).resolve()
    parallel_roots = [Path(parallel_root) for parallel_root in parallel_roots]
    #     print(parallel_roots)
    _class_ind_maps = defaultdict(list)
    source_pathes = list(source_root.glob(glob_str))
    if add_source:
        _class_ind_maps["spoofing"] = ["genuine", "spoof"]
    for source_path in tqdm(source_pathes):
        for parallel_root in parallel_roots:
            playback_device = parallel_root.parts[-2].lower().replace(" ", "")
            recording_device = parallel_root.parts[-3].lower().replace(" ", "")
            #             print(f"{playback_device}, {recording_device}")
            if not (playback_device in _class_ind_maps["playback_device"]):
                _class_ind_maps["playback_device"].append(playback_device)
            if not (recording_device in _class_ind_maps["recording_device"]):
                _class_ind_maps["recording_device"].append(recording_device)
            source_rlp = source_path.relative_to(source_root)
            parallel_path = parallel_root / source_rlp
            if parallel_path.exists():
                data[source_path].append({
                    "path": parallel_path,
                    "spoofing": "spoof",
                    "playback_device": playback_device,
                    "recording_device": recording_device
                })
        if add_source:
            if len(data[source_path]) > 0:
                data[source_path].insert(
                    0, {
                        "path": source_path,
                        "spoofing": "genuine",
                        "playback_device": None,
                        "recording_device": None
                    })
    class_ind_maps = defaultdict(dict)
    print(_class_ind_maps)
    for task_name, task_classes in _class_ind_maps.items():
        for cls_ind, cls_name in enumerate(sorted(task_classes)):
            class_ind_maps[task_name][cls_name] = cls_ind
    return data, class_ind_maps


def get_source_parallels(args):
    source_path, source_root, parallel_roots, add_source = args
    data = defaultdict(list)
    for parallel_root in parallel_roots:
        playback_device = parallel_root.parts[-2].lower().replace(" ", "")
        recording_device = parallel_root.parts[-3].lower().replace(" ", "")
        source_rlp = source_path.relative_to(source_root)
        parallel_path = parallel_root / source_rlp
        if parallel_path.exists():
            data[source_path].append({
                "path": parallel_path,
                "spoofing": "spoof",
                "playback_device": playback_device,
                "recording_device": recording_device
            })
    if add_source:
        if len(data[source_path]) > 0:
            data[source_path].insert(
                0, {
                    "path": source_path,
                    "spoofing": "genuine",
                    "playback_device": None,
                    "recording_device": None
                })
    return data


def parse_data_mp(source_root, parallel_roots, glob_str="**/*.wav", add_source=False):
    """
    assert that parallel_root wil contain folders of following structure:
    PARALLEL_ROOT/record_17/IPhone 12 Pro Max/JBL CLIP3/distance=60-loudness=15-recording_mode=default/RELATIVE_PATH_TO_WAV_FROM_SOURCE
    """
    data = defaultdict(list)
    source_root = Path(source_root).resolve()
    parallel_roots = [Path(parallel_root) for parallel_root in parallel_roots]
    _class_ind_maps = defaultdict(list)
    source_pathes = list(source_root.glob(glob_str))
    if add_source:
        _class_ind_maps["spoofing"] = ["genuine", "spoof"]

    result = []
    with tqdm(total=len(source_pathes)) as pbar:
        with mp.Pool(processes=24) as p:
            for parallels in p.imap(
                    get_source_parallels,
                    zip(source_pathes, [source_root] * len(source_pathes),
                        [parallel_roots] * len(source_pathes), [add_source] * len(source_pathes))):
                result.append(parallels)
                pbar.update(1)

    data = {}
    result = [e for e in result if len(e) > 0]
    for parallels_dict in result:
        for source_path, parallels in parallels_dict.items():
            data[source_path] = parallels
            for sample in parallels:
                if sample["spoofing"] == "genuine":
                    continue
                if not (sample["playback_device"] in _class_ind_maps["playback_device"]):
                    _class_ind_maps["playback_device"].append(sample["playback_device"])
                if not (sample["recording_device"] in _class_ind_maps["recording_device"]):
                    _class_ind_maps["recording_device"].append(sample["recording_device"])

    class_ind_maps = defaultdict(dict)
    print(_class_ind_maps)
    for task_name, task_classes in _class_ind_maps.items():
        for cls_ind, cls_name in enumerate(sorted(task_classes)):
            class_ind_maps[task_name][cls_name] = cls_ind
    return data, class_ind_maps


parse_data = parse_data_mp


def merge_data(*data_dicts):
    merged = defaultdict(list)
    for dd in data_dicts:
        for key, value in dd.items():
            merged[key].extend(value)
    return merged


def convert_data_dict_to_dataset_data_list(data_dict: dict):
    data_list = []
    for source_path, parallel_pathes in data_dict.items():
        data_list.append({"source": source_path, "parallel": parallel_pathes})
    return data_list


def generate_augmentor_func(augmentor_setup):
    augmentors_dict = {}
    for task_label_tuple, augmentor_to_eval in augmentor_setup.items():
        if isinstance(task_label_tuple, list):
            task_label_tuple = tuple(task_label_tuple)
        augmentors_dict[task_label_tuple] = eval(augmentor_to_eval)

    def augment(samples, utt_cls_dict):
        for k, v in utt_cls_dict.items():
            aug_key = f"{k}:{v}"
            if aug_key in augmentors_dict:
                samples = augmentors_dict[aug_key](samples)
        if "*" in augmentors_dict:
            samples = augmentors_dict["*"](samples)
        return samples

    return augment


class AntispoofingClassifierParallel(pl.LightningModule):
    def __init__(
        self,
        model_config: Dict = {},
        train_config: Dict = {},
        data_config: Dict = {},
        seed: int = DEFAULT_SEED,
        workers: int = 16,
    ):
        super(AntispoofingClassifierParallel, self).__init__()
        set_random_seed(seed)
        self.workers = workers
        self.data_config = data_config
        self.model_config = model_config
        self.train_config = train_config

        self.epochs = train_config["epochs"]
        # self.batch_size = train_config["batch_size"]
        self.steps_per_epoch = train_config["steps_per_epoch"]

        self.result_df = pd.DataFrame()
        self.generate_lr_array()
        self.nnet = AudioClassificationModel(**model_config)
        # loss_conf = train_config["loss"]
        # self.loss = eval(loss_conf["type"])(**loss_conf["params"])
        self.activation = eval(train_config["activation"])

    def setup(self, *args, **kwargs):
        # Dump configs
        self.output_dir = Path(self.trainer.logger.log_dir).resolve()
        print(f"output_dir : {self.output_dir}")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        (self.output_dir / "data_config.json").write_text(json.dumps(self.data_config, indent=4))
        (self.output_dir / "model_config.json").write_text(json.dumps(self.model_config, indent=4))
        (self.output_dir / "train_config.json").write_text(json.dumps(self.train_config, indent=4))
        (self.output_dir / "class_ind_maps.json").write_text(
            json.dumps(self.class_ind_maps, indent=4))

    def prepare_data(self):
        # Load augmentor
        if "augmentor" in self.data_config:
            self.augmentor = generate_augmentor_func(self.data_config["augmentor"])
        else:
            self.augmentor = None

        # Load data:
        train_data_to_merge = []
        all_class_ind_maps = []

        for train_data_setup in self.data_config["dataset"]["train"]:
            parsed_train_part, class_ind_maps = parse_data(
                train_data_setup["source_root"],
                train_data_setup["parallel_roots"],
                add_source=self.data_config["use_source"])
            print(
                f"{train_data_setup['source_root']} : {len(parsed_train_part)}, mean number of parallel : {np.mean([len(v) for v in parsed_train_part.values()])}"
            )
            pprint(class_ind_maps)
            remove_path_num = 0
            for src_path in list(parsed_train_part.keys()):
                if len(parsed_train_part[src_path]) == 0:
                    remove_path_num += 1
                    del parsed_train_part[src_path]
            print(f"remove_path_num : {remove_path_num}")
            train_data_to_merge.append(parsed_train_part)
            all_class_ind_maps.append(class_ind_maps)
        merged_train = merge_data(*train_data_to_merge)
        merged_train = convert_data_dict_to_dataset_data_list(merged_train)
        print(
            f"train after merge : {len(merged_train)}, mean number of parallel : {np.mean([len(v['parallel']) for v in merged_train])}"
        )
        train_data, val_data = train_test_split(merged_train,
                                                test_size=self.data_config["valid_size"] /
                                                len(merged_train))
        print(f"val_data : {len(val_data)}")

        self.train_data = train_data
        self.class_ind_maps = defaultdict(list)
        for class_ind_maps in all_class_ind_maps:
            for class_task, task_cls_ind_map in class_ind_maps.items():
                self.class_ind_maps[class_task].extend(
                    [cls_name for cls_name in task_cls_ind_map.keys()])

        for class_task in self.class_ind_maps.keys():
            self.class_ind_maps[class_task] = dict([
                (cls_name, cls_ind)
                for cls_ind, cls_name in enumerate(sorted(set(self.class_ind_maps[class_task])))
            ])
            # Define None class index as -100 as it is default ignore_index for torch.nn.CrossEntropyLoss
            # Refer to doc for more info https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            self.class_ind_maps[class_task][None] = -100

        pprint(self.class_ind_maps)
        pprint([(k, len(v)) for k, v in self.class_ind_maps.items()])

        self.ind_class_maps = defaultdict(dict)
        for class_task in self.class_ind_maps.keys():
            for k, v in self.class_ind_maps[class_task].items():
                self.ind_class_maps[class_task][v] = k

        pprint(self.ind_class_maps)

        self.val_data = {"validation": val_data}
        for test_setup in self.data_config["dataset"]["test"]:
            parsed_test_set, test_class_ind_maps = parse_data(
                test_setup["test_setup"]["source_root"],
                test_setup["test_setup"]["parallel_roots"],
                add_source=self.data_config["use_source"])
            for task_name in test_class_ind_maps.keys():
                for test_cls_name in test_class_ind_maps[task_name].keys():
                    assert test_cls_name in self.class_ind_maps[task_name]
            test_set = convert_data_dict_to_dataset_data_list(parsed_test_set)
            _, test_set = train_test_split(test_set,
                                           test_size=min(
                                               1.0,
                                               test_setup["samples_per_domain"] / len(test_set)))
            print(f"test size {test_setup['test_name']} : {len(test_set)}")
            self.val_data[test_setup["test_name"]] = test_set

    def generate_lr_array(self):
        # Generate lr-array
        self.lr_array = np.concatenate(
            [generate_lr_array(**lr_conf) for lr_conf in self.train_config["lr_scheduler"]])

    def forward(self, x):
        return self.nnet(x)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        # OPTIONAL
        #         x, labels = batch
        # print(f"x.size() : {x.size()}, labels.size() : {labels.size()}")
        pred = self.nnet(batch["x"])
        losses = {}
        splited_preds = {}
        for task_name in sorted(self.class_ind_maps.keys()):
            task_predictions = pred[task_name]
            splited_preds[task_name] = self.activation(task_predictions)
            loss = F.cross_entropy(task_predictions, batch[task_name])
            # loss = self.loss(task_predictions,batch[task_name])
            losses[task_name] = loss
        del batch["x"]
        return {
            'loss': sum(losses.values()),
            'losses': losses,
            'labels': batch,
            'pred': splited_preds
        }

    def training_step(self, batch, batch_idx):
        # REQUIRED
        #         print(f"x.size() : {batch['x'].size()}")
        pred = self.nnet(batch["x"])
        losses = {}
        accs = {}
        for task_name in sorted(self.class_ind_maps.keys()):
            task_predictions = pred[task_name]
            if (task_name == "spoofing") and self.data_config["use_source"]:
                cls_weights = torch.from_numpy(np.array([self.data_config["num_parallel"] - 1,
                                                         1])).float().to(batch[task_name].device)
            else:
                cls_weights = None
            loss = F.cross_entropy(task_predictions, batch[task_name], weight=cls_weights)
            # loss = self.loss(task_predictions,batch[task_name])
            accs[task_name] = torch.mean(
                torch.eq(torch.argmax(task_predictions, dim=-1), batch[task_name]).to(torch.float))
            losses[task_name] = loss

            self.log(f'{task_name}-loss',
                     losses[task_name],
                     prog_bar=True,
                     logger=True,
                     on_step=True,
                     on_epoch=False,
                     add_dataloader_idx=False)
            self.log(f'{task_name}-acc',
                     accs[task_name],
                     prog_bar=True,
                     logger=True,
                     on_step=True,
                     on_epoch=False,
                     add_dataloader_idx=False)

        total_loss = sum(losses.values())
        return {'loss': total_loss}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        df_update_data = {"epoch": [self.current_epoch]}
        losses, accs, eers = {}, {}, {}

        # TODO: Remove KOSTUL when someone provides only one validation dataset, outputs is not list of lists
        if len(self.val_data) == 1:
            outputs = [outputs]

        for val_set_name, val_set_results in zip(self.val_data.keys(), outputs):
            for task_name in sorted(self.class_ind_maps.keys()):
                # pass
                predictions = torch.cat([x['pred'][task_name]
                                         for x in val_set_results]).cpu().numpy()
                labels = torch.cat([x['labels'][task_name] for x in val_set_results]).cpu().numpy()
                num_classes = predictions.shape[1]
                # ohes = torch.eye(num_classes,device=None,dtype=torch.float)[labels]

                avg_val_loss = torch.stack([x['losses'][task_name]
                                            for x in val_set_results]).mean()
                avg_val_acc = np.mean(
                    np.equal(np.argmax(predictions, axis=-1), labels)[labels >= 0])

                df_update_data[f"{task_name}-loss-{val_set_name}"] = avg_val_loss.item()
                df_update_data[f"{task_name}-acc-{val_set_name}"] = avg_val_acc.item()

                losses[f"{task_name}-{val_set_name}"] = df_update_data[
                    f"{task_name}-loss-{val_set_name}"]
                accs[f"{task_name}-{val_set_name}"] = df_update_data[
                    f"{task_name}-acc-{val_set_name}"]

                cls_names = [
                    self.ind_class_maps[task_name][cls_ind]
                    for cls_ind in sorted(self.class_ind_maps[task_name].values())
                ]
                conf_mat = confusion_matrix(labels,
                                            np.argmax(predictions, axis=-1),
                                            normalize="true",
                                            labels=np.arange(len(cls_names)))
                conf_mat_fig = plot_confusion_matrix(conf_mat,
                                                     y_names=cls_names,
                                                     x_names=cls_names,
                                                     title=f"cm-{task_name}",
                                                     matap="magma_r",
                                                     sort_mat=False,
                                                     transpose_axes=False,
                                                     cell_scale=1.5,
                                                     cmap_lims=None,
                                                     ax=None,
                                                     mat_mult=100.0)
                self.logger.experiment.add_figure(f"cm-{task_name}-{val_set_name}",
                                                  conf_mat_fig,
                                                  global_step=self.global_step,
                                                  close=True,
                                                  walltime=None)

        self.log("dummy_loss",
                 1,
                 prog_bar=False,
                 logger=True,
                 on_step=False,
                 on_epoch=True,
                 add_dataloader_idx=False)
        if self.global_step > 0:
            self.result_df = self.result_df.append(pd.DataFrame(df_update_data))
            self.result_df.to_csv(str(Path(self.trainer.logger.log_dir) / "logs.csv"), index=False)
        for task_name in sorted(self.class_ind_maps.keys()):
            self.log(f"val_losses",
                     losses,
                     prog_bar=False,
                     logger=True,
                     on_step=False,
                     on_epoch=True,
                     add_dataloader_idx=False)
            self.log(f"val_accs",
                     accs,
                     prog_bar=False,
                     logger=True,
                     on_step=False,
                     on_epoch=True,
                     add_dataloader_idx=False)
        return {}

    def configure_optimizers(self):
        opt_conf = self.train_config["optimizer"]
        Optimizer = eval(opt_conf["type"])
        optimizer = Optimizer(
            self.nnet.parameters(),
            lr=self.lr_array[0],  # Default LR for all groups of params
            **opt_conf["params"])

        scheduler = {
            'scheduler':
            torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda it: self.lr_array[it] / self.lr_array[0]),  # The LR schduler
            'interval':
            'step',  # The unit of the scheduler's step size
            'frequency':
            1,  # The frequency of the scheduler
            'reduce_on_plateau':
            False,  # For ReduceLROnPlateau scheduler
            'monitor':
            'val_eer'  # Metric to monitor
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        # REQUIRED
        train_dataset = ParallelDataset(
            data=self.train_data,
            size=self.steps_per_epoch * self.train_config["train_batch_size"],
            augmentor=self.augmentor,
            class_ind_maps=self.class_ind_maps,
            add_zero_parallel=self.data_config["use_source"],
            transform=transforms.Lambda(lambda np_audio: torch.from_numpy(np_audio)[:, None, :]),
            num_parallel_versions=self.data_config["num_parallel"],
            samplerate=self.data_config["samplerate"],
            utt_len_sec=self.data_config["utt_len_sec"])
        return DataLoader(dataset=train_dataset,
                          batch_size=self.train_config["train_batch_size"],
                          shuffle=False,
                          sampler=None,
                          batch_sampler=None,
                          num_workers=self.workers,
                          collate_fn=parallel_collate_func,
                          pin_memory=True,
                          drop_last=False,
                          timeout=0,
                          worker_init_fn=None,
                          multiprocessing_context=None)

    def val_dataloader(self):
        # OPTIONAL
        val_dls = []
        for val_set_name in self.val_data.keys():
            val_dataset = ParallelDataset(
                data=self.val_data[val_set_name],
                class_ind_maps=self.class_ind_maps,
                transform=transforms.Lambda(
                    lambda np_audio: torch.from_numpy(np_audio)[:, None, :]),
                num_parallel_versions=None,
                samplerate=self.data_config["samplerate"],
                utt_len_sec=self.data_config["utt_len_sec"])
            val_dl = DataLoader(dataset=val_dataset,
                                batch_size=self.train_config["test_batch_size"],
                                shuffle=False,
                                sampler=None,
                                batch_sampler=None,
                                num_workers=self.workers,
                                collate_fn=parallel_collate_func,
                                pin_memory=True,
                                drop_last=True,
                                timeout=0,
                                worker_init_fn=None,
                                multiprocessing_context=None)
            val_dls.append(val_dl)
        return val_dls


# ------------------------------------
#        EXAMPLE DATA CONFIG
# ------------------------------------

AUGS_ROOT = "/media/ssdraid0cgpu01/data/augmentations/"

data_config = {
    "utt_len_sec": 2.0,
    "num_parallel": 4,
    "samplerate": 16000,
    "valid_size": 20000,
    "use_source": True,
    "augmentor": {
        "*":
        f"""Sequential([
            OneOf([
                CustomNoises(noises_folder="{AUGS_ROOT}/MUSAN_15_sec_cut-16K/", snrs=(4., 16.), prob=1.),
                CustomNoises(noises_folder="{AUGS_ROOT}/DEMAND_15_sec_cut-16K/", snrs=(4., 16.), prob=1.),
                CustomNoises(noises_folder="{AUGS_ROOT}/DECASE_2017_Task3_Acoustic_Scenes-16K/", snrs=(4., 16.), prob=1.),
            ],prob=0.6),
        ],prob=1.0)""",
        "spoofing:genuine":
        f"""CustomNoises(noises_folder="/media/ssdraid0cgpu01/data/augmentations/InOfficeAcousticRoomSilence/", snrs=(0., 9.), prob=0.8)"""
    },
    "dataset": {
        "train": [
            {
                "source_root":
                "/media/ssdraid0cgpu01/data/antispoofing/16K/tests/LargeReplayTest_v1.0/Human/",
                "parallel_roots":
                list(
                    Path(
                        "/media/ssdraid0cgpu01/data/antispoofing/16K/tests/LargeReplayTest_v1.0/Replay/"
                    ).glob("*/*/*/*/"))
            },
            {
                "source_root":
                "/media/ssdraid0cgpu01/data/antispoofing/16K/Train/Human_source_InternalReplayTrain2021/",
                "parallel_roots":
                list(
                    Path(
                        "/media/ssdraid0cgpu01/data/antispoofing/16K/Train/InternalReplayTrain2021/"
                    ).glob("*/*/*/*/"))
            },
        ],
        "test": [
            {
                "test_name": "NewData",
                "samples_per_domain": 20000,
                "test_setup": {
                    "source_root":
                    "/media/ssdraid0cgpu01/data/antispoofing/16K/tests/LargeReplayTest_v2.0.0/Human/",
                    "parallel_roots":
                    list(
                        Path(
                            "/media/ssdraid0cgpu01/data/antispoofing/16K/tests/LargeReplayTest_v2.0.0/Replay"
                        ).glob("*/*/*/*/"))
                },
            },
        ]
    }
}

# ------------------------------------
#        EXAMPLE TRAIN CONFIG
# ------------------------------------

num_epochs = 48
steps_per_epoch = 3000
train_config = {
    "epochs":
    num_epochs,
    "train_batch_size":
    32,
    "test_batch_size":
    4,
    "steps_per_epoch":
    steps_per_epoch,
    "lr_scheduler": [
        dict(scheduler_type="Cosine",
             base_lr=1e-4,
             max_lr=3e-2,
             gamma=0.7,
             period_init_len=6,
             period_exp_fac=1.,
             steps_per_epoch=steps_per_epoch,
             num_epochs=num_epochs)
    ],
    "optimizer": {
        "type": "torch.optim.SGD",
        "params": dict(momentum=0.9, weight_decay=1e-4, nesterov=True)
    },
    "loss": {
        "type": "torch.nn.CrossEntropyLoss",
        "params": {}
    },
    "activation":
    "partial(torch.nn.functional.softmax,dim=-1)"
}
# ------------------------------------
#        EXAMPLE MODEL CONFIG
# ------------------------------------

a = 0.75
model_config = dict(features=None,
                    backbone={
                        "type":
                        "RawNet",
                        "trainable":
                        True,
                        "params":
                        dict(normalize_input=True,
                             init_conv_params=dict(
                                 in_channels=1,
                                 out_channels=int(a * 96),
                                 stride=3,
                                 kernel_size=3,
                                 padding=0,
                             ),
                             block_setup=[
                                 (int(a * 96), int(a * 128), True, 1),
                                 (int(a * 128), int(a * 128), True, 3),
                                 (int(a * 128), int(a * 160), True, 1),
                                 (int(a * 160), int(a * 160), True, 3),
                                 (int(a * 160), int(a * 192), True, 1),
                                 (int(a * 192), int(a * 192), True, 3),
                                 (int(a * 192), int(a * 256), True, 1),
                                 (int(a * 256), int(a * 256), True, 3),
                                 (int(a * 256), int(a * 288), True, 1),
                                 (int(a * 288), int(a * 288), True, 3),
                                 (int(a * 288), int(a * 288), True, 1),
                             ])
                    },
                    pooling={
                        "type": "StatsPooling1D",
                        "trainable": True,
                        "params": dict(mode="std")
                    },
                    cls_head={
                        "type":
                        "MultiTaskClassificationHead",
                        "trainable":
                        True,
                        "params":
                        dict(input_features_chan=int(a * 288) * 2,
                             head_setups={
                                 'spoofing': 2,
                                 'playback_device': 2,
                                 'recording_device': 3
                             },
                             head_hidden_layers=[
                                 (256, 0.0, None),
                             ])
                    })
