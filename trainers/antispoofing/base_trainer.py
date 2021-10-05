import json
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# adding pipeline root to sys.path
pipeline_root = Path(__file__).resolve().parents[2]
if str(pipeline_root) not in sys.path:
    print(f"Adding pipeline root in sys.path: {pipeline_root}")
    sys.path.append(str(pipeline_root))

import datautils.parsing.antispoofing as parsing
from datautils.audio_processing.augmentors import *
from datautils.dataset import Dataset, simple_collate_func, transforms
from datautils.parsing.antispoofing import (flatten_data_dict, parse_data_config,
                                            parse_data_config_v2, subset_test_data)
from datautils.parsing.common import (flatten_shallow_file_dict, get_shallow_file_dict,
                                      keys_to_ints, merge_dicts,
                                      test_train_split_shallow_file_dict)
from models.model_builders import AudioClassificationModel
from utils.lr_scheduler import (generate_lr_array, lr_array_to_sched_func)
from utils.metrics import EER


DEFAULT_SEED = 23


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class AntispoofingClassifier(pl.LightningModule):
    def __init__(
        self,
        model_config: Dict = {},
        train_config: Dict = {},
        data_config: Dict = {},
        nnet: nn.Module = None,
        spec_augs: nn.Module = None,
        seed: int = DEFAULT_SEED,
        workers: int = 16,
    ):
        super(AntispoofingClassifier, self).__init__()
        set_random_seed(seed)
        self.output_dir = None
        self.workers = workers
        self.data_config = data_config
        self.model_config = model_config
        self.train_config = train_config

        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.steps_per_epoch = train_config["steps_per_epoch"]

        self.result_df = pd.DataFrame()
        self.generate_lr_array()
        if nnet is None:
            self.nnet = AudioClassificationModel(**model_config)
            if spec_augs is not None:
                features = self.nnet.features
                spec_augs = spec_augs if spec_augs is not None else lambda x: x
                self.nnet.spec_augs = None
                self.nnet.features = None
                self.transforms = (transforms.Lambda(lambda np_audio: spec_augs(
                    features((torch.from_numpy(np_audio)[None, None, :])))[0]),
                                   transforms.Lambda(lambda np_audio: features(
                                       (torch.from_numpy(np_audio)[None, None, :]))[0]))
            else:
                self.transforms = None
        else:
            self.transforms = None
            self.nnet = nnet
        loss_conf = train_config["loss"]
        self.loss = eval(loss_conf["type"])(**loss_conf["params"])
        self.activation = eval(train_config["activation"])

    def setup(self, *args, **kwargs):
        # Dump configs
        self.output_dir = Path(self.trainer.logger.log_dir).resolve()
        print(f"output_dir : {self.output_dir}")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        (self.output_dir / "val_data").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "data_config.json").write_text(json.dumps(self.data_config, indent=4))
        (self.output_dir / "model_config.json").write_text(json.dumps(self.model_config, indent=4))
        (self.output_dir / "train_config.json").write_text(json.dumps(self.train_config, indent=4))

    def prepare_data(self):
        # Load augmentor
        if "augmentor" in self.data_config:
            self.augmentor = build_augmentor_from_config(self.data_config["augmentor"])
        else:
            self.augmentor = None

        # Load data:
        print(f"exclude_substrings : {self.data_config['dataset'].get('exclude',[])}")
        parser_version = self.data_config.get("parse_data_func", "v1")
        if parser_version in ["v1", "v2"]:
            parse_data_func = {"v1": parse_data_config, "v2": parse_data_config_v2}[parser_version]
            train_data = parse_data_func(self.data_config["dataset"]["train"],
                                         ext="wav",
                                         exclude_substrings=self.data_config["dataset"].get(
                                             "exclude", []))
            train_data, val_data = test_train_split_shallow_file_dict(
                train_data, test_size=self.data_config["valid_size"])
            val_data = flatten_data_dict(val_data)
            print(f"val_data : {len(val_data)}")

            self.train_data = train_data

            self.val_data = {"validation": val_data}
            for test_setup in self.data_config["dataset"]["test"]:
                self.val_data[test_setup["test_name"]] = subset_test_data(
                    test_setup["test_setup"],
                    ext="wav",
                    domain_subset_size=test_setup["samples_per_domain"])

        elif parser_version in ["v3"]:
            train_data = defaultdict(list)
            for data_part in self.data_config["dataset"]["train"]:
                parse_func_name = data_part["parse_func"]
                parse_func = eval(f"parsing.{parse_func_name}")
                data = parse_func(**data_part["args"])
                for label, file_list in data.items():
                    print(
                        f"parse_func_name : {parse_func_name}, num_files : {len(file_list)}, label : {label}"
                    )
                    train_data[label].append(file_list)
            train_data, val_data = test_train_split_shallow_file_dict(
                train_data, test_size=self.data_config["valid_size"])

            self.train_data = train_data
            self.val_data = {"validation": flatten_data_dict(val_data)}
            for test_setup in self.data_config["dataset"]["test"]:
                test_data = defaultdict(list)
                for data_part in test_setup["test_setup"]:
                    parse_func_name = data_part["parse_func"]
                    parse_func = eval(f"parsing.{parse_func_name}")
                    data = parse_func(**data_part["args"])
                    for label, file_list in data.items():
                        print(
                            f"parse_func_name : {parse_func_name}, num_files : {len(file_list)}, label : {label}"
                        )
                        test_data[label].append(file_list)
                self.val_data[test_setup["test_name"]] = flatten_data_dict(test_data)

    def generate_lr_array(self):
        # Generate lr-array
        self.lr_array = np.concatenate(
            [generate_lr_array(**lr_conf) for lr_conf in self.train_config["lr_scheduler"]])

    def forward(self, x):
        return self.nnet(x)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        # OPTIONAL
        x, labels = batch
        pred = self.nnet(x)
        loss = self.loss(pred, labels)

        return {'loss': loss, 'labels': labels, 'pred': self.activation(pred)}

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, labels = batch
        pred = self.nnet(x)
        loss = self.loss(pred, labels)
        acc = torch.mean(torch.eq(torch.argmax(pred, dim=-1), labels).to(torch.float))
        self.log('loss',
                 loss,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=False,
                 add_dataloader_idx=False)
        self.log('acc',
                 acc,
                 prog_bar=True,
                 logger=True,
                 on_step=True,
                 on_epoch=False,
                 add_dataloader_idx=False)
        return {'loss': loss, 'log': {'loss': loss, 'acc': acc}}

    def validation_epoch_end(self, outputs):
        # To DO remove from here
        # OPTIONAL
        df_update_data = {"epoch": [self.current_epoch]}
        losses, accs, eers = {}, {}, {}
        for val_set_name, val_set_results in zip(self.val_data.keys(), outputs):
            predictions = torch.cat([x['pred'] for x in val_set_results]).cpu().numpy()
            labels = torch.cat([x['labels'] for x in val_set_results]).cpu().numpy()
            scores_path = str(
                self.output_dir / "val_data" /
                f"{val_set_name}-epoch={self.current_epoch}_step={self.global_step}-predictions.npy"
            )
            np.save(scores_path, {"predictions": predictions, "labels": labels})
            num_classes = predictions.shape[1]
            ohes = torch.eye(num_classes, device=None, dtype=torch.float)[labels]

            tar_cls_ind = self.data_config["target_cls_ind"]
            eer, eer_thr = EER(ohes[:, tar_cls_ind], predictions[:, tar_cls_ind])

            avg_val_loss = torch.stack([x['loss'] for x in val_set_results]).mean()
            avg_val_acc = np.mean(np.equal(np.argmax(predictions, axis=-1), labels))

            df_update_data[f"loss-{val_set_name}"] = avg_val_loss.item()
            df_update_data[f"acc-{val_set_name}"] = avg_val_acc.item()
            df_update_data[f"eer-{val_set_name}"] = eer

            losses[val_set_name] = df_update_data[f"loss-{val_set_name}"]
            accs[val_set_name] = df_update_data[f"acc-{val_set_name}"]
            eers[val_set_name] = df_update_data[f"eer-{val_set_name}"]
            # Dump validation results to disk
            # val_data = {
            #     "predictions" : predictions,
            #     "labels" : labels
            # }
            # val_save_dir = Path(self.logger.log_dir)/"validation"
            # val_save_dir.mkdir(exist_ok=True,parents=True)
            # np.save(val_save_dir/f"val-data-ep={self.current_epoch:03}.npy", val_data)
        if self.global_step > 0:
            self.result_df = self.result_df.append(pd.DataFrame(df_update_data))
            self.result_df.to_csv(str(Path(self.trainer.logger.log_dir) / "logs.csv"), index=False)
        self.log("dummy_loss",
                 1,
                 prog_bar=False,
                 logger=True,
                 on_step=False,
                 on_epoch=True,
                 add_dataloader_idx=False)
        self.log("val_loss",
                 losses["validation"],
                 prog_bar=False,
                 logger=True,
                 on_step=False,
                 on_epoch=True,
                 add_dataloader_idx=False)
        self.log("val_losses",
                 losses,
                 prog_bar=False,
                 logger=True,
                 on_step=False,
                 on_epoch=True,
                 add_dataloader_idx=False)
        self.log("val_accs",
                 accs,
                 prog_bar=False,
                 logger=True,
                 on_step=False,
                 on_epoch=True,
                 add_dataloader_idx=False)
        self.log("val_eers",
                 eers,
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
            'scheduler': torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda it: self.lr_array[it] / self.lr_array[0]),  # The LR schduler
            'interval': 'step',  # The unit of the scheduler's step size
            'frequency': 1,  # The frequency of the scheduler
            'reduce_on_plateau': False,  # For ReduceLROnPlateau scheduler
            'monitor': 'val_eer'  # Metric to monitor
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        # REQUIRED
        if self.transforms is not None:
            train_transform, val_transform = self.transforms
        else:
            train_transform = val_transform = transforms.Lambda(
                lambda np_audio: torch.from_numpy(np_audio)[None, :])
        train_dataset = Dataset(
            data=self.train_data,
            size=self.steps_per_epoch * self.batch_size,
            augmentor=self.augmentor,
            transform=train_transform,
            utt_len_sec=self.data_config["utt_len_sec"],
            samplerate=self.data_config["samplerate"],
            convert_to_ohe=False,
        )
        return DataLoader(dataset=train_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          sampler=None,
                          batch_sampler=None,
                          num_workers=self.workers,
                          collate_fn=simple_collate_func,
                          pin_memory=True,
                          drop_last=False,
                          timeout=0,
                          worker_init_fn=None,
                          multiprocessing_context=None)

    def val_dataloader(self):
        for test_name, test_data in self.val_data.items():
            print(f"self.output_dir : {self.output_dir}")
            if self.output_dir is not None:
                target_path = self.output_dir / f"{test_name}-test-data.json"
                if not target_path.exists():
                    target_path.write_text(json.dumps([[str(p), l] for p, l in test_data]))
        # OPTIONAL
        val_dls = []
        if self.transforms is not None:
            train_transform, val_transform = self.transforms
        else:
            train_transform = val_transform = transforms.Lambda(
                lambda np_audio: torch.from_numpy(np_audio)[None, :])
        for val_set_name in self.val_data.keys():
            val_dataset = Dataset(
                data=self.val_data[val_set_name],
                # size=self.steps_per_epoch * self.batch_size,
                augmentor=DummyAugmentor(),
                transform=val_transform,
                utt_len_sec=self.data_config["utt_len_sec"],
                samplerate=self.data_config["samplerate"],
                convert_to_ohe=False)
            val_dl = DataLoader(dataset=val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                sampler=None,
                                batch_sampler=None,
                                num_workers=self.workers,
                                collate_fn=simple_collate_func,
                                pin_memory=True,
                                drop_last=True,
                                timeout=0,
                                worker_init_fn=None,
                                multiprocessing_context=None)
            val_dls.append(val_dl)
        print(f"VAL_DLS ORDER : {self.val_data.keys()}")
        return val_dls


# ------------------------------------
#        EXAMPLE DATA CONFIG
# ------------------------------------

'''
TRAIN_ROOT = "/media/ssdraid0cgpu01/data/antispoofing/16K/Train"
TESTS_ROOT = "/media/ssdraid0cgpu01/data/antispoofing/16K/tests/"
AUGS_ROOT = "/media/ssdraid0cgpu01/data/augmentations/"

replay_train_datasets = {
    f"{TRAIN_ROOT}/Human": 0,
    #   f"{TRAIN_ROOT}/Human_clean" : 0,
    f"{TRAIN_ROOT}/Human_source_InternalReplayTrain2021": 0,
    f"{TRAIN_ROOT}/Reply": 1,
    f"{TRAIN_ROOT}/InternalReplayTrain2021": 1,
}

data_config = {
    "utt_len_sec":
    3.0,
    "samplerate":
    16000,
    "valid_size":
    5000,
    "target_cls_ind":
    0,
    "augmentor": [{
        "target_classes": [0],
        "augmentor":
        f"""Sequential([ 
                CustomNoises(noises_folder="/media/ssdraid0cgpu01/data/augmentations/InOfficeAcousticRoomSilence", snrs=(3., 12.), prob=0.7),
                OneOf([
                    CustomNoises(noises_folder="{AUGS_ROOT}/MUSAN_15_sec_cut-16K/", snrs=(3., 12.), prob=1.),
                    CustomNoises(noises_folder="{AUGS_ROOT}/DEMAND_15_sec_cut-16K/", snrs=(3., 12.), prob=1.),
                    CustomNoises(noises_folder="{AUGS_ROOT}/DECASE_2017_Task3_Acoustic_Scenes-16K/", snrs=(3., 12.), prob=1.),
                ],prob=0.7),
            ],prob=1.0)""",
    }, {
        "target_classes": [1],
        "augmentor":
        f"""Sequential([ 
                OneOf([
                    CustomNoises(noises_folder="{AUGS_ROOT}/MUSAN_15_sec_cut-16K/", snrs=(4., 12.), prob=1.),
                    CustomNoises(noises_folder="{AUGS_ROOT}/DEMAND_15_sec_cut-16K/", snrs=(4., 12.), prob=1.),
                    CustomNoises(noises_folder="{AUGS_ROOT}/DECASE_2017_Task3_Acoustic_Scenes-16K/", snrs=(4., 12.), prob=1.),
                ],prob=0.4),
            ],prob=1.0)""",
    }],
    "dataset": {
        "train":
        replay_train_datasets,
        "test": [{
            "test_name": "Test4.0",
            "samples_per_domain": 500,
            "test_setup": {
                f"{TESTS_ROOT}/Test4.0/Human": 0,
                f"{TESTS_ROOT}/Test4.0/Reply": 1,
            },
        }, {
            "test_name": "LargeReplayTest_v1.0",
            "samples_per_domain": 500,
            "test_setup": {
                f"{TESTS_ROOT}/LargeReplayTest_v1.0/Human": 0,
                f"{TESTS_ROOT}/LargeReplayTest_v1.0/Replay": 1,
            },
        }]
    }
}
'''

# ------------------------------------
#        EXAMPLE TRAIN CONFIG
# ------------------------------------

'''
num_epochs = 40
steps_per_epoch = 3000
train_config = {
    "epochs":
    num_epochs,
    "batch_size":
    128,
    "steps_per_epoch":
    steps_per_epoch,
    "lr_scheduler": [
        dict(scheduler_type="JigsawLog",
             base_lr=1e-5,
             max_lr=5e-3,
             gamma=0.5,
             period_init_len=7,
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
'''
