import argparse
import json
import os
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning import loggers as pl_loggers

from trainers.antispoofing.base_trainer import AntispoofingClassifier
from trainers.antispoofing.parallel_trainer import \
    AntispoofingClassifierParallel
from utils.configure import configure


def run_training(data_config: dict,
                 model_config: dict,
                 train_config: dict,
                 Trainer: pl.LightningModule,
                 exp_dir: str,
                 gpus_ids: str = "0"):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_ids
    num_gpus = len(gpus_ids.split(","))

    model = Trainer(
        model_config=model_config,
        train_config=train_config,
        data_config=data_config,
    )
    save_dir = str(exp_dir)
    trainer = pl.Trainer(
        gpus=num_gpus,
        logger=pl_loggers.TensorBoardLogger(save_dir),
        default_root_dir=save_dir,
        prepare_data_per_node=False,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                save_weights_only=True,
                save_top_k=-1,  # Save all checkpoints
                verbose=True,
                monitor="dummy_loss",
                mode="min"),
            pl.callbacks.LearningRateMonitor(),
            pl.callbacks.GPUStatsMonitor()
        ],
        max_epochs=train_config["epochs"],
        min_epochs=train_config["epochs"],
        accumulate_grad_batches=1,
        precision=32,
        amp_level='O2',
    )

    trainer.fit(model)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    class SplitArgs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, values.split(','))

    parser.add_argument("task", type=str, help="antispoofing or device_detector")
    parser.add_argument(
        "dataset_setup",
        type=str,
        action=SplitArgs,
        help="Dataset setup in form 'antispoofing lrpd_office,lrpd_aparts,asv17_train'")
    parser.add_argument("--train_config",
                        type=Path,
                        help="Path to train config serialized into JSON",
                        default=root / "configs/common/train_config.json")
    parser.add_argument("--model_config",
                        type=Path,
                        help="Path to model config serialized into JSON",
                        default=root / "configs/common/model_config.json")
    parser.add_argument("--gpus",
                        type=str,
                        help="IDs of GPUs to train on. For example: 0,1",
                        default="0")
    parser.add_argument("--exp_dir", type=Path, help="Path to experiment folder", default=None)

    args = parser.parse_args()
    data = yaml.safe_load((root / 'data.yml').read_text())
    exp_dir = args.exp_dir if args.exp_dir is not None else root / "experiments" / args.task / "_".join(args.dataset_setup)

    print('Parsed args:', args, sep='\n')
    print('Data config:', data, sep='\n')

    train_config = json.loads(args.train_config.read_text())
    model_config = json.loads(args.model_config.read_text())
    data_config, model_config = configure(task=args.task,
                                          lrpd_root=data['lrpd_root'],
                                          asv17_root=data['asv17_root'],
                                          noise_roots=data['noise_roots'],
                                          dataset_setup=args.dataset_setup,
                                          model_config=model_config)

    Trainer = {
        "antispoofing": AntispoofingClassifier,
        "device_detector": AntispoofingClassifierParallel
    }[args.task]

    run_training(data_config=data_config,
                 model_config=model_config,
                 train_config=train_config,
                 Trainer=Trainer,
                 exp_dir=exp_dir,
                 gpus_ids=args.gpus)
