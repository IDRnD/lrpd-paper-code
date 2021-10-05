from typing import List
from pathlib import Path


def build_augmentation_config(noise_roots=[], snr_lims=(3., 15.), prob=0.5):
    augmentors = []
    for nr in noise_roots:
        augmentors.append(
            f"CustomNoises(noises_folder='{str(nr)}', snrs={str(snr_lims)}, prob=1.)")
    return f"OneOf([{','.join(augmentors)}],prob={prob})"


def make_data_config_antispoofing(dataset_setup, lrpd_root, asv17_root, noise_roots):
    train_dataset = []
    lrpd_root = str(lrpd_root)
    asv17_root = str(asv17_root)
    added_lrpd_human = False
    for ds_part in dataset_setup:
        if "lrpd" in ds_part:
            part = ds_part.split("_")[1]
            if not added_lrpd_human:
                train_dataset.append({
                    "parse_func": "parse_dir",
                    "args": {
                        "path": f"{lrpd_root}/source_trn",
                        "label": 0
                    }
                })
                added_lrpd_human = True
            train_dataset.append({
                "parse_func": "parse_dir",
                "args": {
                    "path": f"{lrpd_root}/trn_{part}",
                    "label": 1
                }
            })

        if "asv17" in ds_part:
            part = ds_part.split("_")[1]
            train_dataset.append({
                "parse_func": "parse_asv17",
                "args": {
                    "asv_spoof_root": asv17_root,
                    "part": part,
                    "return_as": "dict"
                }
            })

    dataset_config = {
        "train":
        train_dataset,
        "test": [
            {
                "test_name":
                "LRPD-val",
                "test_setup": [
                    {
                        "parse_func": "parse_dir",
                        "args": {
                            "path": f"{lrpd_root}/source_val",
                            "label": 0
                        }
                    },
                    {
                        "parse_func": "parse_dir",
                        "args": {
                            "path": f"{lrpd_root}/val_aparts",
                            "label": 1
                        }
                    },
                ],
            },
            {
                "test_name":
                "ASVSpoof2017-eval",
                "test_setup": [
                    {
                        "parse_func": "parse_asv17",
                        "args": {
                            "asv_spoof_root": asv17_root,
                            "part": "eval",
                            "return_as": "dict"
                        }
                    },
                ],
            },
            {
                "test_name":
                "ASVSpoof2017-dev",
                "test_setup": [
                    {
                        "parse_func": "parse_asv17",
                        "args": {
                            "asv_spoof_root": asv17_root,
                            "part": "dev",
                            "return_as": "dict"
                        }
                    },
                ],
            },
        ]
    }
    augmentation_setup = build_augmentation_config(noise_roots)
    data_config = {
        "utt_len_sec":
        3.0,
        "samplerate":
        16000,
        "valid_size":
        5000,
        "target_cls_ind":
        0,
        "parse_data_func":
        "v3",
        "augmentor": [{
            "target_classes": [0],
            "augmentor": augmentation_setup
        }, {
            "target_classes": [1],
            "augmentor": augmentation_setup
        }],
        "dataset":
        dataset_config,
    }
    return data_config


def make_data_config_device_detector(dataset_setup, lrpd_root, asv17_root, noise_roots):
    lrpd_root = str(lrpd_root)
    asv17_root = str(asv17_root)

    train_dataset = []
    for ds_part in dataset_setup:
        if "lrpd" in ds_part:
            part = ds_part.split("_")[1]
            train_dataset.append({
                "source_root":
                f"{lrpd_root}/source_trn",
                "parallel_roots":
                list(map(str, list(Path(f"{lrpd_root}/trn_{part}").glob("*/*/*/*/"))))
            })
    data_config = {
        "utt_len_sec": 2.0,
        "num_parallel": 4,
        "samplerate": 16000,
        "valid_size": 5000,
        "use_source": False,
        "augmentor": {
            "*": build_augmentation_config(noise_roots)
        },
        "dataset": {
            "train":
            train_dataset,
            "test": [
                {
                    "test_name": "LRPD-val",
                    "samples_per_domain": 3000,
                    "test_setup": {
                        "source_root":
                        f"{lrpd_root}/source_val",
                        "parallel_roots":
                        list(map(str, list(Path(f"{lrpd_root}/val_aparts").glob("*/*/*/*/"))))
                    },
                },
            ]
        }
    }
    return data_config


def configure(task: str, lrpd_root: str, asv17_root: str, dataset_setup: List[str],
              noise_roots: List[str], model_config: dict):
    if lrpd_root is not None:
        lrpd_root = Path(lrpd_root)
    else:
        raise Exception(
            "Path to LRPD is None, you should configure data.yml before running training")
    if asv17_root is not None:
        asv17_root = Path(asv17_root)
    else:
        raise Exception(
            "Path to ASVSpoof2017 dataset is None, you should configure data.yml before running training"
        )
    noise_roots = [Path(nr) for nr in noise_roots]
    # Validate args:
    assert lrpd_root.exists()
    assert asv17_root.exists()
    assert all(map(lambda p: p.exists(), noise_roots))

    VALID_DATASETS = {
        "lrpd_office",
        "lrpd_aparts",
        "asv17_train",
        "asv17_eval",
        "asv17_dev",
    }
    assert set(dataset_setup) <= VALID_DATASETS

    # Generate data config for specific task
    data_config = {
        "antispoofing": make_data_config_antispoofing,
        "device_detector": make_data_config_device_detector
    }[task](dataset_setup, lrpd_root, asv17_root, noise_roots)

    # Modify number of classes for each head of device detector based
    # on dataset used (number of recording and playback devices presenteed)
    if task == "device_detector":
        rec_devs = []
        pb_devs = []
        for data_part in data_config["dataset"]["train"]:
            for pr in data_part["parallel_roots"]:
                p = Path(pr)
                rec_devs.append(p.parts[-3].lower().replace(" ", ""))
                pb_devs.append(p.parts[-2].lower().replace(" ", ""))
        pb_devs = set(pb_devs)
        rec_devs = set(rec_devs)

        num_playback = len(pb_devs)
        num_recording = len(rec_devs)

        model_config['cls_head'] = {
            "type":
            "MultiTaskClassificationHead",
            "trainable":
            True,
            "params":
            dict(
                input_features_chan=model_config['backbone']["params"]['block_setup'][-1][1] * 2,
                head_setups={
                    'playback_device': (num_playback, 0.25),
                    'recording_device': (num_recording, 0.25)  # 16
                },
                head_hidden_layers=[
                    (256, 0.0, "ReLU"),
                    (128, 0.0, "ReLU"),
                ])
        }
    return data_config, model_config