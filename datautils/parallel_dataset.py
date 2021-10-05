import traceback
from collections import defaultdict
from typing import Dict, List, Callable

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from .audio_processing.augmentors import *
from .audio_processing.common import *


def indices_to_one_hot(cls_indeces, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(cls_indeces).reshape(-1)
    return np.eye(nb_classes)[targets]


def parallel_collate_func(batch):
    batch_dict = defaultdict(list)
    for sample in batch:
        if sample is None:
            continue
        for key in sample.keys():
            if isinstance(sample[key], np.ndarray):
                sample[key] = torch.from_numpy(sample[key])
            batch_dict[key].append(sample[key])
    for key in batch_dict.keys():
        batch_dict[key] = torch.cat(batch_dict[key], dim=0)
    return batch_dict


class ParallelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: List[List[Dict]],
        class_ind_maps: Dict,
        augmentor: Callable = lambda samples, classes: samples,
        transform: nn.Module = transforms.Lambda(
            lambda np_audio: torch.from_numpy(np_audio)[:, None, :]),
        num_parallel_versions=None,
        add_zero_parallel=False,
        utt_len_sec: float = 3.,
        size: int = None,
        samplerate: int = 16000,
        convert_to_ohe: bool = False,
        mode: str = "torch",
        debug: bool = True,
    ):
        """
        --------------------------
        class_ind_maps example = {
            "playback_device" : {
                'jbl_flip_4' : 0, 
                'srs-xb12' : 1
            },
            "recording_device" : {
                'samsung_s8+' : 0, 
                'iphone_7' : 1, 
                'honor_10x_lite' : 2, 
                'iphone_xr' : 3
            }
        },
        --------------------------
        data dict example = {
            source Path : {
                "source" : Path
                "parallel" : [
                    {
                        "path" : Path,
                        "playback_device" : str
                        "recording_device" : str
                        other aux fields
                    }
                ]
            },
            source Path : {
                "source" : Path
                "parallel" : [
                    {
                        "path" : Path,
                        "playback_device" : str
                        "recording_device" : str
                        other aux fields
                    }
                ]
            },
            ...
        }
        """
        # data might be dict {spk_ind : List[spk_utts]} or list of tuples : (utt_path, spk_ind)
        # spec nn.Module might be passed into Dataset to calculate features on CPU (for TPU for example)
        assert mode in ["keras", "torch"]
        print(f"created ds with : {len(data)}")
        self.data = data
        self.size = size
        self.mode = mode
        self.debug = debug
        self.transform = transform
        self.class_ind_maps = class_ind_maps
        self.add_zero_parallel = add_zero_parallel
        self.random = num_parallel_versions is not None
        self.num_parallel_versions = num_parallel_versions
        # if isinstance(augmentor,BaseAugmentor):
        #     # Wrap to handle self.augmentor(samples,cls_ind) signature
        #     self.augmentor = lambda samples,cls_ind : augmentor(samples)
        # else:
        self.augmentor = augmentor
        self.convert_to_ohe = convert_to_ohe
        # Initial min/max output_len in samples
        self.utt_len_samples = int(utt_len_sec * samplerate)
        self.load_audio = get_audio_loader(samplerate=samplerate, raise_error=True)

    def get_utterances(self, index):
        if self.num_parallel_versions is None:
            # Deterministice batch sampling for test
            # will sample source + all parallel
            parallel_data_patch = self.data[index]
            parallel_recs = parallel_data_patch["parallel"]
            rec_inds = np.arange(len(parallel_recs))
        else:
            # Undeterministice batch sampling for train
            # will sample source + num_parallel_versions picked at random
            parallel_data_patch = random.choice(self.data)
            parallel_recs = parallel_data_patch["parallel"]
            rec_inds = np.random.permutation(np.arange(
                len(parallel_recs)))[:min(len(parallel_recs), self.num_parallel_versions)]
            if self.add_zero_parallel:
                if 0 not in rec_inds:
                    rec_inds[0] = 0

        utts = []
        for rec_ind in rec_inds:
            prec = dict(parallel_recs[rec_ind])
            # Remove unused keys from data dict
            for k in list(prec.keys()):
                if k not in (["path"] + list(self.class_ind_maps.keys())):
                    del prec[k]
            # Convert class names to class inds
            for cls_task_name in self.class_ind_maps.keys():
                prec[cls_task_name] = self.class_ind_maps[cls_task_name][prec[cls_task_name]]
            utts.append(prec)
        return utts

    def __getitem__(self, index):
        try:
            # Pick random speaker, random utterance
            utts = self.get_utterances(index)
            return self.load_sample(utts)
        except IndexError as ex:
            raise ex
        except Exception as ex:
            if self.debug:
                print(f"Error on file : {utts} | {ex}")
                traceback.print_exc()
            return None

    def load_sample(self, utts):
        data_patch = defaultdict(list)
        for utt in utts:
            audio_path = utt["path"]
            del utt["path"]
            samples = normalize(self.load_audio(audio_path))
            samples = normalize(self.augmentor(samples, utt))
            samples = extend_signal(samples, target_len=self.utt_len_samples, repeat=True)
            data_patch["x"].append(samples)
            for cls_task_name in utt.keys():
                data_patch[cls_task_name].append(utt[cls_task_name])

        min_len = min([parallel_samples.shape[-1] for parallel_samples in data_patch["x"]])
        samples = np.array([
            data_patch["x"][i][..., :min_len] for i in range(len(data_patch["x"]))
        ])  # shape : [num_utts, self.utt_len_samples]
        # Cut all with same cut
        if self.random:
            samples = cut_signals(samples, target_len=self.utt_len_samples, random=True)
        else:
            samples = cut_signals(samples, target_len=self.utt_len_samples, random=False)

        if np.any(np.isnan(samples)):
            raise Exception("There are NaNs in signal!")

        data_patch["x"] = self.transform(samples)

        for cls_task_name in utt.keys():
            data_patch[cls_task_name] = np.array(data_patch[cls_task_name], dtype=np.int)
            if self.convert_to_ohe:
                data_patch[cls_task_name] = indices_to_one_hot(
                    data_patch[cls_task_name], len(self.class_ind_maps[cls_task_name]))
            data_patch[cls_task_name] = torch.from_numpy(data_patch[cls_task_name])
        return data_patch

    def __len__(self):
        if self.size is not None:
            return self.size
        else:
            return len(self.data)
