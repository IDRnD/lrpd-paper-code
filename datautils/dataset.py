import traceback
from typing import Dict, List, Tuple, Union

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


def simple_collate_func(batch):
    xs, ys = [], []
    for sample in batch:
        if sample is not None:
            x, y = sample
            xs.append(x)
            ys.append(y)

    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return xs, ys


class Dataset(torch.utils.data.Dataset):
    # Train dataset for training speaker verification
    # Samples random utterance from random speaker
    def __init__(
        self,
        data: Union[Dict[int, List[List[Path]]], List[Tuple[Path, int]]],
        size: int = None,  # = steps_per_epoch * batch_size
        augmentor: BaseAugmentor = DummyAugmentor(),
        transform: nn.Module = transforms.Lambda(
            lambda np_audio: torch.from_numpy(np_audio)[None, :]),
        utt_len_sec: float = 3.,
        samplerate: int = 16000,
        convert_to_ohe: bool = False,
        mode: str = "torch",
        debug: bool = True,
    ):
        """
        data dict example = {
            0 : [
                [Path,Path,...], # Subdomain files
                [Path,Path,...], # Subdomain files
            ],
            1: [
                [Path, Path, ...], # Subdomain files
                [Path, Path, ...], # Subdomain files
            ],
        }
        """
        # data might be dict {spk_ind : List[spk_utts]} or list of tuples : (utt_path, spk_ind)
        # spec nn.Module might be passed into Dataset to calculate features on CPU (for TPU for example)
        assert mode in ["keras", "torch"]
        print(f"created ds with : {len(data)}")
        self.data = data
        self.mode = mode
        self.debug = debug
        self.transform = transform
        if isinstance(augmentor, BaseAugmentor):
            # Wrap to handle self.augmentor(samples,cls_ind) signature
            self.augmentor = lambda samples, cls_ind: augmentor(samples)
        else:
            self.augmentor = augmentor
        self.convert_to_ohe = convert_to_ohe
        # Initial min/max output_len in samples
        self.utt_len_samples = int(utt_len_sec * samplerate)
        self.load_audio = get_audio_loader(samplerate=samplerate)

        if isinstance(self.data, dict):
            self.random = True
            self.cls_inds = list(self.data.keys())
            assert size is not None
            self.num_classes = len(self.cls_inds)
            self.size = size
        elif isinstance(self.data, list):
            self.random = False
            self.num_classes = len(set([e[1] for e in self.data]))
            self.size = len(self.data)
        else:
            raise Exception(
                f"Data must be of dict or list type not {type(data)}")

    def get_utterance(self, index):
        if isinstance(self.data, list):
            # Sequence sampling dataset for testing:
            utt_path, cls_ind = self.data[index]
        elif isinstance(self.data, dict):
            # Random sampling dataset for training:
            # Generate random in case its dict
            # Makes speaker-uniform random sampling of utterances
            cls_ind = random.choice(self.cls_inds)
            rnd_file_or_subdomain_filelist = random.choice(self.data[cls_ind])
            if isinstance(rnd_file_or_subdomain_filelist, list):
                utt_path = random.choice(rnd_file_or_subdomain_filelist)
            elif isinstance(
                    rnd_file_or_subdomain_filelist, Path) or isinstance(
                        rnd_file_or_subdomain_filelist, str):
                utt_path = rnd_file_or_subdomain_filelist
            else:
                raise NotImplemented()
        return utt_path, cls_ind

    def __getitem__(self, index):
        try:
            # Pick random speaker, random utterance
            utt_path, cls_ind = self.get_utterance(index)
            return self.load_sample(utt_path, cls_ind)
        except IndexError as ex:
            raise ex
        except Exception as ex:
            if self.debug:
                print(f"Error on file : {utt_path} | {ex}")
                traceback.print_exc()
            return None

    def load_sample(self, utt_path, cls_ind):
        samples = normalize(self.load_audio(utt_path))
        samples = normalize(self.augmentor(samples, cls_ind))
        samples = extend_signal(samples,
                                target_len=self.utt_len_samples,
                                repeat=True)
        if self.random:
            samples = cut_signals(samples,
                                  target_len=self.utt_len_samples,
                                  random=True)
        else:
            samples = cut_signals(samples,
                                  target_len=self.utt_len_samples,
                                  random=False)

        if np.any(np.isnan(samples)):
            raise Exception("There are NaNs in signal!")

        x = self.transform(samples)

        y = cls_ind
        if isinstance(cls_ind, int):
            if self.convert_to_ohe:
                y = indices_to_one_hot([y], nb_classes=self.num_classes)[0]
            y = torch.from_numpy(np.array(y))
        return x.float(), y

    def __len__(self):
        return self.size
