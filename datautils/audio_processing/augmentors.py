import random
from abc import abstractmethod
from collections import defaultdict
from pprint import pprint
from typing import List, Tuple, Union

import numpy as np
from scipy.signal import fftconvolve

from .common import *


def random_int(val : Union[int,Tuple[int,int]]):
    if isinstance(val,tuple):
        low, high = val
        assert low < high
        return np.random.randint(low, high)
    elif isinstance(val, int):
        return val
    else:
        raise TypeError()

def random_float(val : Union[float,Tuple[float,float]]):
    if isinstance(val,tuple):
        low, high = val
        assert low < high
        return low + np.random.rand() * (high - low)
    elif isinstance(val, float):
        return val
    else:
        raise TypeError()

class BaseAugmentor(object):
    def __init__(self, prob = 0.5):
        self.prob = prob

    @abstractmethod
    def __apply__(self, signal : np.ndarray):
        pass

    def __call__(self, signal : np.ndarray):
        if np.random.rand() <= self.prob:
            return self.__apply__(signal)
        else:
            return signal

class DummyAugmentor(BaseAugmentor):
    def __apply__(self, signal : np.ndarray):
        return signal
        
# ------------------------------------
#            COMPOSE AUGS
# ------------------------------------    

class OneOf(BaseAugmentor):
    def __init__(self, augmentors : List[BaseAugmentor], prob = 0.5):
        super(OneOf,self).__init__(prob)
        self.augmentors = augmentors

    def __apply__(self, signal : np.ndarray):
        aug_ind = np.random.randint(0,len(self.augmentors))
        return self.augmentors[aug_ind](signal)
    
    def __str__(self):
        aug_str = '\n\t'.join([f"{a}," for a in self.augmentors])
        return f"<OneOf([{aug_str}],prob={self.prob})>"
    
    def __repr__(self):
        return str(self)

class Sequential(BaseAugmentor):
    def __init__(self, augmentors : List[BaseAugmentor], prob = 0.5):
        super(Sequential,self).__init__(prob)
        self.augmentors = augmentors

    def __apply__(self, signal : np.ndarray):
        augmented = signal
        for aug in self.augmentors:
            augmented = aug(augmented)
        return augmented
    
    def __str__(self):
        aug_str = '\n\t'.join([f"{a}," for a in self.augmentors])
        return f"<Sequential([{aug_str}],prob={self.prob})>"

    def __repr__(self):
        return str(self)
    
# ------------------------------------
#             NOISE AUGS
# ------------------------------------ 

class CustomNoises(BaseAugmentor):
    def __init__(self, 
                 noises_folder : str,
                 samplerate : int = 16000,
                 snrs = (6.,30.),
                 prob : float = 0.5):
        super(CustomNoises,self).__init__(prob=prob)
        self.snrs = snrs
        self.load_audio = get_audio_loader(samplerate=samplerate)
        self.noised_pathes = list(Path(noises_folder).glob("**/*.wav"))
        # Freeze main signal noise function
        print(f"Custom noizes : {len(self.noised_pathes)}")
        assert len(self.noised_pathes) > 0, noises_folder

    def __load_random_noize__(self):
        rnd_noise_path = random.choice(self.noised_pathes)
        random_noise = self.load_audio(rnd_noise_path)
        return random_noise
            
    def __apply__(self, signal: np.ndarray):
        return mix_single_signal(signal,noise=self.__load_random_noize__(), snr=random_float(self.snrs), noise_cut_random=True)

class Reverb(BaseAugmentor):
    def __init__(self, 
            rirs_folder : str,
            prob : float = 0.5, 
            rirs_extension : str ="wav",
            samplerate : int = 16000):
        super(Reverb,self).__init__(prob=prob)
        self.rirs_pathes = list(Path(rirs_folder).resolve().glob(f"**/*.{rirs_extension}"))
        self.load_audio = get_audio_loader(samplerate=samplerate)
        assert len(self.rirs_pathes) > 0
        print("RIRs number : {}".format(len(self.rirs_pathes)))

    def __apply__(self, signal : np.ndarray):
        random_rir_fp = random.choice(self.rirs_pathes)
        random_rir = self.load_audio(random_rir_fp)
        reverb_signal = fftconvolve(signal, random_rir, mode="same")
        return reverb_signal[:signal.shape[0]]

# ------------------------------------
#             OTHER AUGS
# ------------------------------------ 

class BandPassFilter(BaseAugmentor):
    def __init__(self,
                 high_cut : Union[float,Tuple[float,float]] = (2000.,3500.),
                 low_cut : Union[float,Tuple[float,float]] = (20.,25.),
                 samplerate : int = 16000,
                 prob : float = 0.5,
                 ):
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.samplerate = samplerate
        super(BandPassFilter,self).__init__(prob=prob)

    def __apply__(self, signal : np.ndarray):
        signal = normalize(signal)
        high_freq = random_float(self.high_cut)
        low_freq = random_float(self.low_cut)
        print(f"{low_freq}, {high_freq}")
        signal = butter_highpass_filter(butter_lowpass_filter(signal, cutOff=high_freq, fs=self.samplerate, order=10), cutOff=low_freq, fs=self.samplerate, order=10)
        # signal = butter_bandpass_filter(signal,random_float(self.low_cut),random_float(self.high_cut),fs=self.samplerate,order=5)
        return signal

class LowPassFilter(BaseAugmentor):
    def __init__(self,
                 high_cut : Union[float,Tuple[float,float]] = (2000.,3500.),
                 samplerate : int = 16000,
                 prob : float = 0.5,
                 ):
        self.high_cut = high_cut
        self.samplerate = samplerate
        super(LowPassFilter,self).__init__(prob=prob)

    def __apply__(self, signal : np.ndarray):
        signal = normalize(signal)
        high_freq = random_float(self.high_cut)
        signal = butter_lowpass_filter(signal, cutOff=high_freq, fs=self.samplerate, order=10)
        return signal

# ------------------------------------
#             UTIL FUNCS
# ------------------------------------ 

def build_augmentor_multihead(augmentor_setup):
    pprint(augmentor_setup)

    def augmentor(utt, label):
        if label in augmentor_setup:
            return augmentor_setup[label](utt)
        elif "*" in augmentor_setup:
            return augmentor_setup["*"](utt)

    return augmentor

def build_augmentor_multihead_from_config(conf):
    """Class specific augmentations"""
    augmentor_setup = defaultdict(list)
    for aug_setup in conf:
        augmentor = aug_setup["augmentor"]
        for cls_ind in aug_setup["target_classes"]:
            augmentor_setup[cls_ind].append(eval(augmentor))
    for key in augmentor_setup.keys():
        augmentor_setup[key] = Sequential(augmentor_setup[key],prob=1.0)
    return build_augmentor_multihead(augmentor_setup)

def build_augmentor(augmentor_setup):
    pprint(augmentor_setup)
    def augmentor(utt,label):
        return augmentor_setup[label](utt)
    return augmentor

def build_augmentor_from_config(conf):
    """Class specific augmentations"""
    augmentor_setup = defaultdict(list)
    for aug_setup in conf:
        augmentor = aug_setup["augmentor"]
        for cls_ind in aug_setup["target_classes"]:
            augmentor_setup[cls_ind].append(eval(augmentor))
    for key in augmentor_setup.keys():
        augmentor_setup[key] = Sequential(augmentor_setup[key],prob=1.0)
    return build_augmentor(augmentor_setup)
