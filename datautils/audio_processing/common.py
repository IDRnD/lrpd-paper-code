import os
import tempfile
import warnings
from pathlib import Path
from typing import Union

import librosa
import numpy as np
import soundfile
from scipy.io import wavfile
from scipy.signal import butter, lfilter


_EPS = 1e-8


def get_audio_loader(samplerate: int = 16000,
                     backend: str = "librosa",
                     zero_data_check: bool = True,
                     raise_error: bool = False):
    # Might be different realizations
    def check_zero_output_wrapper(func):
        def wrapped_func(*args, **kwargs):
            result = func(*args, **kwargs)
            if not np.any(result):
                warnings.warn(
                    f"Zero output warning : {func.__name__}({args},{kwargs}) output has all zero values"
                )
                if raise_error:
                    raise Exception(
                        f"Zero output warning : {func.__name__}({args},{kwargs}) output has all zero values"
                    )
            return result

        return wrapped_func

    def load_audio(path: Union[str, Path]):
        if backend == "librosa":
            # s = str(path).encode('ascii','surrogateescape').decode().encode('utf-8')
            return librosa.core.load(path, sr=samplerate, mono=True)[0]
        elif backend == "wavfile":
            sr, data = wavfile.read(str(path))
            assert sr == samplerate
            return data
        else:
            NotImplemented()

    if zero_data_check:
        return check_zero_output_wrapper(load_audio)
    else:
        return load_audio


def normalize(signal: np.ndarray, axis=-1):
    max_value = np.abs(signal).max(axis=axis, keepdims=True)
    max_value = np.clip(max_value, _EPS, None)
    return signal / max_value


def get_audio_loader_v2(samplerate: int = 16000,
                        backend: str = "librosa",
                        zero_data_check: bool = True):
    # Might be different realizations
    def check_zero_output_wrapper(func):
        def wrapped_func(*args, **kwargs):
            result = func(*args, **kwargs)
            if not np.any(result):
                warnings.warn(
                    f"Zero output warning : {func.__name__}({args},{kwargs}) output has all zero values"
                )
            return result

        return wrapped_func

    def load_audio(path: Union[str, Path]):
        if backend == "librosa":
            s = str(path).encode('ascii', 'surrogateescape').decode().encode('utf-8')
            return librosa.core.load(s, sr=samplerate, mono=True)[0]
        elif backend == "wavfile":
            sr, data = wavfile.read(str(path))
            assert sr == samplerate
            return data
        else:
            NotImplemented()

    if zero_data_check:
        return check_zero_output_wrapper(load_audio)
    else:
        return load_audio


def extend_signal(signal: np.ndarray, target_len: int, repeat: bool = True, axis=-1):
    if signal.shape[axis] >= target_len:
        # Do not extend
        return signal

    if repeat:
        while signal.shape[axis] < target_len:
            signal = np.concatenate((signal, signal), axis=axis)
    else:
        if signal.shape[axis] < target_len:
            event_offset_samples = np.random.randint(target_len - signal.shape[axis] + 1)
            tail_length = target_len - signal.shape[axis] - event_offset_samples
            paddings = [(0, 0)] * signal.ndim
            paddings[axis] = (event_offset_samples, tail_length)
            signal = np.pad(signal, pad_width=paddings, mode='constant', constant_values=0)
    return signal


def cut_signals(signal, target_len: int, random=True):
    if random:
        max_offset = signal.shape[-1] - target_len
        assert max_offset >= 0
        start_ind = np.random.randint(0, max(max_offset, 1))
        stop_ind = start_ind + target_len
        return signal[..., start_ind:stop_ind]
    else:
        return signal[..., :target_len]


def kaldi_signal_power(x: np.ndarray, axis=-1, keepdims=False):
    return np.sum(x * x, axis=axis, keepdims=keepdims) / x.shape[axis]


def compute_noise_scaling_factor(signal: np.ndarray,
                                 noize: np.ndarray,
                                 snr: float,
                                 method=kaldi_signal_power,
                                 axis=None,
                                 keepdims=False,
                                 eps: float = 1e-7) -> float:
    def pow_2_db(ratio):
        return 10 * np.log10(ratio)

    def db_2_pow(db):
        return 10**(db / 10)

    original_sn_rmse_ratio = method(
        signal, axis=axis, keepdims=keepdims) / (method(noize, axis=axis, keepdims=keepdims) + eps)
    target_sn_rmse_ratio = db_2_pow(snr)
    signal_scaling_factor = target_sn_rmse_ratio / (original_sn_rmse_ratio + eps)
    return signal_scaling_factor


def mix_multiple_signals(singals: np.ndarray,
                         noise: np.ndarray,
                         snr: float,
                         noise_cut_random: bool = True,
                         eps: float = _EPS):
    """
    singals : shape = (num_signals,num_samples)
    """
    assert singals.ndim == 2
    assert noise.ndim == 1
    signals_fix_len = singals.shape[1]
    #     print(signals_fix_len)

    singals = normalize(singals)
    noise = normalize(noise)

    # Fit noise to signals_fix_len
    noise = extend_signal(noise, target_len=signals_fix_len)
    if noise_cut_random:
        max_offset = noise.shape[0] - signals_fix_len
        start_point = np.random.randint(0, max(max_offset, 1))
        stop_point = start_point + signals_fix_len
        noise = noise[start_point:stop_point]
    else:
        noise = noise[:signals_fix_len]

    signals_scaling_factors = compute_noise_scaling_factor(singals,
                                                           noise,
                                                           snr=snr,
                                                           axis=-1,
                                                           keepdims=True,
                                                           eps=eps)
    mixed_signals_noise = singals + noise[None, :] / (signals_scaling_factors + eps)
    return mixed_signals_noise


def mix_single_signal(singal: np.ndarray,
                      noise: np.ndarray,
                      snr: float,
                      noise_cut_random: bool = True,
                      eps: float = _EPS):
    assert singal.ndim == 1
    return mix_multiple_signals(singal[None, :],
                                noise=noise,
                                snr=snr,
                                noise_cut_random=noise_cut_random,
                                eps=eps)[0, :]


def get_energy_stft(
        signal,
        sr: int = 16000,
        num_samples_per_per_fft_comp: float = 31.25,  # nfft = int(num_samples_per_per_fft_comp*sr)
        smooth_wind_size_sec: float = 0.1,
        stft_win_len_sec: float = 0.025,
        stft_hop_len_sec: float = 0.01):
    nfft = int(num_samples_per_per_fft_comp * sr)
    stft = np.abs(
        librosa.stft(signal,
                     n_fft=nfft,
                     hop_length=int(stft_hop_len_sec * sr),
                     win_length=int(stft_win_len_sec * sr)))
    smooth_wind_size = int(smooth_wind_size_sec * sr)
    kernel = np.ones(smooth_wind_size) / smooth_wind_size
    energy = stft.sum(axis=0)
    smoothed_energy = np.convolve(energy, kernel, mode="same")
    return smoothed_energy


def butter_bandpass_filter(data, lowcut=20, highcut=3800, fs=16000, order=5):
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass_filter(data, cutOff=2000, fs=16000, order=8):
    def butter_lowpass(cutOff, fs, order=5):
        nyq = 0.5 * fs
        normalCutoff = cutOff / nyq
        b, a = butter(order, normalCutoff, btype='low', analog=False)
        return b, a

    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_highpass_filter(data, cutOff=100, fs=16000, order=8):
    def butter_highpass(cutOff, fs, order=5):
        nyq = 0.5 * fs
        normalCutoff = cutOff / nyq
        b, a = butter(order, normalCutoff, btype='high', analog=False)
        return b, a

    b, a = butter_highpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def generate_fake_wavs(num: int = 10, sr: int = 16000, duration_sec: float = 3.0):
    output_files = []
    tmpdirname = tempfile.mktemp()
    os.makedirs(tmpdirname, exist_ok=True)
    for i in range(num):
        fake_wav_array = np.random.rand(int(sr * duration_sec), 1)
        fake_wav_name = f"{tmpdirname}/fake_wav_{i}.wav"
        soundfile.write(fake_wav_name, fake_wav_array, sr)
        output_files.append(fake_wav_name)

    return output_files
