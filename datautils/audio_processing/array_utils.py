import numpy as np

def framing(sig, kernel_size=10, strides=5):
    """
    transform a signal into a series of overlapping frames.

    Args:
     sig            (array) : a mono audio signal (Nx1) from which to compute features.
     fs               (int) : the sampling frequency of the signal we are working with.
                              Default is 16000.
     win_len        (float) : window length in sec.
                              Default is 0.025.
     win_hop        (float) : step between successive windows in sec.
                              Default is 0.01.

    Returns:
     array of frames.
     frame length.
    """
    # compute frame length and frame step (convert from seconds to samples)
    frame_length = kernel_size
    frame_step = strides
    signal_length = len(sig)
    frames_overlap = frame_length - frame_step

    # Make sure that we have at least 1 frame+
    num_frames = np.abs(signal_length - frames_overlap) // np.abs(frame_length - frames_overlap)
    rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)

    # Pad Signal to make sure that all frames have equal number of samples
    # without truncating any samples from the original signal
    #      if rest_samples != 0:
    #          pad_signal_length = int(frame_step - rest_samples)
    #          z = np.zeros((pad_signal_length))
    #          pad_signal = np.append(sig, z)
    #          num_frames += 1
    #      else:
    pad_signal = sig

    # make sure to use integers as indices
    frame_length = int(frame_length)
    frame_step = int(frame_step)
    num_frames = int(num_frames)

    # compute indices
    idx1 = np.tile(np.arange(0, frame_length), (num_frames, 1))
    idx2 = np.tile(np.arange(0, num_frames * frame_step, frame_step),
                (frame_length, 1)).T
    indices = idx1 + idx2
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames

def interp1d(x : np.ndarray, N : int, mode="linear"):
    # 1D interpolation
    assert mode in ["nearest","linear"]
    assert x.ndim == 1
    assert N > 0
    M = x.shape[0]
    coords = (np.arange(N) * M/N)
    interpolated = None
    if mode == "nearest":
        interpolated = x[coords.astype(np.int32)]
    if mode == "linear":
        coords_bot = np.floor(coords).astype(np.int32)
        coords_top = np.clip(np.ceil(coords),0,M-1).astype(np.int32)
        a =  coords - coords_bot
        interpolated = x[coords_bot]*(1-a) + a*x[coords_top]
    return interpolated

def signal_power_framed(signal,window_size=2048,step=256,method=np.std):
    return np.std(framing(signal,window_size,step),axis=-1)