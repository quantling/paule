import ctypes
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import librosa
import torch.nn

DIR = os.path.dirname(__file__)
_FILE_ENDING = ''
if sys.platform.startswith('linux'):
    _FILE_ENDING = '.so'
elif sys.platform.startswith('win32'):
    _FILE_ENDING = '.dll'
elif sys.platform.startswith('darwin'):
    _FILE_ENDING = '.dylib'

VTL = ctypes.cdll.LoadLibrary(os.path.join(DIR, 'vocaltractlab_api/libVocalTractLabApi' + _FILE_ENDING))
del _FILE_ENDING


# This should be done on all cp_deltas
#np.max(np.stack((np.abs(np.min(delta, axis=0)), np.max(delta, axis=0))), axis=0)
#np.max(np.stack((np.abs(np.min(cp_param, axis=0)), np.max(cp_param, axis=0))), axis=0)

# absolute value from max / min

#Vocal tract parameters: "HX HY JX JA LP LD VS VO TCX TCY TTX TTY TBX TBY TRX TRY TS1 TS2 TS3"
#Glottis parameters: "f0 pressure x_bottom x_top chink_area lag rel_amp double_pulsing pulse_skewness flutter aspiration_strength "


cp_means = np.array([ 5.3000e-01, -5.0800e+00, -3.0000e-02, -3.7300e+00,  7.0000e-02,
                      7.3000e-01,  4.8000e-01, -5.0000e-02,  9.6000e-01, -1.5800e+00,
                      4.4600e+00, -9.3000e-01,  2.9900e+00, -5.0000e-02, -1.4600e+00,
                     -2.2900e+00,  2.3000e-01,  1.2000e-01,  1.2000e-01,  1.0720e+02,
                      4.1929e+03,  3.0000e-02,  3.0000e-02,  6.0000e-02,  1.2200e+00,
                      8.4000e-01,  5.0000e-02,  0.0000e+00,  2.5000e+01, -1.0000e+01])
cp_stds = np.array([1.70000e-01, 4.00000e-01, 4.00000e-02, 6.30000e-01, 1.20000e-01,
                    2.20000e-01, 2.20000e-01, 9.00000e-02, 4.90000e-01, 3.10000e-01,
                    3.80000e-01, 3.70000e-01, 3.50000e-01, 3.50000e-01, 4.60000e-01,
                    3.80000e-01, 6.00000e-02, 1.00000e-01, 1.80000e-01, 9.86000e+00,
                    3.29025e+03, 2.00000e-02, 2.00000e-02, 1.00000e-02, 0.00100e+00,
                    2.00000e-01, 0.00100e+00, 0.00100e+00, 0.00100e+00, 0.00100e+00])

cp_theoretical_means = np.array([ 5.00000e-01, -4.75000e+00, -2.50000e-01, -3.50000e+00,
        0.00000e+00,  1.00000e+00,  5.00000e-01,  4.50000e-01,
        5.00000e-01, -1.00000e+00,  3.50000e+00, -2.50000e-01,
        5.00000e-01,  1.00000e+00, -1.00000e+00, -3.00000e+00,
        5.00000e-01,  5.00000e-01,  0.00000e+00,  3.20000e+02,
        1.00000e+04,  1.25000e-01,  1.25000e-01,  0.00000e+00,
        1.57075e+00,  0.00000e+00,  5.00000e-01,  0.00000e+00,
        5.00000e+01, -2.00000e+01])

cp_theoretical_stds = np.array([5.00000e-01, 1.25000e+00, 2.50000e-01, 3.50000e+00, 1.00000e+00,
       3.00000e+00, 5.00000e-01, 5.50000e-01, 3.50000e+00, 2.00000e+00,
       2.00000e+00, 2.75000e+00, 3.50000e+00, 4.00000e+00, 3.00000e+00,
       3.00000e+00, 5.00000e-01, 5.00000e-01, 1.00000e+00, 2.80000e+02,
       1.00000e+04, 1.75000e-01, 1.75000e-01, 2.50000e-01, 1.57075e+00,
       1.00000e+00, 5.00000e-01, 5.00000e-01, 5.00000e+01, 2.00000e+01])


def librosa_melspec(wav, sample_rate):
    melspec = librosa.feature.melspectrogram(wav, n_fft=1024, hop_length=220, n_mels=60, sr=sample_rate, power=1.0, fmin=10, fmax=12000)
    melspec_db = librosa.amplitude_to_db(melspec, ref=0.15)
    return np.array(melspec_db.T, order='C')


def normalize_cp(cp):
    return (cp - cp_theoretical_means) / cp_theoretical_stds


def inv_normalize_cp(norm_cp):
    return cp_theoretical_stds * norm_cp + cp_theoretical_means


# -83.52182518111363
mel_mean_librosa = librosa_melspec(np.zeros(5000), 44100)[0, 0]
mel_std_librosa = abs(mel_mean_librosa)


def normalize_mel_librosa(mel):
    return (mel - mel_mean_librosa) / mel_std_librosa


def inv_normalize_mel_librosa(norm_mel):
    return mel_std_librosa * norm_mel + mel_mean_librosa


def read_cp(filename):
    with open(filename, 'rt') as cp_file:
        # skip first 6 lines
        for _ in range(6):
            cp_file.readline()
        glottis_model = cp_file.readline().strip()
        if glottis_model != 'Geometric glottis':
            print(glottis_model)
            raise ValueError(f'glottis model is not "Geometric glottis" in file {filename}')
        n_states = int(cp_file.readline().strip())
        cp_param = np.zeros((n_states, 19 + 11))
        for ii, line in enumerate(cp_file):
            kk = ii // 2
            if kk >= n_states:
                raise ValueError(f'more states saved in file {filename} than claimed in the beginning')
            # even numbers are glottis params
            elif ii % 2 == 0:
                glottis_param = line.strip()
                cp_param[kk, 19:30] = np.mat(glottis_param)
            # odd numbers are tract params
            elif ii % 2 == 1:
                tract_param = line.strip()
                cp_param[kk, 0:19] = np.mat(tract_param)
    return cp_param


def speak(cp_param):
    """
    Calls the vocal tract lab to synthesize an audio signal from the cp_param.

    Parameters
    ==========
    cp_param : np.array
        array containing the vocal and glottis parameters for each time step
        which is 110 / 44100 seoconds (roughly 2.5 ms)

    Returns
    =======
    (signal, sampling rate) : np.array, int
        returns the signal which is number of time steps in the cp_param array
        minus one times the time step length, i. e. ``(cp_param.shape[0] - 1) *
        110 / 44100``

    """
    # initialize vtl
    speaker_file_name = ctypes.c_char_p(os.path.join(DIR, 'vocaltractlab_api/JD3.speaker').encode())

    failure = VTL.vtlInitialize(speaker_file_name)
    if failure != 0:
        raise ValueError('Error in vtlInitialize! Errorcode: %i' % failure)

    # get some constants
    audio_sampling_rate = ctypes.c_int(0)
    number_tube_sections = ctypes.c_int(0)
    number_vocal_tract_parameters = ctypes.c_int(0)
    number_glottis_parameters = ctypes.c_int(0)
    number_audio_samples_per_tract_state = ctypes.c_int(0)
    internal_sampling_rate = ctypes.c_double(0)

    VTL.vtlGetConstants(ctypes.byref(audio_sampling_rate),
                        ctypes.byref(number_tube_sections),
                        ctypes.byref(number_vocal_tract_parameters),
                        ctypes.byref(number_glottis_parameters),
                        ctypes.byref(number_audio_samples_per_tract_state),
                        ctypes.byref(internal_sampling_rate))

    assert audio_sampling_rate.value == 44100
    assert number_vocal_tract_parameters.value == 19
    assert number_glottis_parameters.value == 11

    number_frames = cp_param.shape[0]
    frame_steps = 110 #2.5 ms
    # within first parenthesis type definition, second initialisation
    # 2000 samples more in the audio signal for safety
    audio = (ctypes.c_double * int((number_frames-1) * frame_steps + 2000))()

    # init the arrays
    tract_params = (ctypes.c_double * (number_frames * number_vocal_tract_parameters.value))()
    glottis_params = (ctypes.c_double * (number_frames * number_glottis_parameters.value))()

    # fill in data
    tmp = np.ascontiguousarray(cp_param[:, 0:19]) # am einen stück speicher 
    tmp.shape = (number_frames * 19,)
    tract_params[:] = tmp
    del tmp

    tmp = np.ascontiguousarray(cp_param[:, 19:30])
    tmp.shape = (number_frames * 11,)
    glottis_params[:] = tmp
    del tmp

    # Reset time-domain synthesis
    failure = VTL.vtlSynthesisReset()
    if failure != 0:
        raise ValueError(f'Error in vtlSynthesisReset! Errorcode: {failure}')

    # Call the synthesis function. It may calculate a few seconds.
    failure = VTL.vtlSynthBlock(
                    ctypes.byref(tract_params),  # input
                    ctypes.byref(glottis_params),  # input
                    number_frames,
                    frame_steps,
                    ctypes.byref(audio),  # output
                    0)
    if failure != 0:
        raise ValueError('Error in vtlSynthBlock! Errorcode: %i' % failure)

    VTL.vtlClose()

    return (np.array(audio[:-2000]), 44100)


def audio_padding(sig, samplerate, winlen=0.010):
    """
    Pads the signal by half a window length on each side with zeros.

    Parameters
    ==========
    sig : np.array
        the audio signal
    samplerate : int
        sampling rate
    winlen : float
        the window size in seconds

    """
    pad = int(np.ceil(samplerate * winlen)/2)
    z = np.zeros(pad)
    pad_signal = np.concatenate((z,sig,z))
    return pad_signal


def mel_to_sig(mel, mel_min=0.0):
    """
    creates audio from a normlised log mel spectrogram.

    Parameters
    ==========
    mel : np.array
        normalised log mel spectrogram (n_mel, seq_length)
    mel_min : float
        original min value (default: 0.0)

    Returns
    =======
    (sig, sampling_rate) : (np.array, int)

    """
    mel = mel + mel_min
    mel = inv_normalize_mel_librosa(mel)
    mel = np.array(mel.T, order='C')
    mel = librosa.db_to_amplitude(mel, ref=0.15)
    sig = librosa.feature.inverse.mel_to_audio(mel, sr=44100, n_fft=1024,
                                             hop_length=220, win_length=1024,
                                             power=1.0, fmin=10, fmax=12000)
    # there are always 110 data points missing compared to the speak function using VTL
    # add 55 zeros to the beginning and the end
    sig = np.concatenate((np.zeros(55), sig, np.zeros(55)))
    return (sig, 44100)


def plot_cp(cp, file_name):
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_axes([0.1, 0.65, 0.8, 0.3], ylim=(-3, 3))
    ax2 = fig.add_axes([0.1, 0.35, 0.8, 0.3], xticklabels=[], sharex=ax1, sharey=ax1)
    ax3 = fig.add_axes([0.1, 0.05, 0.8, 0.3], sharex=ax1, sharey=ax1)

    for ii in range(10):
        ax1.plot(cp[:, ii], label=f'param{ii:0d}')
    ax1.legend()
    for ii in range(10, 20):
        ax2.plot(cp[:, ii], label=f'param{ii:0d}')
    ax2.legend()
    for ii in range(20, 30):
        ax3.plot(cp[:, ii], label=f'param{ii:0d}')
    ax3.legend()
    fig.savefig(file_name, dpi=300)
    plt.close('all')


def plot_mel(mel, file_name):
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(mel.T, aspect='equal', vmin=-5, vmax=20)
    fig.savefig(file_name, dpi=300)
    plt.close('all')


def stereo_to_mono(wave, which="both"):
    """
    Extract a channel from a stereo wave

    Parameters
    ==========
    wave: np.array
        Input wave data.
    which: {"left", "right", "both"} default = "both"
        if `mono`, `which` indicates whether the *left* or the *right* channel
        should be extracted, or whether *both* channels should be averaged.

    Returns
    =======
    data: numpy.array

    """
    if which == "left":
        return wave[:, 0]
    if which == "right":
        return wave[:, 1]
    return (wave[:, 0] + wave[:, 1])/2


def pad_same_to_even_seq_length(array):
    if not array.shape[0] % 2 == 0:
        return np.concatenate((array, array[-1:, :]), axis=0)
    else:
        return array


def export_svgs(cps, path='svgs/', hop_length=5):
    """
    hop_length == 5 : roughly 80 frames per second
    hop_length == 16 : roughly 25 frames per second

    """
    n_tract_parameter = 19
    # initialize vtl
    speaker_file_name = ctypes.c_char_p(os.path.join(DIR, 'vocaltractlab_api/JD3.speaker').encode())

    failure = VTL.vtlInitialize(speaker_file_name)
    if failure != 0:
        raise ValueError('Error in vtlInitialize! Errorcode: %i' % failure)

    for ii in range(cps.shape[0] // hop_length):
        jj = ii * hop_length

        tract_params = (ctypes.c_double * 19)()
        tract_params[:] = cps[jj, :n_tract_parameter]

        file_name = os.path.join(path, f'tract{ii:05d}.svg')
        file_name = ctypes.c_char_p(file_name.encode())

        if not os.path.exists(path):
            os.mkdir(path)

        VTL.vtlExportTractSvg(tract_params, file_name)

    VTL.vtlClose()


class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


def add_and_pad(xx, max_len, with_onset_dim=False):
    """
    Pad a sequence with last value to maximal length

    Parameters
    ==========
    xx : 2D np.array
        seuence to be padded (seq_length, feeatures)
    max_len : int
        maximal length to be padded to
    with_onset_dim : bool
        add one features with 1 for the first time step and rest 0 to indicate
        sound onset

    Returns
    =======
    pad_seq : torch.Tensor
        2D padded sequence

    """
    seq_length = xx.shape[0]
    if with_onset_dim:
        onset = np.zeros((seq_length, 1))
        onset[0, 0] = 1
        xx = np.concatenate((xx, onset), axis=1)  # shape len X (features +1)
    padding_size = max_len - seq_length
    padding_size = tuple([padding_size] + [1 for i in range(len(xx.shape) - 1)])
    xx = np.concatenate((xx, np.tile(xx[-1:], padding_size)), axis=0)
    return torch.from_numpy(xx)


def pad_batch_online(lens, data_to_pad, device="cpu", with_onset_dim=False):
    """
    pads and batches data into one single padded batch.

    Parameters
    ==========
    lens : 1D torch.Tensor
        Tensor containing the length of each sample in data_to_pad of one batch
    data_to_pad : series
        series containing the data to pad

    Returns
    =======
    padded_data : torch.Tensors
        Tensors containing the padded and stacked to one batch

    """
    max_len = int(max(lens))
    padded_data = torch.stack(list(data_to_pad.apply(
        lambda x: add_and_pad(x, max_len, with_onset_dim=with_onset_dim)))).to(device)

    return padded_data

