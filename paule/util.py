import ctypes
import io
import os
import shutil
import sys
import tempfile
import warnings
import zipfile

import numpy as np
import matplotlib.pyplot as plt
import librosa
import torch.nn
import pandas as pd
import requests

DIR = os.path.dirname(__file__)
_PREFIX = ''
_FILE_ENDING = ''
if sys.platform.startswith('linux'):
    _FILE_ENDING = '.so'
    _PREFIX = 'lib'
elif sys.platform.startswith('win32'):
    _FILE_ENDING = '.dll'
elif sys.platform.startswith('darwin'):
    _FILE_ENDING = '.dylib'
    _PREFIX = 'lib'

# initialize vtl
VTL = ctypes.cdll.LoadLibrary(os.path.join(DIR, f'vocaltractlab_api/{_PREFIX}VocalTractLabApi{_FILE_ENDING}'))
SPEAKER_FILE_NAME = ctypes.c_char_p(os.path.join(DIR, 'vocaltractlab_api/JD3.speaker').encode())
FAILURE = VTL.vtlInitialize(SPEAKER_FILE_NAME)
if FAILURE != 0:
    raise ValueError('Error in vtlInitialize! Errorcode: %i' % FAILURE)
del SPEAKER_FILE_NAME, FAILURE, _FILE_ENDING, _PREFIX

# get version / compile date
VERSION = ctypes.c_char_p(b' ' * 64)
VTL.vtlGetVersion(VERSION)
print('Version of the VocalTractLab library: "%s"' % VERSION.value.decode())
del VERSION


# This should be done on all cp_deltas
#np.max(np.stack((np.abs(np.min(delta, axis=0)), np.max(delta, axis=0))), axis=0)
#np.max(np.stack((np.abs(np.min(cp_param, axis=0)), np.max(cp_param, axis=0))), axis=0)

# absolute value from max / min

#Vocal tract parameters: " 0  1  2  3  4  5  6  7   8   9  10  11  12  13  14  15  16  17  18"
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

ARTICULATOR = {0: 'vocal folds',
               1: 'tongue',
               2: 'lower incisors',
               3: 'lower lip',
               4: 'other articulator',
               5: 'num articulators',
               }

min_area = 0
max_area = 15

min_length = 0.23962031463970312
max_length = 0.6217119410833707

max_incisor = 18
min_incisor = 14

max_tongue = 1
min_tongue = -1

max_velum = 1
min_velum = 0

# tube sections area 0:6, 7 Incisor position, 8 tongue tip, 9 velum opening
tube_mins = np.concatenate([np.repeat(min_area,7), np.array([min_incisor]), np.array([min_tongue]), np.array([min_velum])])
tube_maxs = np.concatenate([np.repeat(max_area,7), np.array([max_incisor]), np.array([max_tongue]), np.array([max_velum])])

tube_theoretical_means = np.mean(np.stack([tube_mins,tube_maxs]),axis=0)
tube_theoretical_stds = np.std(np.stack([tube_mins,tube_maxs]),axis=0)


def librosa_melspec(wav, sample_rate):
    wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=44100,
            res_type='kaiser_best', fix=True, scale=False)
    melspec = librosa.feature.melspectrogram(y=wav, n_fft=1024, hop_length=220, n_mels=60, sr=44100, power=1.0, fmin=10, fmax=12000)
    melspec_db = librosa.amplitude_to_db(melspec, ref=0.15)
    return np.array(melspec_db.T, order='C', dtype=np.float64)


def normalize_cp(cp):
    return (cp - cp_theoretical_means) / cp_theoretical_stds


def inv_normalize_cp(norm_cp):
    return cp_theoretical_stds * norm_cp + cp_theoretical_means

def normalize_tube(tube):
    return (tube - tube_theoretical_means)/tube_theoretical_stds

def inv_normalize_tube(norm_tube):
    return norm_tube * tube_theoretical_stds + tube_theoretical_means

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
        which is 110 / 44100 seoconds (roughly 2.5 ms) (seq_length, 30)

    Returns
    =======
    (signal, sampling rate) : np.array, int
        returns the signal which is number of time steps in the cp_param array
        minus one times the time step length, i. e. ``(cp_param.shape[0] - 1) *
        110 / 44100``

    """
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
    tmp = np.ascontiguousarray(cp_param[:, 0:19])
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

def array_to_tensor(array):
    """
    Creates a Tensor with batch dim = 1

    Parameters
    ==========
    array : np.array

    Returns
    =======
    torch_tensor : troch.tensor (1, ...)
    """
    torch_tensor = array.copy()
    torch_tensor.shape = (1,) + torch_tensor.shape
    torch_tensor = torch.from_numpy(torch_tensor).detach().clone()
    return torch_tensor

def speak_and_extract_tube_information(cp_param):
    """
    Calls the vocal tract lab to synthesize an audio signal from the cp_param and simultaneously extract tube information
    Parameters
    ==========
    cp_param : np.array
        array containing the vocal and glottis parameters for each time step
        which is 110 / 44100 seoconds (roughly 2.5 ms) (seq_length, 30)
    Returns
    =======
    (signal, sampling rate, tube_info) : np.array, int, dict
        returns - the signal which is number of time steps in the cp_param array
                minus one times the time step length, i. e. ``(cp_param.shape[0] - 1) *
                110 / 44100``
                - the sampling rate (44100)
                - a dictionary containing arrays for the
                tube_length_cm (seq_length, 40 tube segments),
                tube_area_cm2 (seq_length, 40 tube segments),
                tube_articulator (seq_length, 40 tube segments),
                incisor_pos_cm (seq_length,),
                tongue_tip_side_elevation (seq_length, ),
                velum_opening_cm2 (seq_length, ).

    """
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
    frame_steps = 110  # 2.5 ms
    # within first parenthesis type definition, second initialisation
    audio = [(ctypes.c_double * int(frame_steps))() for _ in range(number_frames - 1)]

    # init the arrays
    tract_params = [(ctypes.c_double * (number_vocal_tract_parameters.value))() for _ in range(number_frames)]
    glottis_params = [(ctypes.c_double * (number_glottis_parameters.value))() for _ in range(number_frames)]

    # fill in data
    tmp = np.ascontiguousarray(cp_param[:, 0:19])
    for i in range(number_frames):
        tract_params[i][:] = tmp[i]
    del tmp

    tmp = np.ascontiguousarray(cp_param[:, 19:30])
    for i in range(number_frames):
        glottis_params[i][:] = tmp[i]
    del tmp

    # tube sections
    tube_length_cm = [(ctypes.c_double * 40)() for _ in range(number_frames)]
    tube_area_cm2 = [(ctypes.c_double * 40)() for _ in range(number_frames)]
    tube_articulator = [(ctypes.c_int * 40)() for _ in range(number_frames)]
    incisor_pos_cm = [ctypes.c_double(0) for _ in range(number_frames)]
    tongue_tip_side_elevation = [ctypes.c_double(0) for _ in range(number_frames)]
    velum_opening_cm2 = [ctypes.c_double(0) for _ in range(number_frames)]

    # Reset time-domain synthesis
    failure = VTL.vtlSynthesisReset()
    if failure != 0:
        raise ValueError(f'Error in vtlSynthesisReset! Errorcode: {failure}')

    for i in range(number_frames):
        if i == 0:
            failure = VTL.vtlSynthesisAddTract(0, ctypes.byref(audio[0]),
                                               ctypes.byref(tract_params[i]),
                                               ctypes.byref(glottis_params[i]))
        else:
            failure = VTL.vtlSynthesisAddTract(frame_steps, ctypes.byref(audio[i-1]),
                                               ctypes.byref(tract_params[i]),

                                               ctypes.byref(glottis_params[i]))
        if failure != 0:
            raise ValueError('Error in vtlSynthesisAddTract! Errorcode: %i' % failure)

        # export
        failure = VTL.vtlTractToTube(ctypes.byref(tract_params[i]),
                                     ctypes.byref(tube_length_cm[i]),
                                     ctypes.byref(tube_area_cm2[i]),
                                     ctypes.byref(tube_articulator[i]),
                                     ctypes.byref(incisor_pos_cm[i]),
                                     ctypes.byref(tongue_tip_side_elevation[i]),
                                     ctypes.byref(velum_opening_cm2[i]))

        if failure != 0:
            raise ValueError('Error in vtlTractToTube! Errorcode: %i' % failure)

    audio = np.ascontiguousarray(audio)
    audio.shape = ((number_frames - 1) * frame_steps,)

    arti = [[ARTICULATOR[sec] for sec in list(tube_articulator_i)] for tube_articulator_i in list(tube_articulator)]
    incisor_pos_cm = [x.value for x in incisor_pos_cm]
    tongue_tip_side_elevation = [x.value for x in tongue_tip_side_elevation]
    velum_opening_cm2 = [x.value for x in velum_opening_cm2]

    tube_info = {"tube_length_cm": np.array(tube_length_cm),
                 "tube_area_cm2": np.array(tube_area_cm2),
                 "tube_articulator": np.array(arti),
                 "incisor_pos_cm": np.array(incisor_pos_cm),
                 "tongue_tip_side_elevation": np.array(tongue_tip_side_elevation),
                 "velum_opening_cm2": np.array(velum_opening_cm2)}

    return (audio, 44100, tube_info)


def plot_cp(cp, file_name):
    """
    Plots the trajectory of the 3 control parameters with 10 in each subplot
    ==========
    cp : np.array
        array containing the vocal and glottis parameters for each time step
        which is 110 / 44100 seoconds (roughly 2.5 ms) (seq_length, 30)
    filename: str
        filename for saving plot
    Returns
    =======
    """
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
    """
    Plots the log-mel spectrogram
    ==========
    mel : np.array
        normalised log mel spectrogram (n_mel, seq_length)
    filename: str
        filename for saving plot
    Returns
    =======
    """
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
    wave: np.array

    """
    if which == "left":
        return wave[:, 0]
    if which == "right":
        return wave[:, 1]
    return (wave[:, 0] + wave[:, 1])/2


def pad_same_to_even_seq_length(seq):
    """
    Pad a sequence to an even sequence length by concatenating the last element once

    Parameters
    seq: np.array
    ==========

    Returns
    =======
    seq: np.array
    """

    if not seq.shape[0] % 2 == 0:
        return np.concatenate((seq, seq[-1:, :]), axis=0)
    else:
        return seq

def half_seq_by_average_pooling(seq):
    """
    Half a sequence by averaging adjecent points in time

    Parameters
    seq: np.array
    ==========

    Returns
    =======
    half_seq: np.array
    """
    if len(seq) % 2:
        seq = pad_same_to_even_seq_length(seq)
    half_seq = (seq[::2,:] + seq[1::2,:])/2
    return half_seq

def export_svgs(cps, path='svgs/', hop_length=5):
    """
    hop_length == 5 : roughly 80 frames per second
    hop_length == 16 : roughly 25 frames per second

    """
    n_tract_parameter = 19
    for ii in range(cps.shape[0] // hop_length):
        jj = ii * hop_length

        tract_params = (ctypes.c_double * 19)()
        tract_params[:] = cps[jj, :n_tract_parameter]

        file_name = os.path.join(path, f'tract{ii:05d}.svg')
        file_name = ctypes.c_char_p(file_name.encode())

        if not os.path.exists(path):
            os.mkdir(path)

        VTL.vtlExportTractSvg(tract_params, file_name)


class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

rmse_loss = RMSELoss(eps=0)


def calculate_five_point_stencil_without_padding(trajectory, *, delta_t=1.0):
    """
    Caculates the five-point stencil as a numeric approximation of the first derivative.

    https://en.wikipedia.org/wiki/Five-point_stencil

    Parameters
    ==========
    trajectory : np.array

    Returns
    =======
    derivative : np.array

    Equation
    ========

    .. math::

        f'(x) \approx {\frac {-f(x+2h)+8f(x+h)-8f(x-h)+f(x-2h)}{12h}}}

    """
    xx = trajectory
    return (-xx[:, 4:, :] + 8.0 * xx[:, 3:-1, :] - 8.0 * xx[:, 1:-3, :] + xx[:, :-4, :]) / (12.0 * delta_t)


def numeric_derivative(xx, *, delta_t=1.0):
    xx_prime = calculate_five_point_stencil_without_padding(xx, delta_t=delta_t)
    return xx_prime


def local_linear(trajectory, *, delta_t=1.0):
    """
    A locally linear trajectory will return a torch.tensor of zeros.

    """
    tt = trajectory
    return (2 * tt[:, 1:-1, :] - tt[:, :-2, :] - tt[:, 2:, :]) / (2 * delta_t)


def get_vel_acc_jerk(trajectory, *, lag=None, delta_t=1.0):
    """
    Approximates the velocity, acceleration, jerk for the given trajectory for a given lag

    Parameters
    ==========
    trajectory : np.array
    lag : int (deprecated; ignored)

    Returns
    =======
    (velocity, acceleration, jerk) : np.array, np.array, np.array
        returns the approximated velocity, acceleration and jerk of the trajectory for a given lag

    """
    if lag is not None:
        warnings.warn("lag should not used anymore and is ignored", DeprecationWarning, stacklevel=2)
    velocity = numeric_derivative(trajectory, delta_t=delta_t)
    acc = numeric_derivative(velocity, delta_t=delta_t)
    jerk = numeric_derivative(acc, delta_t=delta_t)
    return velocity, acc, jerk


def cp_trajectory_loss(Y_hat, tgts):
    """
    Calculates an additive loss using the RMSE between predicted and target position, velocity, acc and jerk

    Parameters
    ==========
    Y_hat : 3D torch.tensor
        model predictions (batch, seq_length, features)
    tgts : 3D torch.tensor
        target tensor (batch, seq_length, features)

    Returns
    =======
    (loss, pos_loss, vel_loss, acc_loss, jerk_loss) : torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor
        returns the summed total loss and all individual sub-losses
    """

    velocity, acc, jerk = get_vel_acc_jerk(tgts)
    velocity2, acc2, jerk2 = get_vel_acc_jerk(tgts, lag=2)
    velocity4, acc4, jerk4 = get_vel_acc_jerk(tgts, lag=4)

    Y_hat_velocity, Y_hat_acceleration, Y_hat_jerk = get_vel_acc_jerk(Y_hat)
    Y_hat_velocity2, Y_hat_acceleration2, Y_hat_jerk2 = get_vel_acc_jerk(Y_hat, lag=2)
    Y_hat_velocity4, Y_hat_acceleration4, Y_hat_jerk4 = get_vel_acc_jerk(Y_hat, lag=4)

    pos_loss = rmse_loss(Y_hat, tgts)
    vel_loss = rmse_loss(Y_hat_velocity, velocity) + rmse_loss(Y_hat_velocity2, velocity2) + rmse_loss(Y_hat_velocity4, velocity4)
    jerk_loss = rmse_loss(Y_hat_jerk, jerk) + rmse_loss(Y_hat_jerk2, jerk2) + rmse_loss(Y_hat_jerk4, jerk4)
    acc_loss = rmse_loss(Y_hat_acceleration, acc) + rmse_loss(Y_hat_acceleration2, acc2) + rmse_loss(Y_hat_acceleration4, acc4)

    loss = pos_loss + vel_loss + acc_loss + jerk_loss
    return loss, pos_loss, vel_loss, acc_loss, jerk_loss


def add_and_pad(xx, max_len, with_onset_dim=False):
    """
    Pad a sequence with last value to maximal length

    Parameters
    ==========
    xx : np.array
        sequence to pad (seq_length, features)
    max_len : int
        maximal length to be padded to
    with_onset_dim : bool
        add one features with 1 for the first time step and rest 0 to indicate
        sound onset

    Returns
    =======
    pad_seq : torch.Tensor
        padded sequence (max_len, features)

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
    lens : 1D torch.tensor
        Tensor containing the length of each sample in data_to_pad of one batch
    data_to_pad : pd.Series
        series containing the data to pad

    Returns
    =======
    padded_data : torch.tensors
        Tensors containing the padded and stacked to one batch

    """
    max_len = int(max(lens))
    padded_data = torch.stack(list(data_to_pad.apply(
        lambda x: add_and_pad(x, max_len, with_onset_dim=with_onset_dim)))).to(device)

    return padded_data


def cps_to_ema_and_mesh(cps, file_prefix, *, path=""):
    """
    Calls the vocal tract lab to generate synthesized EMA trajectories.

    Parameters
    ==========
    cps : np.array
        2D array containing the vocal and glottis parameters for each time step
        which is 110 / 44100 seoconds (roughly 2.5 ms); first dimension is
        sequence and second is vocal tract lab parameters, i. e. (n_sequence,
        30)
    file_prefix : str
        the prefix of the files written
    path : str
        path where to put the output files

    Returns
    =======
    None : None
        all output is writen to files.

    """
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

    number_frames = cps.shape[0]

    # init the arrays
    tract_params = (ctypes.c_double * (number_frames * number_vocal_tract_parameters.value))()
    glottis_params = (ctypes.c_double * (number_frames * number_glottis_parameters.value))()

    # fill in data
    tmp = np.ascontiguousarray(cps[:, 0:19])
    tmp.shape = (number_frames * 19,)
    tract_params[:] = tmp
    del tmp

    tmp = np.ascontiguousarray(cps[:, 19:30])
    tmp.shape = (number_frames * 11,)
    glottis_params[:] = tmp
    del tmp

    number_ema_points = 3
    surf = (ctypes.c_int * number_ema_points)()
    surf[:] = np.array([16, 16, 16])  # 16 = TONGUE

    vert = (ctypes.c_int * number_ema_points)()
    vert[:] = np.array([115, 225, 335])  # Tongue Back (TB) = 115; Tongue Middle (TM) = 225; Tongue Tip (TT) = 335

    if not os.path.exists(path):
        os.mkdir(path)

    failure = VTL.vtlTractSequenceToEmaAndMesh(
            ctypes.byref(tract_params), ctypes.byref(glottis_params),
            number_vocal_tract_parameters, number_glottis_parameters,
            number_frames, number_ema_points,
            ctypes.byref(surf), ctypes.byref(vert),
            path.encode(), file_prefix.encode())
    if failure != 0:
        raise ValueError('Error in vtlTractSequenceToEmaAndMesh! Errorcode: %i' % failure)


def cps_to_ema(cps):
    """
    Calls the vocal tract lab to generate synthesized EMA trajectories.

    Parameters
    ==========
    cps : np.array
        2D array containing the vocal and glottis parameters for each time step
        which is 110 / 44100 seoconds (roughly 2.5 ms); first dimension is
        sequence and second is vocal tract lab parameters, i. e. (n_sequence,
        30)

    Returns
    =======
    emas : pd.DataFrame
        returns the 3D ema points for different virtual EMA sensors in a
        pandas.DataFrame

    """
    with tempfile.TemporaryDirectory(prefix='python_paule_') as path:
        file_name = 'pyndl_util_ema_export'
        cps_to_ema_and_mesh(cps, file_prefix=file_name, path=path)
        emas = pd.read_table(os.path.join(path, f"{file_name}-ema.txt"), sep=' ')
    return emas


def seg_to_cps(seg_file):
    """
    Calls the vocal tract lab to read a segment file (seg_file) and returns the
    unnormalised cps.

    Parameters
    ==========
    seg_file : str
        path to the segment file

    Returns
    =======
    cps : np.array
        two dimensional numpy array of the unnormalised control parameter
        trajectories

    """
    segment_file_name = seg_file.encode()

    with tempfile.TemporaryDirectory() as tmpdirname:
        gesture_file_name = os.path.join(tmpdirname, 'vtl_ges_file.txt').encode()
        failure = VTL.vtlSegmentSequenceToGesturalScore(segment_file_name, gesture_file_name)
        if failure != 0:
            raise ValueError('Error in vtlSegmentSequenceToGesturalScore! Errorcode: %i' % failure)
        cps = ges_to_cps(gesture_file_name.decode())
    return cps


def ges_to_cps(ges_file):
    """
    Calls the vocal tract lab to read a gesture file (ges_file) and returns the
    unnormalised cps.

    Parameters
    ==========
    ges_file : str
        path to the gesture file

    Returns
    =======
    cps : np.array
        two dimensional numpy array of the unnormalised control parameter
        trajectories

    """
    gesture_file_name = ges_file.encode()

    with tempfile.TemporaryDirectory() as tmpdirname:
        tract_sequence_file_name = os.path.join(tmpdirname, 'vtl_tract_seq.txt').encode()
        failure = VTL.vtlGesturalScoreToTractSequence(gesture_file_name, tract_sequence_file_name)
        if failure != 0:
            raise ValueError('Error in vtlGesturalScoreToTractSequence! Errorcode: %i' % failure)

        cps = read_cp(tract_sequence_file_name.decode())
    return cps


def get_area_info_within_oral_cavity(tube_length, tube_area, *, cm_inside=7, calculate="min"):
    """
    Extracts the tube area information within the oral cavity for a given number of cm starting from the lips.

    Parameters
    ==========
    tube_length : np.array
        tube length in cm (seq_length, 40 tube segments)
    tube_area: np.array
        tube area in cm^2 (seq_length, 40 tube segments)
    cm_inside: int
        the number of cm to extend into the oral cavity
    calculate: str
        the information to extrect for each cm

    Returns
    =======
    section_area_per_time : np.array
        the extracted area information per time (seq_length, )
    """
    length_per_time = np.cumsum(tube_length, axis=1)
    section_area_per_time = []
    for t, l in enumerate(length_per_time):
        steps = [length_per_time[t][-1] - i * 1 for i in range(cm_inside + 1)][::-1]
        section_area = []
        for i, step in enumerate(steps[:-1]):
            indices = np.where(np.logical_and(l>=step, l<=steps[i+1]))[0]
            # add one more index, if possible as the next tube section is
            # paritally in this interval
            if indices[-1] < tube_area.shape[1] - 1:
                indices = np.concatenate((indices, indices[-1:] + 1))
            area = tube_area[t, indices]
            if calculate == "raw":
                section_area += [area]
            elif calculate == "mean":
                section_area += [np.mean(area)]
            elif calculate == "binary":
                section_area += [bool(np.sum(area <= 0.001))]
            elif calculate == "min":
                section_area += [np.min(area)]
            else:
                raise Exception(f"calculate must be one of ['raw', 'mean', 'binary', 'min']")
        section_area_per_time += [section_area]
    return np.asarray(section_area_per_time)


def download_pretrained_weights(*, skip_if_exists=True, verbose=True):
    package_path = DIR
    model_weights_path = os.path.join(package_path, 'pretrained_models')
    if os.path.isdir(model_weights_path):
        if skip_if_exists:
            if verbose:
                print(f"pretrained_models exist already. Skip download. Path is {model_weights_path}")
                print(f'Version of pretrained weights is "{get_pretrained_weights_version()}"')
            return
        shutil.rmtree(model_weights_path)

    zip_file_url = "https://nc.mlcloud.uni-tuebingen.de/index.php/s/N4nik8wgxwQHP83/download"
    if verbose:
        print(f"downloading 200 MB of pretrained weights from {zip_file_url}")
        print(f"saving pretrained weights to {model_weights_path}")
    stream = requests.get(zip_file_url, stream=True)
    zip_file = zipfile.ZipFile(io.BytesIO(stream.content))
    zip_file.extractall(package_path)
    if verbose:
        print(f'Version of pretrained weights is "{get_pretrained_weights_version()}"')


def get_pretrained_weights_version():
    """read and return the version of the pretrained weights, <No version file
    found> if no pretrained weights exist"""
    version_path = os.path.join(DIR, 'pretrained_models/version.txt')
    if not os.path.exists(version_path):
        return f"<No version file found at {version_path}>"
    with open(version_path, 'rt') as vfile:
        version = vfile.read().strip()
    return version

