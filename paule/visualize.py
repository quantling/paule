"""
This module contains useful functions to visualize and safe results.

"""

import os
import pickle

import soundfile as sf
from matplotlib import pyplot as plt
import librosa.display
from matplotlib import cm
import numpy as np

from . import util


def visualize_results(results, condition='prefix', folder='data'):
    """
    Stores all results in data/ folder.

    """
    if isinstance(results, str):
        with open(results, 'rb') as pfile:
            results = pickle.load(pfile)

    base_name = os.path.join(folder, f'{condition}')

    # save mel plot
    plot_mels(f"{base_name}_mel.png", results.target_mel, results.initial_pred_mel,
            results.initial_prod_mel, results.pred_mel, results.prod_mel)

    # save audio
    target_sr = prod_sr = 44100
    sf.write(f'{base_name}_planned.flac', results.prod_sig, results.prod_sr)
    sf.write(f'{base_name}_initial.flac', results.initial_sig, results.initial_sr)
    if results.target_sig is not None:
        sf.write(f'{base_name}_target.flac', results.target_sig, results.target_sr)

    # save loss plot
    fig, ax = plt.subplots(figsize=(15, 8), facecolor="white")
    ax.plot(results.planned_loss_steps, label="planned loss", c="C0")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{base_name}_loss.png")

    fig, ax = plt.subplots(figsize=(15, 8), facecolor="white")
    ax.plot(results.prod_loss_steps, label="produced mel loss", c="C1")
    ax.plot(results.planned_mel_loss_steps, label="planned mel loss", c="C0")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{base_name}_loss_mel.png")

    # save subloss plot
    fig, ax = plt.subplots(figsize=(15, 8), facecolor="white")
    ax.plot(results.vel_loss_steps, label="vel loss", c="C2")
    ax.plot(results.jerk_loss_steps, label="jerk loss", c="C3")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{base_name}_loss_subloss.png")

    # save semvec loss plot
    fig, ax = plt.subplots(figsize=(15, 8), facecolor="white")
    ax.plot(results.pred_semvec_loss_steps, label="planned semvec loss", c="C0")
    ax.plot(results.prod_semvec_loss_steps, label="produced semvec loss", c="C1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{base_name}_loss_semvec.png")

    # save speech classifier loss plot
    if hasattr(results, 'pred_speech_classifier_loss_steps'):
        fig, ax = plt.subplots(figsize=(15, 8), facecolor="white")
        ax.plot(results.pred_speech_classifier_loss_steps, label="planned speech classifier loss", c="C0")
        ax.plot(np.array(results.prod_speech_classifier_loss_steps) / 10.0, label="produced speech classifier loss", c="C1")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{base_name}_loss_speech_classifier.png")

    # save cp change plot
    fig = plt.figure(figsize=(15, 12))
    ax1 = fig.add_axes([0.1, 0.68, 0.88, 0.30], xticklabels=[])
    ax2 = fig.add_axes([0.1, 0.36, 0.88, 0.30], xticklabels=[], sharex=ax1)
    ax3 = fig.add_axes([0.1, 0.04, 0.88, 0.30], xticklabels=[], sharex=ax1)
    img1 = results.initial_cp  #target['cp'].iloc[0]
    img2 = results.planned_cp
    img3 = img2 - img1
    ax1.plot(img1[:,  3: 4], label='JA')
    ax1.plot(img1[:,  8: 9], label='TCX')
    ax1.plot(img1[:,  9:10], label='TCY')
    ax1.plot(img1[:, 10:11], label='TTX')
    ax1.plot(img1[:, 11:12], label='TTY')
    ax1.plot(img1[:, 12:13], label='TBX')
    ax1.plot(img1[:, 13:14], label='TBY')
    ax1.plot(img1[:, 14:15], label='TRX')
    ax1.plot(img1[:, 15:16], label='TRY')
    ax1.plot(img1[:, 19:20], label='f0')
    ax1.set_ylabel("initial")
    ax2.plot(img2[:,  3: 4], label='JA')
    ax2.plot(img2[:,  8: 9], label='TCX')
    ax2.plot(img2[:,  9:10], label='TCY')
    ax2.plot(img2[:, 10:11], label='TTX')
    ax2.plot(img2[:, 11:12], label='TTY')
    ax2.plot(img2[:, 12:13], label='TBX')
    ax2.plot(img2[:, 13:14], label='TBY')
    ax2.plot(img2[:, 14:15], label='TRX')
    ax2.plot(img2[:, 15:16], label='TRY')
    ax2.plot(img2[:, 19:20], label='f0')
    ax2.set_ylabel("optimized")
    ax3.plot(img3[:,  3: 4], label='JA')
    ax3.plot(img3[:,  8: 9], label='TCX')
    ax3.plot(img3[:,  9:10], label='TCY')
    ax3.plot(img3[:, 10:11], label='TTX')
    ax3.plot(img3[:, 11:12], label='TTY')
    ax3.plot(img3[:, 12:13], label='TBX')
    ax3.plot(img3[:, 13:14], label='TBY')
    ax3.plot(img3[:, 14:15], label='TRX')
    ax3.plot(img3[:, 15:16], label='TRY')
    ax3.plot(img3[:, 19:20], label='f0')
    ax3.set_ylabel("difference")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(f'{base_name}_cps.png')

    # save svgs and create mp4s
    path = f"{base_name}_initial_svgs/"
    if not os.path.exists(path):
        os.mkdir(path)
    util.export_svgs(util.inv_normalize_cp(results.initial_cp), path=path)
    system_call = f'cd {path}; /usr/bin/ffmpeg -hide_banner -loglevel error -r 80 -width 768 -i tract%05d.svg -i ../{condition}_initial.flac -c:v libx264 -pix_fmt yuv420p ../{condition}_initial_80Hz.mp4'
    return_value = os.system(system_call)
    if return_value != 0:
        print("WARNING: creating the initial animation went wrong")
    ## recode to 60 Hz
    #system_call = f'cd {path}; /usr/bin/ffmpeg -loglevel error -i ../{condition}_initial_80Hz.mp4 -r 60 ../{condition}_initial_60Hz.mp4'
    #return_value = os.system(system_call)
    #if return_value != 0:
    #    print("WARNING: creating the initial animation went wrong")

    path = f"{base_name}_planned_svgs/"
    if not os.path.exists(path):
        os.mkdir(path)
    util.export_svgs(util.inv_normalize_cp(results.planned_cp), path=path)
    system_call = f'cd {path}; /usr/bin/ffmpeg -hide_banner -loglevel error -r 80 -width 768 -i tract%05d.svg -i ../{condition}_planned.flac -c:v libx264 -pix_fmt yuv420p ../{condition}_planned_80Hz.mp4'
    return_value = os.system(system_call)
    if return_value != 0:
        print("WARNING: creating the planning animation went wrong")
    ## recode to 60 Hz
    #system_call = f'cd {path}; /usr/bin/ffmpeg -loglevel error -i ../{condition}_planned_80Hz.mp4 -r 60 ../{condition}_planned_60Hz.mp4'
    #return_value = os.system(system_call)
    #if return_value != 0:
    #    print("WARNING: creating the planned animation went wrong")

    #plt.show()  # this shows all saved figures


def plot_mels(file_name, target_mel, initial_pred_mel, initial_prod_mel,
        pred_mel, prod_mel):
    """
    Plots target, initial prediction, initial production, prediction and
    production log mel spectrograms.

    Parameters
    ==========
    file_name : str or True
    target_mel : np.array
    initial_pred_mel : np.array
    initial_prod_mel : np.array
    pred_mel : np.array
    prod_mel : np.array

    """
    fig, ax = plt.subplots(nrows=6, figsize=(15, 18), facecolor="white")
    librosa.display.specshow(target_mel.T, y_axis='mel',
            x_axis='time', sr=44100, hop_length=220, ax=ax[0],
            cmap=cm.magma)
    ax[0].set_title("Target", fontsize=18)
    librosa.display.specshow(initial_prod_mel.T, y_axis='mel',
            x_axis='time', sr=44100, hop_length=220, ax=ax[1],
            cmap=cm.magma)
    ax[1].set_title("Initial Produced", fontsize=18)
    librosa.display.specshow(initial_pred_mel.T, y_axis='mel',
            x_axis='time', sr=44100, hop_length=220, ax=ax[2],
            cmap=cm.magma)
    ax[2].set_title("Initial Prediction", fontsize=18)
    librosa.display.specshow(pred_mel.T, y_axis='mel',
            x_axis='time', sr=44100, hop_length=220, ax=ax[3],
            cmap=cm.magma)
    ax[3].set_title("Planned Prediction", fontsize=18)
    librosa.display.specshow(prod_mel.T, y_axis='mel',
            x_axis='time', sr=44100, hop_length=220, ax=ax[4],
            cmap=cm.magma)
    ax[4].set_title("Planned Produced", fontsize=18)
    librosa.display.specshow(target_mel.T, y_axis='mel',
            x_axis='time', sr=44100, hop_length=220, ax=ax[5],
            cmap=cm.magma)
    ax[5].set_title("Target", fontsize=18)

    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[2].set_xticks([])
    ax[3].set_xticks([])
    ax[4].set_xticks([])
    ax[0].set_xlabel("")
    ax[1].set_xlabel("")
    ax[2].set_xlabel("")
    ax[3].set_xlabel("")
    ax[4].set_xlabel("")
    ax[5].set_xlabel("Time (s)", fontsize=15)
    ax[0].set_ylabel("Hz", fontsize=15)
    ax[1].set_ylabel("Hz", fontsize=15)
    ax[2].set_ylabel("Hz", fontsize=15)
    ax[3].set_ylabel("Hz", fontsize=15)
    ax[4].set_ylabel("Hz", fontsize=15)
    ax[5].set_ylabel("Hz", fontsize=15)

    fig.tight_layout()

    if file_name is True:  # only if identical to "True" interactive
                           # blocking plotting
       plt.show()
    else:
        fig.savefig(file_name)

