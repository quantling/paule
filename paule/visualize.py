"""
This module contains useful functions to visualize and safe results.

"""

import os

import soundfile as sf
from matplotlib import pyplot as plt
import librosa.display

from . import util


def vis_result(result, condition='prefix', folder='data'):
    """
    Stores all results in data/ folder.

    """
    CONDITION = condition
    (planned_cp, inv_cp, target_sig, target_mel, prod_sig, prod_mel, pred_mel, loss_steps,
     loss_mel_steps, loss_semvec_steps, loss_jerk_steps, loss_velocity_steps, loss_prod_steps) = result

    # produce from inv_cp
    inv_sig, inv_sr = util.speak(util.inv_normalize_cp(inv_cp))
    inv_mel = util.librosa_melspec(inv_sig, inv_sr)
    inv_mel = util.normalize_mel_librosa(inv_mel)

    target_sr = prod_sr = 44100
    sf.write(f'{folder}/{CONDITION}_planned.flac', 4 * prod_sig, prod_sr)
    sf.write(f'{folder}/{CONDITION}_inv.flac', 4 * inv_sig, inv_sr)
    sf.write(f'{folder}/{CONDITION}_target.flac', 4 * target_sig, target_sr)

    fig = plt.figure()
    plt.plot(loss_steps, c='blue', label='loss', lw=3)
    step_size = int(len(loss_jerk_steps)/len(loss_prod_steps))
    plt.plot(range(step_size - 1, (len(loss_prod_steps)) * step_size, step_size), loss_prod_steps, c='green', label='prod loss', lw=3)
    plt.plot(loss_jerk_steps, ls=':', c='blue', label='jerk loss')
    plt.plot(loss_velocity_steps, ls='--', c='blue', label='velocity loss')
    plt.plot(loss_mel_steps, ls='-', c='blue', label='mel loss')
    plt.plot(loss_semvec_steps, ls='-', c='yellow', label='semvec loss')
    #plt.hlines(1.5e-4, 0, len(loss_steps), ls='-', color='orange', label='asympt. train loss')
    #plt.hlines(5e-4, 0, len(loss_steps), ls=':', color='orange', label='asympt. test loss')
    plt.yscale('log')
    plt.legend()
    fig.savefig(f'{folder}/{CONDITION}_loss.png')


    fig = plt.figure()
    step_size = int(len(loss_jerk_steps)/len(loss_prod_steps))
    plt.plot(range(step_size - 1, (len(loss_prod_steps)) * step_size, step_size), loss_prod_steps, c='green', label='prod loss', lw=3)
    plt.yscale('log')
    plt.legend()
    fig.savefig(f'{folder}/{CONDITION}_prod-loss.png')


    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.68, 0.88, 0.30], xticklabels=[])
    ax2 = fig.add_axes([0.1, 0.36, 0.88, 0.30], xticklabels=[], sharex=ax1)
    ax3 = fig.add_axes([0.1, 0.04, 0.88, 0.30], xticklabels=[], sharex=ax1)
    img1 = inv_cp  #target['cp'].iloc[0]
    img2 = planned_cp
    img3 = img2 - img1
    ax1.plot(img1[:, 8:16])  # , label='tongue'
    ax1.plot(img1[:, 19:20], label='f0')
    ax1.set_ylabel("initial")
    ax2.plot(img2[:, 8:16], label='tongue')
    ax2.plot(img2[:, 19:20], label='f0')
    ax2.set_ylabel("optimized")
    ax3.plot(img3[:, 8:16], label='tongue')
    ax3.plot(img3[:, 19:20], label='f0')
    ax3.set_ylabel("difference")
    ax1.legend()
    fig.savefig(f'{folder}/{CONDITION}_cps.png')


    os.makedirs(f'{folder}/{CONDITION}_inv_svgs/')
    util.export_svgs(util.inv_normalize_cp(inv_cp), path=f'{folder}/{CONDITION}_inv_svgs/')
    system_call = f'cd {folder}/{CONDITION}_inv_svgs/; /usr/bin/ffmpeg -hide_banner -loglevel error -r 80 -width 600 -i tract%05d.svg -i ../{CONDITION}_inv.flac ../{CONDITION}_inv.mp4'
    os.system(system_call)

    os.makedirs(f'{folder}/{CONDITION}_planned_svgs/')
    util.export_svgs(util.inv_normalize_cp(planned_cp), path=f'{folder}/{CONDITION}_planned_svgs/')
    system_call = f'cd {folder}/{CONDITION}_planned_svgs/; /usr/bin/ffmpeg -hide_banner -loglevel error -r 80 -width 600 -i tract%05d.svg -i ../{CONDITION}_planned.flac ../{CONDITION}_planned.mp4'
    os.system(system_call)


    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.68, 0.88, 0.30], xticklabels=[])
    ax2 = fig.add_axes([0.1, 0.36, 0.88, 0.30], xticklabels=[], sharex=ax1)
    ax3 = fig.add_axes([0.1, 0.04, 0.88, 0.30], xticklabels=[], sharex=ax1)
    librosa.display.specshow(target_mel.T, y_axis='mel', sr=44100, hop_length=220, fmin=10, fmax=12000, ax=ax1, vmin=0, vmax=1)
    ax1.set_ylabel('target')
    librosa.display.specshow(pred_mel.T, y_axis='mel', sr=44100, hop_length=220, fmin=10, fmax=12000, ax=ax2, vmin=0, vmax=1)
    ax2.set_ylabel('predicted')
    librosa.display.specshow(prod_mel.T, y_axis='mel', sr=44100, hop_length=220, fmin=10, fmax=12000, ax=ax3, vmin=0, vmax=1)
    ax3.set_ylabel('produced')
    fig.savefig(f'{folder}/{CONDITION}_target_predicted_produced.png')


    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.68, 0.88, 0.30], xticklabels=[])
    ax2 = fig.add_axes([0.1, 0.36, 0.88, 0.30], xticklabels=[], sharex=ax1)
    ax3 = fig.add_axes([0.1, 0.04, 0.88, 0.30], xticklabels=[], sharex=ax1)
    librosa.display.specshow(target_mel.T, y_axis='mel', sr=44100, hop_length=220, fmin=10, fmax=12000, ax=ax1, vmin=0, vmax=1)
    ax1.set_ylabel('target')
    librosa.display.specshow(prod_mel.T, y_axis='mel', sr=44100, hop_length=220, fmin=10, fmax=12000, ax=ax2, vmin=0, vmax=1)
    ax2.set_ylabel('produced')
    librosa.display.specshow(inv_mel.T, y_axis='mel', sr=44100, hop_length=220, fmin=10, fmax=12000, ax=ax3, vmin=0, vmax=1)
    ax3.set_ylabel('initial')
    fig.savefig(f'{folder}/{CONDITION}_target_produced_initial.png')
    plt.show()  # this shows all saved figures

