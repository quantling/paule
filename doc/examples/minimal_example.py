import pickle
import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
import soundfile as sf

from paule import paule, util

tqdm.pandas()


paule_model = paule.Paule(device=torch.device('cuda'))

#target_acoustic = '000003-Wissenschaft.wav'
#target_acoustic = 'tino_essv2020.wav'
target_acoustic = 'test3.wav'


file = target_acoustic
SAVE_DIR = 'results'

save_file = SAVE_DIR + '/' + target_acoustic[:-4]

results = paule_model.plan_resynth(learning_rate_planning=0.01,
        learning_rate_learning=0.001,
        target_acoustic=target_acoustic,
        initialize_from="acoustic",
        objective="acoustic",
        n_outer=20, n_inner=50,
        #n_outer=2, n_inner=8,
        continue_learning=True,
        add_training_data=False,
        log_ii=1,
        log_semantics=True,
        n_batches=6, batch_size=8, n_epochs=5,
        log_gradients=False,
        plot=True, plot_save_file=save_file, seed=None,
        verbose=True)


# save model and optimizer
torch.save(paule_model.pred_model, f"{save_file}_pred_model.pt")
torch.save(paule_model.pred_optimizer, f"{save_file}_pred_optimizer.pt")

# save results without model and optimizer
with open(f"{save_file}.pkl", 'wb') as pfile:
    pickle.dump(results, pfile)


# save initial and planned flac
prod_sr = results.prod_sr
sig_initial = results.sig_steps[0]
sf.write(save_file + "_initial.flac", sig_initial, prod_sr)
prod_sig = results.prod_sig
sf.write(save_file + "_planned.flac", prod_sig, prod_sr)

# save svgs
planned_cp = results.planned_cp
path = save_file + '_svgs/'
if not os.path.exists(path):
    os.mkdir(path)
util.export_svgs(util.inv_normalize_cp(planned_cp), path=path)

# ffmpeg -r 80 -width 600 -i tract%05d.svg -i planned_0.flac planned_0.mp4
# /usr/bin/ffmpeg -r 80 -width 600 -i /home/tino/Documents/phd/projects/paule/results/000003-Wissenschaft_svgs/tract%05d.svg -i results/000003-Wissenschaft_planned.flac planned.mp4


# save loss plot
fig, ax = plt.subplots(figsize=(15, 8), facecolor="white")
ax.plot(results.planned_loss_steps, label="planned loss", c="C0")
ax.legend()
fig.savefig(f"{save_file}_loss.png")

fig, ax = plt.subplots(figsize=(15, 8), facecolor="white")
ax.plot(results.prod_loss_steps, label="produced mel loss", c="C1")
ax.plot(results.planned_mel_loss_steps, label="planned mel loss", c="C0")
ax.legend()
fig.savefig(f"{save_file}_loss_mel.png")

# save subloss plot
fig, ax = plt.subplots(figsize=(15, 8), facecolor="white")
ax.plot(results.vel_loss_steps, label="vel loss", c="C2")
ax.plot(results.jerk_loss_steps, label="jerk loss", c="C3")
ax.legend()
fig.savefig(f"{save_file}_loss_subloss.png")

# save semvec loss plot
fig, ax = plt.subplots(figsize=(15, 8), facecolor="white")
ax.plot(results.pred_semvec_loss_steps, label="planned semvec loss", c="C0")
ax.plot(results.prod_semvec_loss_steps, label="produced semvec loss", c="C1")
ax.legend()
fig.savefig(f"{save_file}_loss_semvec.png")

