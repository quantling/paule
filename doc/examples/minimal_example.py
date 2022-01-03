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
        plot=save_file, seed=None, verbose=True)


# save model and optimizer
torch.save(paule_model.pred_model, f"{save_file}_pred_model.pt")
torch.save(paule_model.pred_optimizer, f"{save_file}_pred_optimizer.pt")

# save results without model and optimizer
with open(f"{save_file}.pkl", 'wb') as pfile:
    pickle.dump(results, pfile)

visualize.visualize_results(results, target_acoustic[:-4], SAVE_DIR)

