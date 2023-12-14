import pickle
import os

import torch
import numpy as np

from paule import paule, util, visualize


util.download_pretrained_weights()

#paule_model = paule.Paule(device=torch.device('cuda'), use_somatosensory_feedback=True)
paule_model = paule.Paule(device=torch.device('cpu'), use_somatosensory_feedback=True)

#target_acoustic = '000003-Wissenschaft.wav'
#target_acoustic = 'froehliche_weihnachten.flac'
#target_acoustic = 'merry_christmas.flac'
target_acoustic = 'frohes_neues_jahr.flac'

SAVE_DIR = 'results'

if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

save_file = SAVE_DIR + '/' + target_acoustic[:-4]

#past_cp = np.load('past_cp.npy')

results = paule_model.plan_resynth(learning_rate_planning=0.01,
        learning_rate_learning=0.001,
        target_acoustic=target_acoustic,
        initialize_from="acoustic",
        #objective="acoustic",
        objective="acoustic_semvec",
        #past_cp=past_cp,
        past_cp=None,
        n_outer=10, n_inner=25,
        #n_outer=4, n_inner=8,
        continue_learning=True,
        add_training_data_pred=False,
        log_ii=1,
        log_semantics=True,
        n_batches=3, batch_size=8, n_epochs=10,
        #n_batches=1, batch_size=8, n_epochs=2,
        log_gradients=False,
        plot=save_file, seed=None, verbose=True)


# save model and optimizer
torch.save(paule_model.pred_model.state_dict(), f"{save_file}_pred_model.pt")
torch.save(paule_model.pred_optimizer.state_dict(), f"{save_file}_pred_optimizer.pt")

# save results without model and optimizer
with open(f"{save_file}.pkl", 'wb') as pfile:
    pickle.dump(results, pfile)

visualize.visualize_results(results, os.path.basename(save_file), SAVE_DIR)

