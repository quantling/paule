"""

Continue Planning
=================
Assuming you have started to plan an utterance and stored the resulting
cp-trajectories and predictive model, this code shows how to continue the
planning process for more iterations.

"""

import pickle

import torch
from paule import paule


target_acoustic = 'test3.wav'
file = target_acoustic
SAVE_DIR = 'results'
DEVICE = torch.device('cpu')

save_file = SAVE_DIR + '/' + target_acoustic[:-4]  #+ '-continued'

# load model
pred_model = torch.load(f"{save_file}_pred_model.pt", map_location=DEVICE)
optimizer = torch.load(f"{save_file}_pred_optimizer.pt", map_location=DEVICE)


# load results (except for the last two)
with open(save_file + '.pkl', 'rb') as pfile:
    old_results = pickle.load(pfile)

initial_planned_cp = old_results.planned_cp

#paule_model = paule.Paule(pred_model=pred_model, pred_optimizer=optimizer, device=DEVICE)
paule_model = paule.Paule(pred_model=pred_model, pred_optimizer=None, device=DEVICE)

save_file += "-continued"

results = paule_model.plan_resynth(learning_rate_planning=0.01,
        learning_rate_learning=0.001,
        target_acoustic=target_acoustic,
        initial_cp=initial_planned_cp,
        initialize_from=None,
        objective="acoustic",
        n_outer=40, n_inner=50,
        continue_learning=True,
        add_training_data=False,
        log_ii=1,
        log_semantics=True,
        n_batches=6, batch_size=8, n_epochs=5,
        log_gradients=False,
        plot=save_file, seed=None, verbose=True)

