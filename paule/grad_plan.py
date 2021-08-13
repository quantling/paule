"""
Here we are doing gradient based planning on a target utterance (target mel spec).

For that we need an already trained predictive model (``pred_model``) that
predicts a mel spec from control parameters (cp) and an inverse model
(``inv_model``) for initialisation that generates cps from mel spec. In this
version we use a third model that maps a mel spec onto a fixed length semantic
vector (``embedder``).

The general idea works as follows:

    1. Initialize the cp for the full utterance time steps with the inverse model and
       the target mel spec. We initialize the target semvec by making a forward
       pass through the embedder.
    2. Predict the full utterance with the predictive model using the
       initialized cp and predict the semvecs from the predicted acoustics.
    3. Compare the predictions with the target mel spec to create a
       discrepancy. Do the same for the target semvec and the predicted semvec.
    4. Plan by using the discrepancies from acoustics and semvecs and jerk and
       velocity loss on the cps to update the cps.
    5. Do this 200 times
    6. after 200 planning updates continue the training of the predictive model
       with an actual synthesized version

"""

import pickle
import random
import time
import os

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.distributions.normal import Normal
from torch.utils import data
from torch.nn import L1Loss, MSELoss
from matplotlib import pyplot as plt
import soundfile as sf
import noisereduce
import librosa
import librosa.display

# Set seed
torch.manual_seed(20200905)
random.seed(20200905)

tqdm.pandas()

from .util import speak, normalize_cp, inv_normalize_cp, normalize_mel_librosa, inv_normalize_mel_librosa, stereo_to_mono, librosa_melspec, pad_same_to_even_seq_length, RMSELoss, mel_to_sig
from . import models

DIR = os.path.dirname(__file__)


#def mse_loss(pred, target):
#    return torch.sum((target - pred).pow(2))

mse_loss = RMSELoss()
l2 = MSELoss()
l1 = L1Loss()


def velocity_loss(pred):
    velocity_lag1 = pred[:, 1:, :] - pred[:, :-1, :]
    #velocity_lag2 = pred[:, 2:, :] - pred[:, :-2, :]
    #velocity_lag4 = pred[:, 4:, :] - pred[:, :-4, :]
    velocity_lag1 = velocity_lag1.pow(2)
    #velocity_lag2 = velocity_lag2.pow(2)
    #velocity_lag4 = velocity_lag4.pow(2)
    return l2(velocity_lag1, torch.zeros_like(velocity_lag1))
    #return torch.sum(velocity_lag1)  #+ 0.5 * torch.sum(velocity_lag2) + 0.25 * torch.sum(velocity_lag4)


def jerk_loss(pred):
    velocity_lag1 = pred[:, 1:, :] - pred[:, :-1, :]
    #velocity_lag2 = pred[:, 2:, :] - pred[:, :-2, :]
    #velocity_lag4 = pred[:, 4:, :] - pred[:, :-4, :]
    acceleration_lag1 = velocity_lag1[:, 1:, :] - velocity_lag1[:, :-1, :]
    #acceleration_lag2 = velocity_lag2[:, 1:, :] - velocity_lag2[:, :-1, :]
    #acceleration_lag4 = velocity_lag4[:, 1:, :] - velocity_lag4[:, :-1, :]
    jerk_lag1 = acceleration_lag1[:, 1:, :] - acceleration_lag1[:, :-1, :]
    #jerk_lag2 = acceleration_lag2[:, 1:, :] - acceleration_lag2[:, :-1, :]
    #jerk_lag4 = acceleration_lag4[:, 1:, :] - acceleration_lag4[:, :-1, :]
    jerk_lag1 = jerk_lag1.pow(2)
    #jerk_lag2 = jerk_lag2.pow(2)
    #jerk_lag4 = jerk_lag4.pow(2)
    return l2(jerk_lag1, torch.zeros_like(jerk_lag1))
    #return torch.sum(jerk_lag1)  #+ 0.25 * torch.sum(jerk_lag2)  + 1/16 * torch.sum(jerk_lag4)


class Paule():

    """
    The class paule keeps track of the the state associated with the Predictive
    Articulatory speech synthesise Using Lexical Embeddings.

    This state especially are the weights of the predictive, inverse and
    embedder model as well as data used for continue learning.

    """

    def __init__(self, *, pred_model=None, inv_model=None, embedder=None,
            cp_gen_model=None, mel_gen_model=None, device=torch.device('cpu')):

        # load the pred_model, inv_model and embedder here
        # for cpu
        self.device = device

        # PREDictive MODEL (cp -> mel)
        if pred_model:
            self.pred_model = pred_model
        else:
            self.pred_model = models.ForwardModel_MelTimeSmoothResidual(mel_smooth_layers=0).double()
            self.pred_model.load_state_dict(torch.load(os.path.join(DIR, "pretrained_models/predictive/model_pred_model_5_4_180_0_lr_0001_50_00001_50_000001_50_0000001_200.pt"), map_location=self.device))
        self.pred_model = self.pred_model.to(self.device)

        # INVerse MODEL (mel -> cp)
        if inv_model:
            self.inv_model = inv_model
        else:
            self.inv_model = models.InverseModel_MelTimeSmoothResidual().double()
            self.inv_model.load_state_dict(torch.load(os.path.join(DIR, "pretrained_models/inverse/inv_model_3_4_180_5_lr_0001_50_00001_50_000001_50_0000001_200.pt"), map_location=self.device))
        self.inv_model = self.inv_model.to(self.device)

        # EMBEDDER (mel -> semvec)
        if embedder:
            self.embedder = embedder
        else:
            self.embedder = models.MelEmbeddingModel_MelSmoothResidualUpsampling(mel_smooth_layers=3).double()
            self.embedder.load_state_dict(torch.load(os.path.join(DIR, "pretrained_models/embedder/model_recorded_embed_model_3_4_180_8192_rmse_lr_00001_400.pt"), map_location=self.device))
        self.embedder = self.embedder.to(self.device)

        # CP GENerative MODEL
        if cp_gen_model:
            self.cp_gen_model = cp_gen_model
        else:
            self.cp_gen_model = models.Generator().double()
            self.cp_gen_model.load_state_dict(torch.load(os.path.join(DIR, "pretrained_models/cp_gan/conditional_trained_cp_generator_whole_critic_it_5_10_20_40_80_100_366.pt"), map_location=self.device))
        self.cp_gen_model = self.cp_gen_model.to(self.device)

        # MEL GENerative MODEL
        if mel_gen_model:
            self.mel_gen_model = mel_gen_model
        else:
            self.mel_gen_model = models.Generator(output_size=60).double()
            self.mel_gen_model.load_state_dict(torch.load(os.path.join(DIR, "pretrained_models/mel_gan/conditional_trained_mel_generator_synthesized_critic_it_5_10_20_40_80_100_315.pt"), map_location=self.device))
        self.mel_gen_model = self.mel_gen_model.to(self.device)


        self.data = pd.read_pickle(os.path.join(DIR, 'data/full_data_tino.pkl'))

        self.pred_optimizer = torch.optim.Adam(self.pred_model.parameters(), lr=0.001)
        self.pred_criterion = mse_loss



    def plan_resynth(self, *, target_acoustic=None, target_semvec=None,
            target_seq_length=None, inv_cp=None,
            initialize_from='semvec', objective='acoustic_semvec',
            n_outer=40, n_inner=200, plot=False, log_semantics=False,
            reduce_noise=False, seed=None, verbose=False):
        """
        plans resynthesis cp trajectories.

        Parameters
        ----------
        target_acoustic : str; (target_sig, target_sr)
        target_semvec : torch.tensor
        target_seq_length : int (None)
        inv_cp : torch.tensor
        initialize_from : {'semvec', 'acoustic', None}
            can be None, if inv_cp are given
        objective : {'acoustic_semvec', 'acoustic', 'semvec'}
        n_outer : int (40)
        n_inner : int (200)
        plot : bool (False)
        log_semantics : bool (False)
        reduce_noise : bool (False)
        verbose : bool (False)

        """
        if seed:
            torch.manual_seed(seed)
        if target_acoustic is None and target_semvec is None:
            raise ValueError("Either target_acoustic or target_semvec has to be not None.")

        if isinstance(target_acoustic, str):
            target_sig, target_sr = sf.read(target_acoustic)
            if len(target_sig.shape) == 2:
                target_sig = stereo_to_mono(target_sig)
            assert target_sr == 44100, 'sampling rate of wave name must be 44100'
            if reduce_noise:
                target_noise = target_sig[0:5000]
                target_sig = noisereduce.reduce_noise(target_sig, target_noise)
        elif target_acoustic is None:
            pass
        else:
            target_sig, target_sr = target_acoustic

        if target_acoustic is None and target_seq_length is None:
            raise ValueError("if target_acoustic is None you need to give a target_seq_length")
        if target_acoustic is None:
            mel_gen_noise = torch.randn(1, 1, 100).to(self.device)
            if not isinstance(target_semvec, torch.Tensor):
                target_semvec = torch.tensor(target_semvec)
            mel_gen_semvec = target_semvec.view(1, 300).detach().clone()
            target_mel = self.mel_gen_model(mel_gen_noise, target_seq_length, mel_gen_semvec)
            target_mel = target_mel.detach().clone()
            target_sig, target_sr = mel_to_sig(target_mel.view(target_mel.shape[1], target_mel.shape[2]).cpu().numpy())
        else:
            target_mel = librosa_melspec(target_sig, target_sr)
            target_mel = normalize_mel_librosa(target_mel)
            target_mel -= target_mel.min()
            target_mel.shape = (1, ) + target_mel.shape
            target_mel = torch.from_numpy(target_mel)
            target_seq_length = target_mel.shape[1]
        target_mel= target_mel.to(self.device)


        if target_semvec is None:
            with torch.no_grad():
                target_semvec = self.embedder(target_mel, (torch.tensor(target_mel.shape[1]),))
        else:
            target_semvec = target_semvec.clone()
        target_semvec = target_semvec.to(self.device)


        if plot:
            librosa.display.specshow(target_mel[-1, :, :].detach().cpu().numpy().T, y_axis='mel', sr=44100, hop_length=220, fmin=10, fmax=12000)
            plt.show()

        learning_rate = 0.001

        # 1.0 create variables for logging
        loss_prod_steps = list()
        loss_steps = list()
        loss_mel_steps = list()
        loss_semvec_steps = list()
        loss_jerk_steps = list()
        loss_velocity_steps = list()


        # 1.1 predict inv_cp
        if inv_cp is None:
            if initialize_from == 'acoustic':
                xx = target_mel.detach().clone().to(self.device)
                with torch.no_grad():
                    inv_cp = self.inv_model(xx)
                inv_cp = inv_cp.detach().cpu().numpy()
                inv_cp.shape = (inv_cp.shape[1], inv_cp.shape[2])
                del xx
            elif initialize_from == 'semvec':
                cp_gen_noise = torch.randn(1, 1, 100).to(self.device)
                if not isinstance(target_semvec, torch.Tensor):
                    target_semvec = torch.tensor(target_semvec)
                cp_gen_semvec = target_semvec.view(1, 300).detach().clone()
                inv_cp = self.cp_gen_model(cp_gen_noise, 2 * target_seq_length, cp_gen_semvec)
                inv_cp = inv_cp.detach().cpu().numpy()
                inv_cp.shape = (inv_cp.shape[1], inv_cp.shape[2])
            else:
                raise ValueError("initialize_from has to be either 'acoustic' or 'semvec'")
        else:
            assert inv_cp.shape[0] == target_mel.shape[1] * 2 , f"inv_cp {inv_cp.shape[0]}, target_mel {target_mel.shape[1] * 2}"

        # 1.3 create initial xx
        #inv_cp = np.zeros_like(inv_cp)
        xx_new = inv_cp.copy()
        xx_new.shape = (1, xx_new.shape[0], xx_new.shape[1])
        xx_new = torch.from_numpy(xx_new)
        xx_new = xx_new.to(self.device)
        xx_new.requires_grad_()
        xx_new.retain_grad()

        # 1.4 adaptive weights for loss
        with torch.no_grad():
            pred_mel = self.pred_model(xx_new)
            pred_semvec = self.embedder(pred_mel, (torch.tensor(pred_mel.shape[1]),))
        assert pred_semvec.shape == target_semvec.shape
        initial_mel_loss = mse_loss(pred_mel, target_mel)
        initial_jerk_loss = jerk_loss(xx_new)
        initial_velocity_loss = velocity_loss(xx_new)
        initial_semvec_loss = mse_loss(pred_semvec, target_semvec)
        jerk_weight = float(0.5 * initial_mel_loss / initial_jerk_loss)
        velocity_weight = float(0.5 * initial_mel_loss / initial_velocity_loss)
        semvec_weight = float(1.0 * initial_mel_loss / initial_semvec_loss)
        del initial_semvec_loss

        del initial_mel_loss, initial_jerk_loss, initial_velocity_loss

        if objective == 'acoustic_semvec':
            def criterion(pred_mel, target_mel, pred_semvec, target_semvec, cps):
                return (velocity_weight * velocity_loss(cps) + jerk_weight * jerk_loss(cps)
                        + mse_loss(pred_mel, target_mel)
                        + semvec_weight * mse_loss(pred_semvec, target_semvec))
        elif objective == 'acoustic':
            def criterion(pred_mel, target_mel, cps):
                return (velocity_weight * velocity_loss(cps) + jerk_weight * jerk_loss(cps)
                        + mse_loss(pred_mel, target_mel))
        elif objective == 'semvec':
            def criterion(pred_semvec, target_semvec, cps):
                return (velocity_weight * velocity_loss(cps) + jerk_weight * jerk_loss(cps)
                        + semvec_weight * mse_loss(pred_semvec, target_semvec))
        else:
            raise ValueError("objective has to be one of 'acoustic_semvec', 'acoustic' or 'semvec'")

        # continue learning
        for ii_outer in tqdm(range(n_outer)):
            # imagine and plan
            for ii in range(n_inner):
                pred_mel = self.pred_model(xx_new)
                if objective in ('semvec', 'acoustic_semvec') or (log_semantics and ii % 20 == 0):
                    seq_length = pred_mel.shape[1]
                    pred_semvec = self.embedder(pred_mel, (torch.tensor(seq_length),))

                if objective == 'acoustic':
                    discrepancy = criterion(pred_mel, target_mel, xx_new)
                elif objective == 'acoustic_semvec':
                    discrepancy = criterion(pred_mel, target_mel, pred_semvec, target_semvec, xx_new)
                elif objective == 'semvec':
                    discrepancy = criterion(pred_semvec, target_semvec, xx_new)
                else:
                    raise ValueError(f'unkown objective {objective}')
                discrepancy.backward()

                if ii % 20 == 0:
                    loss_steps.append(float(discrepancy.item()))
                    loss_mel_steps.append(float(mse_loss(pred_mel, target_mel)))
                    if objective in ('semvec', 'acoustic_semvec') or log_semantics:
                        loss_semvec_steps.append(float(semvec_weight * mse_loss(pred_semvec, target_semvec)))
                    loss_jerk_steps.append(float(jerk_weight * jerk_loss(xx_new)))
                    loss_velocity_steps.append(float(velocity_weight * velocity_loss(xx_new)))

                grad = xx_new.grad.detach()
                if verbose:
                    print(f"grad.abs().max() {grad.abs().max()}")
                if grad.max() > 10:
                    if verbose:
                        print("WARNING: gradient is larger than 10")
                    grad[grad > 10.0] = 10.0
                if grad.min() < -10:
                    if verbose:
                        print("WARNING: gradient is smaller than -10")
                    grad[grad < -10.0] = -10.0

                xx_new = xx_new.detach()
                xx_new = xx_new - learning_rate * grad
                xx_new.requires_grad_()
                xx_new.retain_grad()
                if xx_new.max() > 1.05:
                    if verbose:
                        print("WARNING: planned cps are larger than 1.05")
                    xx_new = xx_new.detach()
                    xx_new[xx_new > 1.02] = 1.02
                    xx_new.requires_grad_()
                    xx_new.retain_grad()
                if xx_new.min() < -1.05:
                    if verbose:
                        print("WARNING: planned cps are smaller than -1.05")
                    xx_new = xx_new.detach()
                    xx_new[xx_new < -1.02] = -1.02
                    xx_new.requires_grad_()
                    xx_new.retain_grad()

            # adjust relative loss weights
            #if ii_outer % 5 == 0:
            #    mel_loss = mse_loss(pred_mel, target_mel)
            #    jerk_weight = float(0.5 * mel_loss / jerk_loss(xx_new))
            #    velocity_weight = float(0.5 * mel_loss / velocity_loss(xx_new))
            #    semvec_weight = float(1.0 * mel_loss / mse_loss(pred_semvec, target_semvec))
            #    del mel_loss

            # execute and continue learning
            sig, sr = speak(inv_normalize_cp(xx_new[-1, :, :].detach().cpu().numpy()))
            # TODO replace zeroing with constant schwa sound
            if any(np.isnan(sig)):
                print(f"xx_new.max() {xx_new.max()}")
                print(f"xx_new.min() {xx_new.min()}")
                print(f"WARNING: {np.sum(np.isnan(sig))} NaNs in synthesis: zero cps and sig")
                sig = np.zeros_like(sig)
                xx_new = torch.zeros_like(xx_new)
                xx_new.requires_grad_()
                xx_new.retain_grad()
            prod_mel = librosa_melspec(sig, sr)
            prod_mel = normalize_mel_librosa(prod_mel)
            prod_mel.shape = pred_mel.shape
            prod_mel = torch.from_numpy(prod_mel)
            prod_mel = prod_mel.to(self.device)

            # update with new sample
            self.pred_optimizer.zero_grad()
            pred_mel = self.pred_model(xx_new)
            pred_loss = self.pred_criterion(pred_mel, prod_mel)
            pred_loss.backward()
            self.pred_optimizer.step()
            prod_loss = mse_loss(prod_mel, target_mel)
            print(f"PRED LOSS: {discrepancy.item():.2e}   PROD LOSS: {prod_loss.item():.2e}")
            loss_prod_steps.append(prod_loss.item())

            # update with 10 old samples
            for index, (xx, norm_mel) in self.data[['cp_norm', 'melspec_norm']].sample(10).iterrows():
                self.pred_optimizer.zero_grad()

                xx = xx.copy()
                xx = pad_same_to_even_seq_length(xx)
                xx.shape = (1,) + xx.shape
                seq_length = xx.shape[0]
                xx = torch.from_numpy(xx)
                xx = xx.to(self.device)
                pred_mel = self.pred_model(xx)
                
                norm_mel = norm_mel.copy()
                norm_mel.shape = (1,) + norm_mel.shape
                norm_mel -= norm_mel.min()
                norm_mel = torch.tensor(norm_mel)
                norm_mel = norm_mel.to(self.device)
                pred_loss = self.pred_criterion(pred_mel, norm_mel)
                pred_loss.backward()
                self.pred_optimizer.step()
        planned_cp = xx_new[-1, :, :].detach().cpu().numpy()
        prod_sig = sig
        target_mel = target_mel[-1, :, :].detach().cpu().numpy()
        prod_mel = prod_mel[-1, :, :].detach().cpu().numpy()
        with torch.no_grad():
            pred_mel = self.pred_model(xx_new)
        pred_mel = pred_mel[-1, :, :].detach().cpu().numpy()
        return (planned_cp, inv_cp, target_sig, target_mel, prod_sig, prod_mel, pred_mel, loss_steps,
                loss_mel_steps, loss_semvec_steps, loss_jerk_steps, loss_velocity_steps, loss_prod_steps)

