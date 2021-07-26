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
from torch.nn import L1Loss
from matplotlib import pyplot as plt
import soundfile as sf
import noisereduce
import librosa
import librosa.display

# Set seed
torch.manual_seed(20200905)
random.seed(20200905)

tqdm.pandas()

from .util import speak, normalize_cp, inv_normalize_cp, normalize_mel_librosa, inv_normalize_mel_librosa, stereo_to_mono, librosa_melspec, pad_same_to_even_seq_length, RMSELoss
from . import models

DIR = os.path.dirname(__file__)


#def mse_loss(pred, target):
#    return torch.sum((target - pred).pow(2))

mse_loss = RMSELoss()
l1 = L1Loss()


def velocity_loss(pred):
    velocity_lag1 = pred[:, 1:, :] - pred[:, :-1, :]
    #velocity_lag2 = pred[:, 2:, :] - pred[:, :-2, :]
    #velocity_lag4 = pred[:, 4:, :] - pred[:, :-4, :]
    velocity_lag1 = velocity_lag1.pow(2)
    #velocity_lag2 = velocity_lag2.pow(2)
    #velocity_lag4 = velocity_lag4.pow(2)
    return l1(velocity_lag1, torch.zeros_like(velocity_lag1))
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
    return l1(jerk_lag1, torch.zeros_like(jerk_lag1))
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
            self.embedder = models.MelEmbeddingModel_MelSmoothResidualUpsampling(mel_smooth_layers=0).double()
            self.embedder.load_state_dict(torch.load(os.path.join(DIR, "pretrained_models/embedder/model_embed_model_0_4_180_8192_rmse_lr_00001_200.pt"), map_location=self.device))
        self.embedder = self.embedder.to(self.device)

        # CP GENerative MODEL
        if cp_gen_model:
            self.cp_gen_model = cp_gen_model
        else:
            self.cp_gen_model = models.Generator().double()
            self.cp_gen_model.load_state_dict(torch.load(os.path.join(DIR, "pretrained_models/cp_gan/conditional_trained_cp_generator_whole_critic_it_5_10_20_40_80_100_366.pt"), map_location=self.device))

        self.cp_gen_model = self.cp_gen_model.to(self.device)


        self.data = pd.read_pickle(os.path.join(DIR, 'data/full_data_tino.pkl'))

        self.pred_optimizer = torch.optim.Adam(self.pred_model.parameters(), lr=0.001)
        self.pred_criterion = mse_loss



    def plan_resynth(self, wave_name, *, inv_cp=None, target_semvec=None, plot=False, use_semantics=True, log_semantics=False, n_outer=40, n_inner=200, reduce_noise=False):
        if isinstance(wave_name, str):
            target_sig, target_sr = sf.read(wave_name)
            if len(target_sig.shape) == 2:
                target_sig = stereo_to_mono(target_sig)
            assert target_sr == 44100, 'sampling rate of wave name must be 44100'
            if reduce_noise:
                target_noise = target_sig[0:5000]
                target_sig = noisereduce.reduce_noise(target_sig, target_noise)
        else:
            target_sig = wave_name
            target_sr = 44100
        target_mel = librosa_melspec(target_sig, target_sr)
        target_mel = normalize_mel_librosa(target_mel)
        target_mel -= target_mel.min()
        target_mel.shape = (1, ) + target_mel.shape
        target_mel = torch.from_numpy(target_mel)
        target_mel= target_mel.to(self.device)

        if use_semantics or log_semantics:
            if target_semvec is None:
                with torch.no_grad():
                    target_semvec = self.embedder(target_mel, (torch.tensor(target_mel.shape[1]),))
            else:
                target_semvec = target_semvec.clone()
            target_semvec = target_semvec.to(self.device)


        if plot:
            librosa.display.specshow(target_mel[-1, :, :].detach().cpu().numpy().T, y_axis='mel', sr=44100, hop_length=220, fmin=10, fmax=12000)
            plt.show()

        learning_rate = 0.01

        # 1.0 create variables for logging
        loss_prod_steps = list()
        loss_steps = list()
        loss_mel_steps = list()
        loss_semvec_steps = list()
        loss_jerk_steps = list()
        loss_velocity_steps = list()


        # 1.1 predict inv_cp
        if inv_cp is None:
            xx = target_mel.detach().clone().to(self.device)
            with torch.no_grad():
                inv_cp = self.inv_model(xx)
            inv_cp = inv_cp.detach().cpu().numpy()
            inv_cp.shape = (inv_cp.shape[1], inv_cp.shape[2])
            del xx
        else:
            assert inv_cp.shape[0] == target_mel.shape[1] * 2

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
        if use_semantics or log_semantics:
            with torch.no_grad():
                pred_semvec = self.embedder(pred_mel, (torch.tensor(pred_mel.shape[1]),))
            assert pred_semvec.shape == target_semvec.shape
        initial_mel_loss = mse_loss(pred_mel, target_mel)
        initial_jerk_loss = jerk_loss(xx_new)
        initial_velocity_loss = velocity_loss(xx_new)
        if use_semantics or log_semantics:
            initial_semvec_loss = mse_loss(pred_semvec, target_semvec)
        jerk_weight = float(0.5 * initial_mel_loss / initial_jerk_loss)
        velocity_weight = float(0.5 * initial_mel_loss / initial_velocity_loss)
        if use_semantics or log_semantics:
            semvec_weight = float(1.0 * initial_mel_loss / initial_semvec_loss)
            del initial_semvec_loss

        del initial_mel_loss, initial_jerk_loss, initial_velocity_loss

        if use_semantics:
            def criterion(pred_mel, target_mel, pred_semvec, target_semvec, cps):
                return (velocity_weight * velocity_loss(cps) + jerk_weight * jerk_loss(cps)
                        + mse_loss(pred_mel, target_mel)
                        + semvec_weight * mse_loss(pred_semvec, target_semvec))
        else:
            def criterion(pred_mel, target_mel, cps):
                return (velocity_weight * velocity_loss(cps) + jerk_weight * jerk_loss(cps)
                        + mse_loss(pred_mel, target_mel))

        # continue learning
        for _ in tqdm(range(n_outer)):
            # imagine and plan
            for ii in range(n_inner):
                pred_mel = self.pred_model(xx_new)
                if use_semantics or (log_semantics and ii % 20 == 0):
                    seq_length = pred_mel.shape[1]
                    pred_semvec = self.embedder(pred_mel, (torch.tensor(seq_length),))

                if use_semantics:
                    discrepancy = criterion(pred_mel, target_mel, pred_semvec, target_semvec, xx_new)
                else:
                    discrepancy = criterion(pred_mel, target_mel, xx_new)
                discrepancy.backward()

                if ii % 20 == 0:
                    loss_steps.append(float(discrepancy.item()))
                    loss_mel_steps.append(float(mse_loss(pred_mel, target_mel)))
                    if use_semantics or log_semantics:
                        loss_semvec_steps.append(float(semvec_weight * mse_loss(pred_semvec, target_semvec)))
                    loss_jerk_steps.append(float(jerk_weight * jerk_loss(xx_new)))
                    loss_velocity_steps.append(float(velocity_weight * velocity_loss(xx_new)))


                grad = xx_new.grad.detach()
                xx_new = xx_new.detach()
                xx_new = xx_new - learning_rate * grad

                xx_new.requires_grad_()
                xx_new.retain_grad()

            # execute and continue learning
            sig, sr = speak(inv_normalize_cp(xx_new[-1, :, :].detach().cpu().numpy()))
            #if any(np.isnan(sig)):
            #    sig = np.zeros_like(sig)
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

