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
from collections import namedtuple

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.distributions.normal import Normal
from torch.utils import data
from torch.nn import L1Loss, MSELoss
from matplotlib import pyplot as plt
import soundfile as sf
import librosa.display
from matplotlib import cm

# Set seed
torch.manual_seed(20200905)
random.seed(20200905)

tqdm.pandas()

from .util import (speak, inv_normalize_cp, normalize_mel_librosa,
        stereo_to_mono, librosa_melspec, RMSELoss, mel_to_sig,
        pad_batch_online)
from .models import (ForwardModel, InverseModel_MelTimeSmoothResidual,
        MelEmbeddingModel_MelSmoothResidualUpsampling, Generator)

DIR = os.path.dirname(__file__)


rmse_loss = RMSELoss(eps=0)
l2 = MSELoss()
l1 = L1Loss()

PlanningResults = namedtuple('PlanningResults', "planned_cp, initial_cp, target_sig, target_sr, target_mel, prod_sig, prod_sr, prod_mel, pred_mel, prod_loss_steps, planned_loss_steps, planned_mel_loss_steps, vel_loss_steps, jerk_loss_steps, pred_semvec_loss_steps, prod_semvec_loss_steps, cp_steps, pred_semvec_steps, prod_semvec_steps, grad_steps, sig_steps, prod_mel_steps, pred_mel_steps, model_loss")


def get_vel_acc_jerk(trajectory, *, lag=1):
    """returns (velocity, acceleration, jerk) tuple"""
    velocity = (trajectory[:, lag:, :] - trajectory[:, :-lag, :]) / lag
    acc = (velocity[:, 1:, :] - velocity[:, :-1, :]) / 1.0
    jerk = (acc[:, 1:, :] - acc[:, :-1, :]) / 1.0
    return velocity, acc, jerk



def velocity_jerk_loss(pred, loss, *, guiding_factor=None):
    """returns (velocity_loss, jerk_loss) tuple"""
    vel1, acc1, jerk1 = get_vel_acc_jerk(pred)
    vel2, acc2, jerk2 = get_vel_acc_jerk(pred, lag=2)
    vel4, acc4, jerk4 = get_vel_acc_jerk(pred, lag=4)

    loss = rmse_loss

    # in the lag calculation higher lags are already normalised to standard
    # units
    if guiding_factor is None:
        velocity_loss = (loss(vel1, torch.zeros_like(vel1))
                         + loss(vel2, torch.zeros_like(vel2))
                         + loss(vel4, torch.zeros_like(vel4)))
        jerk_loss = (loss(jerk1, torch.zeros_like(jerk1))
                     + loss(jerk2, torch.zeros_like(jerk2))
                     + loss(jerk4, torch.zeros_like(jerk4)))
    else:
        assert 0.0 < guiding_factor < 1.0
        velocity_loss = (loss(vel1, guiding_factor * vel1.detach().clone())
                         + loss(vel2, guiding_factor * vel2.detach().clone())
                         + loss(vel4, guiding_factor * vel4.detach().clone()))
        jerk_loss = (loss(jerk1, guiding_factor * jerk1.detach().clone())
                     + loss(jerk2, guiding_factor * jerk2.detach().clone())
                     + loss(jerk4, guiding_factor * jerk4.detach().clone()))

    return velocity_loss, jerk_loss



class HardTanhStraightThrough(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u):
        return torch.nn.functional.hardtanh(u)

    @staticmethod
    def backward(ctx, dx):
        return dx

def hardtanh_straight_through(u):
    return HardTanhStraightThrough.apply(u)


class SoftsignStraightThrough(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u):
        return torch.nn.functional.softsign(u)

    @staticmethod
    def backward(ctx, dx):
        return dx

def tanh_straight_through(u):
    return SoftsignStraightThrough.apply(u)

class TanhStraightThrough(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u):
        return torch.nn.functional.tanh(u)

    @staticmethod
    def backward(ctx, dx):
        return dx

def tanh_straight_through(u):
    return TanhStraightThrough.apply(u)


class Paule():
    """
    The class paule keeps track of the the state associated with the Predictive
    Articulatory speech synthesise Using Lexical Embeddings.
    This state especially are the weights of the predictive, inverse and
    embedder model as well as data used for continue learning.
    """

    def __init__(self, *, pred_model=None, pred_optimizer=None, inv_model=None,
                 embedder=None, cp_gen_model=None, mel_gen_model=None,
                 continue_data=None, device=torch.device('cpu')):

        # load the pred_model, inv_model and embedder here
        # for cpu
        self.device = device

        # PREDictive MODEL (cp -> mel)
        if pred_model:
            self.pred_model = pred_model
        else:
            self.pred_model = ForwardModel(num_lstm_layers=1, hidden_size=720).double()
            self.pred_model.load_state_dict(
                torch.load(os.path.join(DIR, "pretrained_models/predictive/pred_model_common_voice_1_720_lr_0001_50_00001_100.pt"),
                           map_location=self.device))
        self.pred_model = self.pred_model.to(self.device)

        # INVerse MODEL (mel -> cp)
        if inv_model:
            self.inv_model = inv_model
        else:
            self.inv_model = InverseModel_MelTimeSmoothResidual(num_lstm_layers=1, hidden_size=720).double()
            self.inv_model.load_state_dict(
                torch.load(os.path.join(DIR, "pretrained_models/inverse/inv_model_common_voice_3_1_720_5_lr_0001_50_00001_100.pt"),
                           map_location=self.device))
        self.inv_model = self.inv_model.to(self.device)

        # EMBEDDER (mel -> semvec)
        if embedder:
            self.embedder = embedder
        else:
            self.embedder = MelEmbeddingModel_MelSmoothResidualUpsampling(mel_smooth_layers=0, num_lstm_layers=1,
                                                                          hidden_size=720).double()
            self.embedder.load_state_dict(torch.load(
                os.path.join(DIR, "pretrained_models/embedder/embed_model_common_voice_syn_rec_0_1_720_8192_rmse_lr_00001_200.pt"),
                map_location=self.device))
        self.embedder = self.embedder.to(self.device)

        # CP GENerative MODEL
        if cp_gen_model:
            self.cp_gen_model = cp_gen_model
        else:
            self.cp_gen_model = Generator().double()
            self.cp_gen_model.load_state_dict(torch.load(
                os.path.join(DIR, "pretrained_models/cp_gan/conditional_trained_cp_generator_whole_critic_it_5_10_20_40_80_100_415.pt"),
                map_location=self.device))
        self.cp_gen_model = self.cp_gen_model.to(self.device)
        self.cp_gen_model.eval()

        # MEL GENerative MODEL
        if mel_gen_model:
            self.mel_gen_model = mel_gen_model
        else:
            self.mel_gen_model = Generator(output_size=60).double()
            self.mel_gen_model.load_state_dict(torch.load(os.path.join(DIR,
                                                                       "pretrained_models/mel_gan/conditional_trained_mel_generator_synthesized_critic_it_5_10_20_40_80_100_400.pt"),
                                                          map_location=self.device))
        self.mel_gen_model = self.mel_gen_model.to(self.device)
        self.mel_gen_model.eval()

        # DATA to continue learning
        # created from geco_embedding_preprocessed_balanced_vectors_checked_extrem_long_removed_valid_matched_prot4
        # self.data = pd.read_pickle(os.path.join(DIR, 'data/continue_data.pkl'))

        #self.data = pd.read_pickle(os.path.join(DIR, 'data/continue_data.pkl'))
        self.continue_data = continue_data

        if pred_optimizer:
            self.pred_optimizer = pred_optimizer
        else:
            self.pred_optimizer = torch.optim.Adam(self.pred_model.parameters(), lr=0.001)
        self.pred_criterion = rmse_loss


    def plan_resynth(self, *, learning_rate_planning=0.01, learning_rate_learning=None,
                     target_acoustic=None,
                     target_semvec=None,
                     target_seq_length=None,
                     initial_cp=None,
                     initialize_from="acoustic",
                     objective="acoustic",
                     n_outer=10, n_inner=50,
                     continue_learning=True,
                     add_training_data=False,
                     log_ii=None,
                     log_semantics=False,
                     n_batches=6, batch_size=8, n_epochs=5,
                     log_gradients=False,
                     plot=False, plot_save_file="test", seed=None,
                     verbose=False):
        """
        plans resynthesis cp trajectories.

        Parameters
        ==========
        learning_rate_planning : float
            learning rate for updating cps
        learning_rate_learning : float
            learning rate for updating predictive model
        target_acoustic : str
            (target_sig, target_sr)
        target_semvec : torch.tensor
        target_seq_length : int (None)
        initial_cp : torch.tensor
        initialize_from : {'semvec', 'acoustic', None}
            can be None, if initial_cp are given
        objective : {'acoustic_semvec', 'acoustic', 'semvec'}
        n_outer : int (40)
        n_inner : int (100)
        continue_learning : bool (True)
            update predictive model with synthesized acoustics
        add_training_data : bool (False)
            update solely on produced acoustics during training or add training data
        log_ii : int
            log results and synthesize audio after ii number of inner iterations
        plot : bool (False)
        plot_save_file : str
            file_name to store plots
        seed : int random seed
        verbose : bool (False)

        """

        if seed:
            torch.manual_seed(seed)
            random.seed(seed)

        if target_acoustic is None and target_semvec is None:
            raise ValueError("Either target_acoustic or target_semvec has to be not None.")

        if learning_rate_learning:
            for param_group in self.pred_optimizer.param_groups:
                param_group['lr'] = learning_rate_learning

        if log_ii is None:
            log_ii = n_inner

        assert log_ii > 0 and log_ii <= n_inner, 'results can only be logged between first and last planning step'

        if isinstance(target_acoustic, str):
            target_sig, target_sr = sf.read(target_acoustic)
            if len(target_sig.shape) == 2:
                target_sig = stereo_to_mono(target_sig)
            # assert target_sr == 44100, 'sampling rate of wave name must be 44100'

        elif target_acoustic is None:
            pass
        else:
            target_sig, target_sr = target_acoustic

        if target_acoustic is None and (target_seq_length is None or target_semvec is None):
            raise ValueError("if target_acoustic is None you need to give a target_seq_length and a target_semvec")

        elif target_acoustic is None:
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
            target_mel.shape = (1,) + target_mel.shape
            target_mel = torch.from_numpy(target_mel)
            target_seq_length = target_mel.shape[1]

        target_mel = target_mel.to(self.device)

        if target_semvec is None:
            with torch.no_grad():
                target_semvec = self.embedder(target_mel, (torch.tensor(target_mel.shape[1]),))
        else:
            target_semvec = target_semvec.clone()
        target_semvec = target_semvec.to(self.device)

        # def constrained_criterion(xx_new, target_mel):
        #    pred_mel = self.pred_model(xx_new)
        #    mel_loss = rmse_loss(pred_mel,target_mel)
        #    velocity_loss, jerk_loss = velocity_jerk_loss(xx_new)
        #    loss = mel_loss + velocity_loss + jerk_loss
        #    return loss, mel_loss, velocity_loss, jerk_loss, pred_mel

        # 1.1 predict initial_cp
        if initial_cp is None:
            if initialize_from == "acoustic":
                xx = target_mel.detach().clone().to(self.device)
                with torch.no_grad():
                    initial_cp = self.inv_model(xx)
                initial_cp = initial_cp.detach().cpu().numpy().clip(min=-1, max=1)
                initial_cp.shape = (initial_cp.shape[1], initial_cp.shape[2])
                del xx
            elif initialize_from == "semvec":
                cp_gen_noise = torch.randn(1, 1, 100).to(self.device)
                if not isinstance(target_semvec, torch.Tensor):
                    target_semvec = torch.tensor(target_semvec)
                cp_gen_semvec = target_semvec.view(1, 300).detach().clone()
                initial_cp = self.cp_gen_model(cp_gen_noise, 2 * target_seq_length, cp_gen_semvec)
                initial_cp = initial_cp.detach().cpu().numpy()
                initial_cp.shape = (initial_cp.shape[1], initial_cp.shape[2])
            else:
                raise ValueError("initialize_from has to be either 'acoustic' or 'semvec'")

        else:
            assert initial_cp.shape[0] == target_mel.shape[
                1] * 2, f"initial_cp {initial_cp.shape[0]}, target_mel {target_mel.shape[1] * 2}"

        # 1.3 create initial xx
        # initial_cp = np.zeros_like(initial_cp)
        xx_new = initial_cp.copy()
        xx_new.shape = (1, xx_new.shape[0], xx_new.shape[1])
        xx_new = torch.from_numpy(xx_new)
        xx_new = xx_new.to(self.device)
        xx_new.requires_grad_()
        xx_new.retain_grad()

        if objective == 'acoustic_semvec':
            def criterion(pred_mel, target_mel, pred_semvec, target_semvec, cps):
                mel_loss = rmse_loss(pred_mel, target_mel)
                # mel_loss_w, time_loss,channel_loss,energy_loss = mel_w_loss(pred_mel,target_mel)
                semvec_loss = rmse_loss(pred_semvec, target_semvec)
                velocity_loss, jerk_loss = velocity_jerk_loss(cps, rmse_loss)
                loss = mel_loss + velocity_loss + jerk_loss + semvec_loss
                return loss, mel_loss, velocity_loss, jerk_loss, semvec_loss

        elif objective == 'acoustic':
            def criterion(pred_mel, target_mel, cps):
                mel_loss = rmse_loss(pred_mel, target_mel)
                # mel_loss_w, time_loss,channel_loss,energy_loss = mel_w_loss(pred_mel,target_mel)
                velocity_loss, jerk_loss = velocity_jerk_loss(cps, rmse_loss)
                loss = mel_loss + velocity_loss + jerk_loss
                return loss, mel_loss, velocity_loss, jerk_loss


        elif objective == 'semvec':
            def criterion(pred_semvec, target_semvec, cps):
                semvec_loss = rmse_loss(pred_semvec, target_semvec)
                velocity_loss, jerk_loss = velocity_jerk_loss(cps, rmse_loss)
                loss = velocity_loss + jerk_loss + semvec_loss
                return loss, velocity_loss, jerk_loss, semvec_loss

        else:
            raise ValueError("objective has to be one of 'acoustic_semvec', 'acoustic' or 'semvec'")

        # 1.0 create variables for logging
        prod_loss_steps = list()
        planned_loss_steps = list()
        planned_mel_loss_steps = list()
        vel_loss_steps = list()
        jerk_loss_steps = list()
        pred_semvec_loss_steps = list()
        prod_semvec_loss_steps = list()

        cp_steps = list()
        pred_semvec_steps = list()
        prod_semvec_steps = list()
        grad_steps = list()
        sig_steps = list()
        pred_mel_steps = list()
        prod_mel_steps = list()
        model_loss = list()
        optimizer = torch.optim.Adam([xx_new], lr=learning_rate_planning)

        # continue learning
        start_time = time.time()
        for ii_outer in tqdm(range(n_outer)):
            # imagine and plan
            pred_mel_steps_ii = list()
            prod_mel_steps_ii = list()
            cp_steps_ii = list()
            pred_semvec_steps_ii = list()
            prod_semvec_steps_ii = list()

            for ii in range(n_inner):
                pred_mel = self.pred_model(xx_new)

                if objective in ('semvec', 'acoustic_semvec') or (
                        log_semantics and ((ii + 1) % log_ii == 0 or ii == 0)):
                    seq_length = pred_mel.shape[1]
                    pred_semvec = self.embedder(pred_mel, (torch.tensor(seq_length),))
                    pred_semvec_steps_ii.append(pred_semvec[-1, :].detach().cpu().numpy().copy())

                # discrepancy,mel_loss, vel_loss, jerk_loss, pred_mel = constrained_criterion(tanh_straight_through(xx_new), target_mel)
                if objective == 'acoustic':
                    discrepancy, mel_loss, vel_loss, jerk_loss = criterion(pred_mel, target_mel, xx_new)
                    if (ii + 1) % log_ii == 0 or ii == 0:
                        planned_loss_steps.append(float(discrepancy.item()))
                        planned_mel_loss_steps.append(float(mel_loss.item()))
                        vel_loss_steps.append(float(vel_loss.item()))
                        jerk_loss_steps.append(float(jerk_loss.item()))

                        if log_semantics:
                            semvec_loss = float(rmse_loss(pred_semvec, target_semvec).item())
                            pred_semvec_loss_steps.append(semvec_loss)
                            

                    if verbose:
                        print("Iteration %d" % ii)
                        print("Planned Loss: ", float(discrepancy.item()))
                        print("Mel Loss: ", float(mel_loss.item()))
                        print("Vel Loss: ", float(vel_loss.item()))
                        print("Jerk Loss: ", float(jerk_loss.item()))
                        if log_semantics:
                            print("Semvec Loss: ", float(semvec_loss))


                elif objective == 'acoustic_semvec':
                    discrepancy, mel_loss, vel_loss, jerk_loss, semvec_loss = criterion(pred_mel, target_mel,
                                                                                        pred_semvec, target_semvec,
                                                                                        xx_new)
                    if (ii + 1) % log_ii == 0 or ii == 0:
                        planned_loss_steps.append(float(discrepancy.item()))
                        planned_mel_loss_steps.append(float(mel_loss.item()))
                        vel_loss_steps.append(float(vel_loss.item()))
                        jerk_loss_steps.append(float(jerk_loss.item()))
                        pred_semvec_loss_steps.append(float(semvec_loss.item()))

                    if verbose:
                        print("Iteration %d" % ii)
                        print("Planned Loss: ", float(discrepancy.item()))
                        print("Mel Loss: ", float(mel_loss.item()))
                        print("Vel Loss: ", float(vel_loss.item()))
                        print("Jerk Loss: ", float(jerk_loss.item()))
                        print("Semvec Loss: ", float(semvec_loss.item()))

                elif objective == 'semvec':
                    discrepancy, vel_loss, jerk_loss, semvec_loss = criterion(pred_semvec, target_semvec, xx_new)
                    mel_loss = rmse_loss(pred_mel, target_mel)
                    
                    if (ii + 1) % log_ii == 0 or ii == 0:
                        planned_loss_steps.append(float(discrepancy.item()))
                        vel_loss_steps.append(float(vel_loss.item()))
                        jerk_loss_steps.append(float(jerk_loss.item()))
                        pred_semvec_loss_steps.append(float(semvec_loss.item()))
                        planned_mel_loss_steps.append(float(mel_loss.item()))

                    if verbose:
                        print("Iteration %d" % ii)
                        print("Planned Loss: ", float(discrepancy.item()))
                        print("Mel Loss: ", float(mel_loss.item()))
                        print("Vel Loss: ", float(vel_loss.item()))
                        print("Jerk Loss: ", float(jerk_loss.item()))
                        print("Semvec Loss: ", float(semvec_loss.item()))


                else:
                    raise ValueError(f'unkown objective {objective}')

                discrepancy.backward()

                # if verbose:
                # print(f"grad.abs().max() {grad.abs().max()}")
                if xx_new.grad.max() > 10:
                    if verbose:
                        print("WARNING: gradient is larger than 10")
                if xx_new.grad.min() < -10:
                    if verbose:
                        print("WARNING: gradient is smaller than -10")

                if log_gradients:
                    grad_steps.append(xx_new.grad.detach().clone())

                if (ii + 1) % log_ii == 0 or ii == 0:
                    xx_new_numpy = xx_new[-1, :, :].detach().cpu().numpy().copy()
                    cp_steps_ii.append(xx_new_numpy)

                    sig, sr = speak(inv_normalize_cp(xx_new_numpy))
                    sig_steps.append(sig)

                    prod_mel = librosa_melspec(sig, sr)
                    prod_mel = normalize_mel_librosa(prod_mel)
                    prod_mel_steps_ii.append(prod_mel.copy())

                    prod_mel.shape = pred_mel.shape
                    prod_mel = torch.from_numpy(prod_mel)
                    prod_mel = prod_mel.to(self.device)
                    pred_mel_steps_ii.append(pred_mel[-1, :, :].detach().cpu().numpy().copy())

                    prod_loss = rmse_loss(prod_mel, target_mel)
                    prod_loss_steps.append(float(prod_loss.item()))

                    if verbose:
                        print("Produced Mel Loss: ", float(prod_loss.item()))

                    if objective in ('semvec', 'acoustic_semvec') or log_semantics:
                        prod_semvec = self.embedder(prod_mel, (torch.tensor(prod_mel.shape[1]),))
                        prod_semvec_steps_ii.append(prod_semvec[-1, :].detach().cpu().numpy().copy())

                        prod_semvec_loss = rmse_loss(prod_semvec, target_semvec)
                        prod_semvec_loss_steps.append(float(prod_semvec_loss.item()))

                        if verbose:
                            print("Produced Semvec Loss: ", float(prod_semvec_loss.item()))
                            print("")
                    else:
                        if verbose:
                            print("")

                optimizer.step()
                with torch.no_grad():
                    # xx_new.data = torch.maximum(-1*torch.ones_like(xx_new),
                    # torch.minimum(torch.ones_like(xx_new),
                    #              xx_new.data - learning_rate * xx_new.grad))
                    # xx_new.data = (xx_new.data - learning_rate * xx_new.grad)
                    # xx_new.data = (xx_new.data - learning_rate * xx_new.grad).clamp(-1,1)
                    xx_new.data = xx_new.data.clamp(-1.05, 1.05) # clamp between -1.05 and 1.05

                xx_new.grad.zero_()

                if ii_outer == 0 and ii == 0:
                    initial_pred_mel = pred_mel[-1, :, :].detach().cpu().numpy().copy()
                    initial_prod_mel = prod_mel[-1, :, :].detach().cpu().numpy().copy()

            if plot:
                target_mel_ii = target_mel[-1, :, :].detach().cpu().numpy().copy()
                prod_mel_ii = prod_mel[-1, :, :].detach().cpu().numpy().copy()
                pred_mel_ii = pred_mel[-1, :, :].detach().cpu().numpy().copy()

                fig, ax = plt.subplots(nrows=5, figsize=(15, 15), facecolor="white")
                librosa.display.specshow(target_mel_ii.T, y_axis='mel', x_axis='time', sr=44100, hop_length=220,
                                         ax=ax[0], cmap=cm.magma)
                librosa.display.specshow(initial_pred_mel.T, y_axis='mel', x_axis='time', sr=44100, hop_length=220,
                                         ax=ax[1], cmap=cm.magma)
                librosa.display.specshow(initial_prod_mel.T, y_axis='mel', x_axis='time', sr=44100, hop_length=220,
                                         ax=ax[2], cmap=cm.magma)
                librosa.display.specshow(pred_mel_ii.T, y_axis='mel', x_axis='time', sr=44100, hop_length=220, ax=ax[3],
                                         cmap=cm.magma)
                librosa.display.specshow(prod_mel_ii.T, y_axis='mel', x_axis='time', sr=44100, hop_length=220, ax=ax[4],
                                         cmap=cm.magma)

                ax[0].set_xticks([])
                ax[1].set_xticks([])
                ax[2].set_xticks([])
                ax[3].set_xticks([])
                ax[0].set_xlabel("")
                ax[1].set_xlabel("")
                ax[2].set_xlabel("")
                ax[3].set_xlabel("")
                ax[4].set_xlabel("Time (s)", fontsize=15)
                ax[0].set_ylabel("Hz", fontsize=15)
                ax[1].set_ylabel("Hz", fontsize=15)
                ax[2].set_ylabel("Hz", fontsize=15)
                ax[3].set_ylabel("Hz", fontsize=15)
                ax[4].set_ylabel("Hz", fontsize=15)

                ax[0].set_title("Target", fontsize=18)
                ax[1].set_title("Inverse Prediction", fontsize=18)
                ax[2].set_title("Inverse Produced", fontsize=18)
                ax[3].set_title("Planned Prediction", fontsize=18)
                ax[4].set_title("Planned Produced", fontsize=18)

                # plt.show()
                fig.savefig(plot_save_file + "_%d" % ii_outer + ".png")

            prod_mel_steps.append(prod_mel_steps_ii)
            cp_steps.append(cp_steps_ii)
            pred_mel_steps.append(pred_mel_steps_ii)
            pred_semvec_steps.append(pred_semvec_steps_ii)
            prod_semvec_steps.append(prod_semvec_steps_ii)

            # execute and continue learning
            if continue_learning:
                produced_data = pd.DataFrame(columns=['cp_norm', 'melspec_norm_synthesized'])
                produced_data["cp_norm"] = cp_steps_ii
                produced_data["melspec_norm_synthesized"] = prod_mel_steps_ii

                if add_training_data:
                    # update with new sample
                    if len(produced_data) < int(0.5 * batch_size) * n_batches:
                        batch_train_new = random.sample(range(len(produced_data)), k=len(produced_data))
                        batch_train = random.sample(range(len(self.data)),
                                                    k=batch_size * n_batches - len(produced_data))

                    else:
                        batch_train = random.sample(range(len(self.data)), k=int(0.5 * batch_size) * n_batches)
                        batch_train_new = random.sample(range(len(produced_data)), k=int(0.5 * batch_size) * n_batches)

                    train_data_samples = self.data[['cp_norm', 'melspec_norm_synthesized']].iloc[
                        batch_train].reset_index(drop=True)
                    produced_data_samples = produced_data.iloc[batch_train_new].reset_index(drop=True)

                    continue_data = pd.concat([train_data_samples, produced_data_samples])
                    # ensure in each batch same amount new and old samples if possible
                    # continue_data.sort_index(inplace=True)
                    # sort by length
                    continue_data["lens_input"] = np.array(continue_data["cp_norm"].apply(len), dtype=int)
                    continue_data["lens_output"] = np.array(continue_data["melspec_norm_synthesized"].apply(len),
                                                            dtype=int)
                    continue_data.sort_values(by="lens_input", inplace=True)

                    n_train_batches = n_batches
                else:
                    batch_train_new = random.sample(range(len(produced_data)), k=len(produced_data))
                    produced_data_samples = produced_data.iloc[batch_train_new]

                    if len(produced_data_samples) < batch_size * n_batches:
                        # rolling batching (fill one more batch with already seen samples)
                        full_batches = len(produced_data_samples) // batch_size
                        samples_to_fill_batch = int(abs(len(produced_data_samples) - batch_size * full_batches + 1))
                        if samples_to_fill_batch > 0:
                            fill_data = produced_data_samples.iloc[:samples_to_fill_batch].copy()
                            continue_data = pd.concat([produced_data_samples, fill_data])
                            if verbose:
                                print("Not enough data produced...Training on %d instead of %d batches!" % (
                                (full_batches + 1), n_batches))
                                if len(continue_data) % batch_size > 0:
                                    print("Reduced last batch...Batchsize %d instead of %d!" % (
                                    (len(continue_data) % batch_size), batch_size))

                            n_train_batches = full_batches + 1

                        else:
                            continue_data = produced_data_samples.copy()
                            print("Not enough produced data...Training on %d instead of %d batches!" % (
                            (full_batches + 1), n_batches))
                            n_train_batches = full_batches
                    else:
                        continue_data = produced_data_samples.copy()
                        n_train_batches = n_batches

                    # ensure in each batch same amount new and old samples if possible
                    # continue_data.sort_index(inplace=True)
                    # sort by length
                    continue_data["lens_input"] = np.array(continue_data["cp_norm"].apply(len), dtype=int)
                    continue_data["lens_output"] = np.array(continue_data["melspec_norm_synthesized"].apply(len),
                                                            dtype=int)
                    continue_data.sort_values(by="lens_input", inplace=True)

                inps = continue_data["cp_norm"]
                tgts = continue_data["melspec_norm_synthesized"]

                lens_input = torch.tensor(np.array(continue_data.lens_input)).to(self.device)
                lens_output = torch.tensor(np.array(continue_data.lens_output)).to(self.device)

                for e in range(n_epochs):
                    avg_loss = list()
                    for j in range(n_train_batches):
                        lens_input_j = lens_input[j * batch_size:(j * batch_size) + batch_size]
                        batch_input = inps.iloc[j * batch_size:(j * batch_size) + batch_size]
                        batch_input = pad_batch_online(lens_input_j, batch_input, self.device)

                        lens_output_j = lens_output[j * batch_size:(j * batch_size) + batch_size]
                        batch_output = tgts.iloc[j * batch_size:(j * batch_size) + batch_size]
                        batch_output = pad_batch_online(lens_output_j, batch_output, self.device)

                        Y_hat = self.pred_model(batch_input, lens_input)

                        self.pred_optimizer.zero_grad()
                        pred_loss = self.pred_criterion(Y_hat, batch_output)
                        pred_loss.backward()
                        self.pred_optimizer.step()

                        avg_loss.append(float(pred_loss.item()))
                    model_loss.append(np.mean(avg_loss))

        planned_cp = xx_new[-1, :, :].detach().cpu().numpy()
        prod_sig = sig
        prod_sr = sr
        target_mel = target_mel[-1, :, :].detach().cpu().numpy()
        prod_mel = prod_mel[-1, :, :].detach().cpu().numpy()
        with torch.no_grad():
            pred_mel = self.pred_model(xx_new)
        pred_mel = pred_mel[-1, :, :].detach().cpu().numpy()

        print("--- %.2f min ---" % ((time.time() - start_time) / 60))

        #  0. planned_cp
        #  1. initial_cp
        #  2. target_sig
        #  3. target_sr
        #  4. target_mel
        #  5. prod_sig
        #  6. prod_sr
        #  7. prod_mel
        #  8. pred_mel
        #  9. prod_loss_steps
        # 10. planned_loss_steps
        # 11. planned_mel_loss_steps
        # 12. vel_loss_steps
        # 13. jerk_loss_steps
        # 14. pred_semvec_loss_steps
        # 15. prod_semvec_loss_steps
        # 16. cp_steps
        # 17. pred_semvec_steps
        # 18. prod_semvec_steps
        # 19. grad_steps
        # 20. sig_steps
        # 21. prod_mel_steps
        # 22. pred_mel_steps
        # 23. model_loss
        # 24. pred_model
        # 25. pred_optimizer

        return PlanningResults(planned_cp, initial_cp, target_sig, target_sr,
                target_mel, prod_sig, prod_sr, prod_mel,
                pred_mel, prod_loss_steps, planned_loss_steps,
                planned_mel_loss_steps, vel_loss_steps, jerk_loss_steps,
                pred_semvec_loss_steps, prod_semvec_loss_steps, cp_steps,
                pred_semvec_steps, prod_semvec_steps, grad_steps, sig_steps,
                prod_mel_steps, pred_mel_steps, model_loss)

