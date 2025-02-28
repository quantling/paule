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

import random
import time
import os
from collections import namedtuple

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.nn import L1Loss, MSELoss
import soundfile as sf

# Set seed
torch.manual_seed(20200905)
random.seed(20200905)

tqdm.pandas()

from .util import (speak, inv_normalize_cp, normalize_mel_librosa,
        stereo_to_mono, librosa_melspec, RMSELoss, get_vel_acc_jerk,
        cp_trajectory_loss, mel_to_sig, pad_batch_online,
        speak_and_extract_tube_information, normalize_tube,
        get_area_info_within_oral_cavity, get_pretrained_weights_version, local_linear)

from .models import (ForwardModel, InverseModelMelTimeSmoothResidual,
        EmbeddingModel, Generator, NonLinearModel, SpeechNonSpeechTransformer, LinearClassifier)

from . import visualize

DIR = os.path.dirname(__file__)


PlanningResults = namedtuple('PlanningResults', "planned_cp, initial_cp, initial_sig, initial_sr, initial_prod_mel,initial_pred_mel, target_sig, target_sr, target_mel, prod_sig, prod_sr, prod_mel, pred_mel, initial_prod_semvec, initial_pred_semvec, prod_semvec, pred_semvec, prod_loss_steps, planned_loss_steps, planned_mel_loss_steps, vel_loss_steps, jerk_loss_steps, pred_semvec_loss_steps, prod_semvec_loss_steps, cp_steps, pred_semvec_steps, prod_semvec_steps, grad_steps, sig_steps, prod_mel_steps, pred_mel_steps, pred_model_loss, inv_model_loss")
PlanningResultsWithSpeechClassifier = namedtuple('PlanningResultsWithSpeechClassifier', "planned_cp, initial_cp, initial_sig, initial_sr, initial_prod_mel, initial_pred_mel, target_sig, target_sr, target_mel, prod_sig, prod_sr, prod_mel, pred_mel, initial_prod_semvec, initial_pred_semvec, prod_semvec, pred_semvec, prod_loss_steps, planned_loss_steps, planned_mel_loss_steps, vel_loss_steps, jerk_loss_steps, pred_semvec_loss_steps, prod_semvec_loss_steps, pred_speech_classifier_loss_steps, prod_speech_classifier_loss_steps, cp_steps, pred_semvec_steps, prod_semvec_steps, grad_steps, sig_steps, prod_mel_steps, pred_mel_steps, pred_model_loss, inv_model_loss")
PlanningResultsWithSomatosensory = namedtuple('PlanningResultsWithSomatosensory', "planned_cp, initial_cp, initial_sig, initial_sr, initial_prod_mel,initial_pred_mel, initial_prod_tube, initial_pred_tube, initial_prod_tube_mel, initial_pred_tube_mel, target_sig, target_sr, target_mel, prod_sig, prod_sr, prod_mel, pred_mel, prod_tube, pred_tube, prod_tube_mel, pred_tube_mel, initial_prod_semvec, initial_pred_semvec, initial_prod_tube_semvec, initial_pred_tube_semvec, prod_semvec, pred_semvec, prod_tube_semvec, pred_tube_semvec, prod_loss_steps, planned_loss_steps, planned_mel_loss_steps, vel_loss_steps, jerk_loss_steps, pred_semvec_loss_steps, prod_semvec_loss_steps, prod_tube_loss_steps, pred_tube_mel_loss_steps,prod_tube_mel_loss_steps, pred_tube_semvec_loss_steps, prod_tube_semvec_loss_steps, cp_steps, pred_semvec_steps, prod_semvec_steps, grad_steps, sig_steps, prod_mel_steps, pred_mel_steps, prod_tube_steps, pred_tube_steps, prod_tube_mel_steps, pred_tube_mel_steps, prod_tube_semvec_steps, pred_tube_semvec_steps, pred_model_loss, inv_model_loss, tube_model_loss, tube_mel_model_loss")


BestSynthesisAcoustic = namedtuple('BestSynthesisAcoustic', "mel_loss, planned_cp, prod_sig, prod_mel, pred_mel")
BestSynthesisSemantic = namedtuple('BestSynthesisSemantic', "semvec_loss, planned_cp, prod_sig, prod_semvec, pred_semvec")
BestSynthesisSomatosensory = namedtuple('BestSynthesisSomatosensory',"tube_loss, tube_mel_loss, tube_semvec_loss, planned_cp, prod_sig, prod_tube, pred_tube, prod_tube_mel, pred_tube_mel, prod_tube_semvec, pred_tube_semvec")

SubLosses = namedtuple('SubLosses', "mel_loss, semvec_loss, velocity_loss, jerk_loss, local_linear_loss, speech_classifier_loss, tube_mel_loss, tube_semvec_loss")

rmse_loss = RMSELoss(eps=0)
mse_loss = l2 = MSELoss()
l1 = L1Loss()
bce_loss = torch.nn.BCEWithLogitsLoss()



def velocity_jerk_loss(pred, *, loss=rmse_loss, guiding_factor=None):
    """returns (velocity_loss, jerk_loss) tuple"""
    #vel1, acc1, jerk1 = get_vel_acc_jerk(pred, delta_t=1.0/401)
    vel1, acc1, jerk1 = get_vel_acc_jerk(pred, delta_t=1.0)

    if guiding_factor is None:
        velocity_loss = loss(vel1, torch.zeros_like(vel1))
        jerk_loss = loss(jerk1, torch.zeros_like(jerk1))
    else:
        assert 0.0 < guiding_factor < 1.0
        velocity_loss = loss(vel1, guiding_factor * vel1.detach().clone())
        jerk_loss = loss(jerk1, guiding_factor * jerk1.detach().clone())

    return velocity_loss, jerk_loss



class Paule():
    """
    The class paule keeps track of the the state associated with the Predictive
    Articulatory speech synthesise Using Lexical Embeddings.
    This state especially are the weights of the predictive, inverse and
    embedder model as well as data used for continue learning.

    """

    def __init__(self, *, pred_model=None, pred_optimizer=None, inv_model=None, inv_optimizer=None,
                 embedder=None, cp_gen_model=None, mel_gen_model=None,
                 use_somatosensory_feedback=False, cp_tube_model=None, tube_optimizer=None,
                 tube_mel_model=None, tube_mel_optimizer=None, tube_embedder=None,
                 continue_data=None, device=torch.device('cpu'), smiling=False,
                 use_speech_classifier=False, speech_classifier=None,
                 speech_classifier_optimizer=None):

        # load the pred_model, inv_model and embedder here
        # for cpu
        self.device = device

        self.smiling = smiling

        print(f'Version of pretrained weights is "{get_pretrained_weights_version()}"')

        if use_somatosensory_feedback and use_speech_classifier:
            raise NotImplementedError("at the moment you have to choose either to use `use_somatosenrosry_feedback=True` OR to use `use_speech_classifier=True` or none")

        # PREDictive MODEL (cp -> mel)
        if pred_model:
            self.pred_model = pred_model
        else:
            self.pred_model = ForwardModel(num_lstm_layers=1, hidden_size=720).double()
            self.pred_model.load_state_dict(
                torch.load(os.path.join(DIR, "pretrained_models/predictive/pred_model_common_voice_1_720_lr_0001_50_00001_50_000001_50_0000001_200.pt"),
                           map_location=self.device,
                           weights_only=True))
            # Non-Linear Perceptron PREDictive MODEL
            #self.pred_model = NonLinearModel(input_channel=30, output_channel=60,
            #                         mode="pred",
            #                         hidden_units=30000,
            #                         on_full_sequence=True,
            #                         add_vel_and_acc=False).double()
            #self.pred_model.load_state_dict(
            #    torch.load(os.path.join(DIR,
            #                            "pretrained_models/predictive/pred_leaky_relu_non_linear_30000_hidden_lr_0001_50_00001_50_000001_50_0000001_200.pt"),
            #               map_location=self.device))

        self.pred_model = self.pred_model.to(self.device)

        # INVerse MODEL (mel -> cp)
        if inv_model:
            self.inv_model = inv_model
        else:
            self.inv_model = InverseModelMelTimeSmoothResidual(num_lstm_layers=1, hidden_size=720).double()
            self.inv_model.load_state_dict(
                torch.load(os.path.join(DIR, "pretrained_models/inverse/inv_model_common_voice_3_1_720_5_lr_0001_50_00001_50_000001_50_0000001_200.pt"),
                           map_location=self.device,
                           weights_only=True))
            # Non-Linear Perceptron INVerse MODEL
            #self.inv_model = NonLinearModel(input_channel=60, output_channel=30,
            #                                 mode="inv",
            #                                 hidden_units=30000,
            #                                 on_full_sequence=True,
            #                                 add_vel_and_acc=False).double()
            #self.inv_model.load_state_dict(
            #    torch.load(os.path.join(DIR,
            #                            "pretrained_models/inverse/inv_leaky_relu_non_linear_30000_hidden_lr_0001_50_00001_50_000001_50_0000001_200.pt"),
            #               map_location=self.device))
        self.inv_model = self.inv_model.to(self.device)

        # EMBEDDER (mel -> semvec)
        if embedder:
            self.embedder = embedder
        else:
            self.embedder = EmbeddingModel(num_lstm_layers=2, hidden_size=720).double()
            self.embedder.load_state_dict(torch.load(
                os.path.join(DIR, "pretrained_models/embedder/embed_model_common_voice_syn_rec_2_720_0_dropout_07_noise_6e05_rmse_lr_00001_200.pt"),
                map_location=self.device,
                weights_only=True))
            self.embedder.eval()
            # Non-Linear Perceptron Embedder
            #self.embedder = NonLinearModel(input_channel=60, output_channel=300,
            #                                mode="embed",
            #                                hidden_units=30000,
            #                                on_full_sequence=True,
            #                                add_vel_and_acc=False).double()
            #self.embedder.load_state_dict(
            #    torch.load(os.path.join(DIR,
            #                            "pretrained_models/embedder/model_embed_sum_leaky_relu_non_linear_30000_hidden_noise_6e05_lr_00001_200.pt"),
            #               map_location=self.device))

        self.embedder = self.embedder.to(self.device)

        # CP GENerative MODEL
        if cp_gen_model:
            self.cp_gen_model = cp_gen_model
        else:
            self.cp_gen_model = Generator().double()
            self.cp_gen_model.load_state_dict(torch.load(
                os.path.join(DIR, "pretrained_models/cp_gan/conditional_trained_cp_generator_whole_critic_it_5_10_20_40_80_100_415.pt"),
                map_location=self.device,
                weights_only=True))
        self.cp_gen_model = self.cp_gen_model.to(self.device)
        self.cp_gen_model.eval()

        # MEL GENerative MODEL
        if mel_gen_model:
            self.mel_gen_model = mel_gen_model
        else:
            self.mel_gen_model = Generator(output_size=60).double()
            self.mel_gen_model.load_state_dict(torch.load(
                os.path.join(DIR, "pretrained_models/mel_gan/conditional_trained_mel_generator_synthesized_critic_it_5_10_20_40_80_100_400.pt"),
                                                          map_location=self.device,
                           weights_only=True))
        self.mel_gen_model = self.mel_gen_model.to(self.device)
        self.mel_gen_model.eval()

        self.use_speech_classifier = use_speech_classifier
        if self.use_speech_classifier:
            # SPEECH CLASSIFIER (binary classifier with 0 is speech-like)
            if speech_classifier:
                self.speech_classifier = speech_classifier
            else:
                self.speech_classifier = LinearClassifier(input_dim=60, output_dim=1)
                #self.speech_classifier = SpeechNonSpeechTransformer(input_dim=60, num_layers=3, nhead=6, output_dim=1)
                self.speech_classifier.load_state_dict(torch.load(
                    os.path.join(DIR, "pretrained_models/speech_classifier/linear_model_rec_as_nonspeech.pt"),
                    #os.path.join(DIR, "pretrained_models/speech_classifier/model_rec_as_nonspeech.pt"),
                           map_location=self.device,
                           weights_only=True))
                self.speech_classifier = self.speech_classifier.double()
            self.speech_classifier = self.speech_classifier.to(self.device)
            self.speech_classifier.eval()

        self.use_somatosensory_feedback = use_somatosensory_feedback
        if self.use_somatosensory_feedback:
            # CP-Tube Model (cp -> tube)
            if cp_tube_model:
                self.cp_tube_model = cp_tube_model
            else:
                self.cp_tube_model = ForwardModel(num_lstm_layers=1,
                                             hidden_size=360,
                                             output_size=10,
                                             input_size=30,
                                             apply_half_sequence=False).double()
                self.cp_tube_model.load_state_dict(torch.load(
                    os.path.join(DIR, "pretrained_models/somatosensory/cp_to_tube_model_1_360_lr_0001_50_00001_100.pt"),
                                                         map_location=self.device,
                           weights_only=True))
            self.cp_tube_model = self.cp_tube_model.to(self.device)

            # Tube-Mel Model (tube -> mel)
            if tube_mel_model:
                self.tube_mel_model = tube_mel_model
            else:
                self.tube_mel_model = ForwardModel(num_lstm_layers=1,
                     hidden_size = 360,
                     output_size = 60,
                    input_size = 10,
                    apply_half_sequence=True).double()
                self.tube_mel_model.load_state_dict(torch.load(
                    os.path.join(DIR, "pretrained_models/somatosensory/tube_to_mel_model_1_360_lr_0001_50_00001_100.pt"),
                                                          map_location=self.device,
                           weights_only=True))
            self.tube_mel_model = self.tube_mel_model.to(self.device)

            # Tube-Embedder Model (tube -> semvec)
            if tube_embedder:
                self.tube_embedder = tube_embedder
            else:
                self.tube_embedder = EmbeddingModel(input_size = 10,
                                                    num_lstm_layers=2,
                                                    hidden_size=720,
                                                    dropout=0.7,
                                                    post_upsampling_size=0).double()
                self.tube_embedder.load_state_dict(torch.load(
                    os.path.join(DIR, "pretrained_models/somatosensory/tube_to_vector_model_2_720_0_dropout_07_noise_6e05_rmse_lr_00001_200.pt"),
                           map_location=self.device,
                           weights_only=True))
            self.tube_embedder = self.tube_embedder.to(self.device)
            self.tube_embedder.eval()

        # DATA to continue learning
        self.continue_data = continue_data
        self.continue_data_limit = 1000  # max amount of training data stored in paule instance

        if self.continue_data is not None:
            if len(self.continue_data) > self.continue_data_limit:
                random_sample = random.sample(range(len(self.continue_data)), self.continue_data_limit)
                self.continue_data = self.continue_data.iloc[random_sample].reset_index(drop=True)

        if pred_optimizer:
            self.pred_optimizer = pred_optimizer
        else:
            self.pred_optimizer = torch.optim.Adam(self.pred_model.parameters(), lr=0.001)
        self.pred_criterion = rmse_loss

        if inv_optimizer:
            self.inv_optimizer = inv_optimizer
        else:
            self.inv_optimizer = torch.optim.Adam(self.inv_model.parameters(), lr=0.001)
        self.inv_criterion = cp_trajectory_loss

        if self.use_somatosensory_feedback:
            if tube_optimizer:
                self.tube_optimizer = tube_optimizer
            else:
                self.tube_optimizer = torch.optim.Adam(self.cp_tube_model.parameters(), lr=0.001)
            self.tube_criterion = rmse_loss
            if tube_mel_optimizer:
                self.tube_mel_optimizer = tube_optimizer
            else:
                self.tube_mel_optimizer = torch.optim.Adam(self.tube_mel_model.parameters(), lr=0.001)
            self.tube_mel_criterion = rmse_loss

        if self.use_speech_classifier:
            if speech_classifier_optimizer:
                self.speech_classifier_optimizer = speech_classifier_optimizer
            else:
                self.speech_classifier_optimizer = torch.optim.Adam(self.speech_classifier.parameters(), lr=0.001)
            self.speech_classifier_criterion = torch.nn.BCEWithLogitsLoss()

        self.best_synthesis_acoustic = None
        self.best_synthesis_semantic = None
        if self.use_somatosensory_feedback:
            self.best_synthesis_somatosensory = None

    def create_epoch_batches(self, df_length, batch_size, shuffle=True, same_size_batching=False, sorted_training_length_keys=None, training_length_dict=None):
        """
        Create Epoch by batching indices of dataset

        Parameters
        ==========
        df_length : int
            total number of samples in training set
        batch_size : int
            number of samples in one batch
        shuffle : bool
            keep order of training set or random shuffle
        same_size_batching : bool
            create epoch of batches with similar long samples to avoid long padding
        sorted_training_length_keys : np.array
            sorted array of unique sequence lengths in training data
        training_length_dict: dict
            dictionary keys containing indices of samples with unique sequence lengths in training data and values
            containing indices of samples with corresponding length

        Returns
        =======
        epoch : list of list
            list of lists containing indices for each batch in one epoch
        """

        if same_size_batching and training_length_dict is None:
            raise ValueError("Dictionary containing indices of samples with corresponding length needed for same_size_batching!")

        if same_size_batching:
            epoch = []  # list of batches
            foundlings = []  # rest samples for each length which do not fit into one batch
            sorted_training_length_keys = np.sort(list(training_length_dict.keys())) # sorted unique lengths in training
            for length in sorted_training_length_keys:  # iterate over each unique length in training data
                length_idxs = training_length_dict[length]  # dictionary containing indices of samples with length
                rest = len(length_idxs) % batch_size
                random.shuffle(length_idxs)  # shuffle indices
                epoch += [length_idxs[i * batch_size:(i * batch_size) + batch_size] for i in
                          range(int(len(length_idxs) / batch_size))]  # cut into batches and append to epoch
                if rest > 0:
                    foundlings += list(length_idxs[
                                       -rest:])  # remaining indices which do not fit into one batch are stored in foundling
            foundlings = np.asarray(foundlings)
            rest = len(foundlings) % batch_size
            epoch += [foundlings[i * batch_size:(i * batch_size) + batch_size] for i in
                      range(int(len(
                          foundlings) / batch_size))]  # cut foudnlings into batches (because inserted sorted this ensures minimal padding)
            if rest > 0:
                epoch += [foundlings[-rest:]]  # put rest into one batch (allow smaller batch)
            random.shuffle(epoch)

        else:
            rest = df_length % batch_size
            idxs = list(range(df_length))
            if shuffle:
                random.shuffle(idxs)  # shuffle indicees
            if rest > 0:
                idxs += idxs[:(
                            batch_size - rest)]  # rolling batching (if number samples not divisible by batch_size append first again)
            epoch = [idxs[i * batch_size:(i * batch_size) + batch_size] for i in
                     range(int(len(idxs) / batch_size))]  # cut into batches
        return epoch

    def plan_iterative(self, *,
                       target_acoustic=None,
                       target_semvecs=None,
                       target_seq_lengths=None,
                       overlap=8, **kwargs):
        pass


    def plan_resynth(self, *, learning_rate_planning=0.01, learning_rate_learning=0.001,
                     learning_rate_learning_inv=None,
                     target_acoustic=None,
                     target_semvec=None,
                     target_seq_length=None,
                     initial_cp=None,
                     past_cp=None,
                     initialize_from="acoustic",
                     objective="acoustic",
                     n_outer=5, n_inner=24,
                     continue_learning=True,
                     continue_learning_inv=False,
                     continue_learning_tube=False,
                     add_training_data_pred=False,
                     add_training_data_inv=False,
                     n_batches=3, batch_size=8, n_epochs=10,
                     log_ii=1,
                     log_semantics=True,
                     log_gradients=False,
                     log_signals = False,
                     log_cps = False,
                     plot=False,
                     seed=None,
                     verbose=True):
        """
        plans resynthesis cp trajectories.

        Parameters
        ==========
        learning_rate_planning : float
            learning rate for updating cps
        learning_rate_learning : float
            learning rate for updating predictive model
        learning_rate_learning_inv: float
            learning rate for updating inverse model
        target_acoustic : str
            (target_sig, target_sr)
        target_semvec : torch.tensor
        target_seq_length : int (None)
        initial_cp : torch.tensor
        past_cp : torch.tensor
            cp that are already executed, but should be conditioned on they
            will be concatenated to the beginning of the initial_cp
        initialize_from : {'semvec', 'acoustic', None}
            can be None, if initial_cp are given
        objective : {'acoustic_semvec', 'acoustic', 'semvec'}
        n_outer : int (40)
        n_inner : int (100)
        continue_learning : bool (True)
            update predictive model with synthesized acoustics
        continue_learning_inv : bool (False)
            update inverse model
        continue_learning_tube : bool (False),
             update tube model with tube information after synthesizing acoustics
        add_training_data_pred : bool (False)
            update predictive model solely on produced acoustics during training or add training data
        add_training_data_inv : bool (False)
            update inverse model solely on produced acoustics during training or add training data
        n_batches : int
            number of batches to train on
        batch_size : int
            number of samples in one batch
        n_epochs : int
            number of epochs to train on with number of batches
        log_ii : int
            log results and synthesize audio after ii number of inner iterations
        plot : bool  or str (False)
            if False no plotting; if a string is given this is used as the path
            where to store the plots; if True interactive blocking plotting is enabled
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
                param_group['lr'] = learning_rate_learning # set learning rate

        if learning_rate_learning_inv:
            for param_group in self.inv_optimizer.param_groups:
                param_group['lr'] = learning_rate_learning_inv # set learning rate

        if log_ii is None:
            log_ii = n_inner

        if log_ii > n_inner:
            raise ValueError('results can only be logged between first and last planning step')

        if isinstance(target_acoustic, str):
            target_sig, target_sr = sf.read(target_acoustic)
            if len(target_sig.shape) == 2:
                target_sig = stereo_to_mono(target_sig)
            # assert target_sr == 44100, 'sampling rate of wave name must be 44100'

        elif target_acoustic is None:
            pass
        elif len(target_acoustic) == 2:
            target_sig, target_sr = target_acoustic
        else:
            if len(target_acoustic.shape) == 2:
                target_mel = target_acoustic.copy()
                target_mel.shape = (1,) + target_mel.shape
                target_mel = torch.from_numpy(target_mel)
            else:
                if not isinstance(target_acoustic, torch.Tensor):
                    target_mel = torch.from_numpy(target_acoustic)
                else:
                    target_mel = target_acoustic.copy()
            if not isinstance(target_mel, torch.Tensor):
                raise ValueError("target_acoustic has to be torch.Tensor at this point")
            target_seq_length = target_mel.shape[1]
            target_sig = None
            target_sr = None


        if target_acoustic is None and (target_seq_length is None or target_semvec is None):
            raise ValueError("if target_acoustic is None you need to give a target_seq_length and a target_semvec")
        elif target_acoustic is None:
            mel_gen_noise = torch.randn(1, 1, 100).to(self.device)
            if not isinstance(target_semvec, torch.Tensor):
                target_semvec = torch.tensor(target_semvec, device=self.device)
            mel_gen_semvec = target_semvec.view(1, 300).detach().clone()
            target_mel = self.mel_gen_model(mel_gen_noise, target_seq_length, mel_gen_semvec)
            target_mel = target_mel.detach().clone()
            target_sig, target_sr = mel_to_sig(target_mel.view(target_mel.shape[1], target_mel.shape[2]).cpu().numpy())
        elif len(target_acoustic) == 2 or isinstance(target_acoustic, str):
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
            if not isinstance(target_semvec, torch.Tensor):
                target_semvec = torch.tensor(target_semvec, device=self.device)
            target_semvec = target_semvec.view(1, 300).detach().clone()
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
                    target_semvec = torch.tensor(target_semvec, device=self.device)
                cp_gen_semvec = target_semvec.view(1, 300).detach().clone()
                initial_cp = self.cp_gen_model(cp_gen_noise, 2 * target_seq_length, cp_gen_semvec)
                initial_cp = initial_cp.detach().cpu().numpy()
                initial_cp.shape = (initial_cp.shape[1], initial_cp.shape[2])
            else:
                raise ValueError("initialize_from has to be either 'acoustic' or 'semvec'")

        else:
            if initialize_from is not None:
                raise ValueError('one of initial_cp and initialize_from has to be None')
            if not initial_cp.shape[0] == (target_mel.shape[1] * 2):
                raise ValueError(f"initial_cp {initial_cp.shape[0]}, target_mel {target_mel.shape[1] * 2}")

        if not past_cp is None and past_cp.shape[0] % 2 != 0:
            raise ValueError("past_cp have to be None or the sequence length has to be an even number")
        # 1.3 create initial xx
        # initial_cp = np.zeros_like(initial_cp)
        if not past_cp is None:
            initial_cp = np.concatenate((past_cp, initial_cp), axis=0)
            # store past_cp as torch tensor to reset after each planning iteration
            past_cp_torch = torch.from_numpy(past_cp)
            past_cp_torch = past_cp_torch.to(self.device)

        xx_new = initial_cp.copy()
        xx_new.shape = (1, xx_new.shape[0], xx_new.shape[1])
        xx_new = torch.from_numpy(xx_new)
        xx_new = xx_new.to(self.device)
        xx_new.requires_grad_()
        xx_new.retain_grad()

        MEL_WEIGHT = 5.0
        VELOCITY_WEIGHT = 80.0  # alternative: 1000
        JERK_WEIGHT = 400.0  # alternative: 4000
        SEMANTIC_WEIGHT = 10.0
        SPEECH_CLASSIFIER_WEIGHT = 0.1
        LOCAL_LINEAR_WEIGHT = 100_000
        TUBE_MEL_WEIGHT = MEL_WEIGHT
        TUBE_SEMANTIC_WEIGHT = SEMANTIC_WEIGHT


        if objective == 'acoustic_semvec':
            if self.use_speech_classifier:
                def criterion(pred_mel, target_mel, pred_semvec, target_semvec, cps, pred_speech_classifier):
                    mel_loss = rmse_loss(pred_mel, target_mel)
                    semvec_loss = rmse_loss(pred_semvec, target_semvec)
                    velocity_loss, jerk_loss = velocity_jerk_loss(cps, loss=mse_loss)
                    ll = local_linear(cps)
                    local_linear_loss = mse_loss(ll, torch.zeros_like(ll, dtype=ll.dtype))
                    speech_classifier_loss = bce_loss(pred_speech_classifier,
                            torch.zeros_like(pred_speech_classifier,
                                dtype=pred_speech_classifier.dtype))

                    mel_loss = MEL_WEIGHT * mel_loss
                    semvec_loss = SEMANTIC_WEIGHT * semvec_loss
                    velocity_loss = VELOCITY_WEIGHT * velocity_loss
                    jerk_loss = JERK_WEIGHT * jerk_loss
                    local_linear_loss = LOCAL_LINEAR_WEIGHT * local_linear_loss
                    speech_classifier_loss = SPEECH_CLASSIFIER_WEIGHT * speech_classifier_loss

                    loss = mel_loss + velocity_loss + jerk_loss + semvec_loss + speech_classifier_loss + local_linear_loss
                    return loss, SubLosses(mel_loss, semvec_loss, velocity_loss, jerk_loss, local_linear_loss, speech_classifier_loss, None, None)

            elif self.use_somatosensory_feedback:
                def criterion(pred_mel, target_mel, pred_semvec, target_semvec, cps, pred_tube_mel, pred_tube_semvec):
                    mel_loss = rmse_loss(pred_mel, target_mel)
                    semvec_loss = rmse_loss(pred_semvec, target_semvec)
                    velocity_loss, jerk_loss = velocity_jerk_loss(cps, loss=mse_loss)
                    ll = local_linear(cps)
                    local_linear_loss = mse_loss(ll, torch.zeros_like(ll, dtype=ll.dtype))
                    tube_mel_loss = rmse_loss(pred_tube_mel, target_mel)
                    tube_semvec_loss = rmse_loss(pred_tube_semvec, target_semvec)

                    mel_loss = MEL_WEIGHT * mel_loss
                    semvec_loss = SEMANTIC_WEIGHT * semvec_loss
                    velocity_loss = VELOCITY_WEIGHT * velocity_loss
                    jerk_loss = JERK_WEIGHT * jerk_loss
                    local_linear_loss = LOCAL_LINEAR_WEIGHT * local_linear_loss
                    tube_mel_loss = TUBE_MEL_WEIGHT * tube_mel_loss
                    tube_semvec_loss = TUBE_SEMANTIC_WEIGHT * tube_semvec_loss

                    loss = mel_loss + velocity_loss + jerk_loss + semvec_loss + local_linear_loss + tube_mel_loss + tube_semvec_loss

                    return loss, SubLosses(mel_loss, semvec_loss, velocity_loss, jerk_loss, local_linear_loss, None, tube_mel_loss, tube_semvec_loss)

            else:
                def criterion(pred_mel, target_mel, pred_semvec, target_semvec, cps):
                    mel_loss = rmse_loss(pred_mel, target_mel)
                    semvec_loss = rmse_loss(pred_semvec, target_semvec)
                    velocity_loss, jerk_loss = velocity_jerk_loss(cps, loss=mse_loss)
                    ll = local_linear(cps)
                    local_linear_loss = mse_loss(ll, torch.zeros_like(ll, dtype=ll.dtype))

                    mel_loss = MEL_WEIGHT * mel_loss
                    semvec_loss = SEMANTIC_WEIGHT * semvec_loss
                    velocity_loss = VELOCITY_WEIGHT * velocity_loss
                    jerk_loss = JERK_WEIGHT * jerk_loss
                    local_linear_loss = LOCAL_LINEAR_WEIGHT * local_linear_loss

                    loss = mel_loss + velocity_loss + jerk_loss + semvec_loss + local_linear_loss

                    return loss, SubLosses(mel_loss, semvec_loss, velocity_loss, jerk_loss, local_linear_loss, None, None, None)

        elif objective == 'acoustic':
            if self.use_speech_classifier:
                def criterion(pred_mel, target_mel, cps, pred_speech_classifier):
                    mel_loss = rmse_loss(pred_mel, target_mel)
                    velocity_loss, jerk_loss = velocity_jerk_loss(cps, loss=mse_loss)
                    ll = local_linear(cps)
                    local_linear_loss = mse_loss(ll, torch.zeros_like(ll, dtype=ll.dtype))
                    speech_classifier_loss = bce_loss(pred_speech_classifier,
                            torch.zeros_like(pred_speech_classifier,
                                dtype=pred_speech_classifier.dtype))

                    mel_loss = MEL_WEIGHT * mel_loss
                    velocity_loss = VELOCITY_WEIGHT * velocity_loss
                    jerk_loss = JERK_WEIGHT * jerk_loss
                    local_linear_loss = LOCAL_LINEAR_WEIGHT * local_linear_loss
                    speech_classifier_loss = SPEECH_CLASSIFIER_WEIGHT * speech_classifier_loss

                    loss = mel_loss + velocity_loss + jerk_loss + local_linear_loss + speech_classifier_loss

                    return loss, SubLosses(mel_loss, None, velocity_loss, jerk_loss, local_linear_loss, speech_classifier_loss, None, None)

            elif self.use_somatosensory_feedback:
                def criterion(pred_mel, target_mel, cps, pred_tube_mel):
                    mel_loss = rmse_loss(pred_mel, target_mel)
                    velocity_loss, jerk_loss = velocity_jerk_loss(cps, loss=mse_loss)
                    ll = local_linear(cps)
                    local_linear_loss = mse_loss(ll, torch.zeros_like(ll, dtype=ll.dtype))
                    tube_mel_loss = rmse_loss(pred_tube_mel, target_mel)
                    tube_semvec_loss = rmse_loss(pred_tube_semvec, target_semvec)

                    mel_loss = MEL_WEIGHT * mel_loss
                    velocity_loss = VELOCITY_WEIGHT * velocity_loss
                    jerk_loss = JERK_WEIGHT * jerk_loss
                    local_linear_loss = LOCAL_LINEAR_WEIGHT * local_linear_loss
                    tube_mel_loss = TUBE_MEL_WEIGHT * tube_mel_loss
                    tube_semvec_loss = TUBE_SEMANTIC_WEIGHT * tube_semvec_loss

                    loss = mel_loss + velocity_loss + jerk_loss + local_linear_loss + tube_mel_loss + tube_semvec_loss
                    return loss, SubLosses(mel_loss, None, velocity_loss, jerk_loss, local_linear_loss, None, tube_mel_loss, tube_semvec_loss)

            else:
                def criterion(pred_mel, target_mel, cps):
                    mel_loss = rmse_loss(pred_mel, target_mel)
                    velocity_loss, jerk_loss = velocity_jerk_loss(cps, loss=mse_loss)
                    ll = local_linear(cps)
                    local_linear_loss = mse_loss(ll, torch.zeros_like(ll, dtype=ll.dtype))

                    mel_loss = MEL_WEIGHT * mel_loss
                    velocity_loss = VELOCITY_WEIGHT * velocity_loss
                    jerk_loss = JERK_WEIGHT * jerk_loss
                    local_linear_loss = LOCAL_LINEAR_WEIGHT * local_linear_loss

                    loss = mel_loss + velocity_loss + jerk_loss + local_linear_loss
                    return loss, SubLosses(mel_loss, None, velocity_loss, jerk_loss, local_linear_loss, None, None, None)

        elif objective == 'semvec':
            if self.use_speech_classifier:
                def criterion(pred_semvec, target_semvec, cps, pred_speech_classifier):
                    semvec_loss = rmse_loss(pred_semvec, target_semvec)
                    velocity_loss, jerk_loss = velocity_jerk_loss(cps, loss=mse_loss)
                    ll = local_linear(cps)
                    local_linear_loss = mse_loss(ll, torch.zeros_like(ll, dtype=ll.dtype))
                    speech_classifier_loss = bce_loss(pred_speech_classifier,
                            torch.zeros_like(pred_speech_classifier,
                                dtype=pred_speech_classifier.dtype))

                    semvec_loss = SEMANTIC_WEIGHT * semvec_loss
                    velocity_loss = VELOCITY_WEIGHT * velocity_loss
                    jerk_loss = JERK_WEIGHT * jerk_loss
                    local_linear_loss = LOCAL_LINEAR_WEIGHT * local_linear_loss
                    speech_classifier_loss = SPEECH_CLASSIFIER_WEIGHT * speech_classifier_loss

                    loss = velocity_loss + jerk_loss + semvec_loss + speech_classifier_loss + local_linear_loss
                    return loss, SubLosses(None, semvec_loss, velocity_loss, jerk_loss, local_linear_loss, speech_classifier_loss, None, None)

            elif self.use_somatosensory_feedback:
                def criterion(pred_semvec, target_semvec, cps, pred_tube_semvec):
                    semvec_loss = rmse_loss(pred_semvec, target_semvec)
                    velocity_loss, jerk_loss = velocity_jerk_loss(cps, loss=mse_loss)
                    ll = local_linear(cps)
                    local_linear_loss = mse_loss(ll, torch.zeros_like(ll, dtype=ll.dtype))
                    tube_mel_loss = rmse_loss(pred_tube_mel, target_mel)
                    tube_semvec_loss = rmse_loss(pred_tube_semvec, target_semvec)

                    semvec_loss = SEMANTIC_WEIGHT * semvec_loss
                    velocity_loss = VELOCITY_WEIGHT * velocity_loss
                    jerk_loss = JERK_WEIGHT * jerk_loss
                    local_linear_loss = LOCAL_LINEAR_WEIGHT * local_linear_loss
                    tube_mel_loss = TUBE_MEL_WEIGHT * tube_mel_loss
                    tube_semvec_loss = TUBE_SEMANTIC_WEIGHT * tube_semvec_loss

                    loss = velocity_loss + jerk_loss + semvec_loss + local_linear_loss + tube_mel_loss + tube_semvec_loss

                    return loss, SubLosses(None, semvec_loss, velocity_loss, jerk_loss, local_linear_loss, None, tube_mel_loss, tube_semvec_loss)

            else:
                def criterion(pred_semvec, target_semvec, cps):
                    semvec_loss = rmse_loss(pred_semvec, target_semvec)
                    velocity_loss, jerk_loss = velocity_jerk_loss(cps, loss=mse_loss)
                    ll = local_linear(cps)
                    local_linear_loss = mse_loss(ll, torch.zeros_like(ll, dtype=ll.dtype))

                    semvec_loss = SEMANTIC_WEIGHT * semvec_loss
                    velocity_loss = VELOCITY_WEIGHT * velocity_loss
                    jerk_loss = JERK_WEIGHT * jerk_loss
                    local_linear_loss = LOCAL_LINEAR_WEIGHT * local_linear_loss

                    loss = velocity_loss + jerk_loss + semvec_loss + local_linear_loss

                    return loss, SubLosses(None, semvec_loss, velocity_loss, jerk_loss, local_linear_loss, None, None, None)

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
        pred_model_loss = list()
        inv_model_loss = list()

        optimizer = torch.optim.Adam([xx_new], lr=learning_rate_planning)

        if self.use_somatosensory_feedback:
            prod_tube_loss_steps = list()
            pred_tube_mel_loss_steps = list()
            prod_tube_mel_loss_steps = list()  # predicted from tube info after calling speak
            pred_tube_semvec_loss_steps = list()
            prod_tube_semvec_loss_steps = list()  # predicted from tube info after calling speak

            pred_tube_steps = list()
            prod_tube_steps = list()
            prod_tube_mel_steps = list()
            pred_tube_mel_steps = list()
            pred_tube_semvec_steps = list()
            prod_tube_semvec_steps = list()

            tube_model_loss = list()
            tube_mel_model_loss = list()
        elif self.use_speech_classifier:
            prod_speech_classifier_loss_steps = list()
            pred_speech_classifier_loss_steps = list()



        # initial results
        with torch.no_grad():
            initial_pred_mel = self.pred_model(xx_new)
            initial_pred_semvec = self.embedder(initial_pred_mel, (torch.tensor(initial_pred_mel.shape[1]),))

        xx_new_numpy = xx_new[-1, :, :].detach().cpu().numpy().copy()

        if self.use_somatosensory_feedback:
            with torch.no_grad():
                initial_pred_tube = self.cp_tube_model(xx_new)
                initial_pred_tube_mel = self.tube_mel_model(initial_pred_tube)
                initial_pred_tube_semvec = self.tube_embedder(initial_pred_tube,(torch.tensor(initial_pred_tube.shape[1]),))

            initial_sig, initial_sr, initial_tube_info = speak_and_extract_tube_information(inv_normalize_cp(xx_new_numpy))

            area_within_oral_cavity = get_area_info_within_oral_cavity(initial_tube_info["tube_length_cm"], initial_tube_info["tube_area_cm2"])
            initial_prod_tube = np.concatenate([area_within_oral_cavity,
                                                np.expand_dims(initial_tube_info["incisor_pos_cm"],axis=1),
                                                np.expand_dims(initial_tube_info["tongue_tip_side_elevation"],axis=1),
                                                np.expand_dims(initial_tube_info["velum_opening_cm2"],axis=1)],axis=1)
            initial_prod_tube = normalize_tube(initial_prod_tube)

            initial_prod_tube.shape = initial_pred_tube.shape
            initial_prod_tube = torch.from_numpy(initial_prod_tube)
            initial_prod_tube = initial_prod_tube.to(self.device)

            with torch.no_grad():
                initial_prod_tube_mel = self.tube_mel_model(initial_prod_tube)
                initial_prod_tube_semvec = self.tube_embedder(initial_prod_tube, (torch.tensor(initial_prod_tube.shape[1]),))

            initial_prod_tube = initial_prod_tube[-1, :, :].detach().cpu().numpy().copy()
            initial_pred_tube = initial_pred_tube[-1, :, :].detach().cpu().numpy().copy()
            initial_prod_tube_mel = initial_prod_tube_mel[-1, :, :].detach().cpu().numpy().copy()
            initial_pred_tube_mel = initial_pred_tube_mel[-1, :, :].detach().cpu().numpy().copy()
            initial_prod_tube_semvec = initial_prod_tube_semvec[-1, :].detach().cpu().numpy().copy()
            initial_pred_tube_semvec = initial_pred_tube_semvec[-1, :].detach().cpu().numpy().copy()

        else:
            initial_sig, initial_sr = speak(inv_normalize_cp(xx_new_numpy))

        initial_prod_mel = librosa_melspec(initial_sig, initial_sr)
        initial_prod_mel = normalize_mel_librosa(initial_prod_mel)
        initial_prod_mel.shape = initial_pred_mel.shape
        initial_prod_mel = torch.from_numpy(initial_prod_mel)
        initial_prod_mel = initial_prod_mel.to(self.device)

        # if past_cp prepend the first past_cp / 2 steps to target_mel
        if not past_cp is None:
            target_mel = torch.cat((initial_prod_mel[:, 0:(past_cp.shape[0] // 2), :], target_mel), dim=1)
            target_mel = target_mel.to(self.device)

        with torch.no_grad():
            initial_prod_semvec = self.embedder(initial_prod_mel, (torch.tensor(initial_prod_mel.shape[1]),))

        initial_prod_mel = initial_prod_mel[-1, :, :].detach().cpu().numpy().copy()
        initial_pred_mel = initial_pred_mel[-1, :, :].detach().cpu().numpy().copy()
        initial_prod_semvec = initial_prod_semvec[-1, :].detach().cpu().numpy().copy()
        initial_pred_semvec = initial_pred_semvec[-1, :].detach().cpu().numpy().copy()


        self.best_synthesis_acoustic = BestSynthesisAcoustic(np.inf, initial_cp, initial_sig, initial_prod_mel, initial_pred_mel)
        self.best_synthesis_semantic = BestSynthesisSemantic(np.inf, initial_cp, initial_sig, initial_prod_semvec, initial_pred_semvec)
        if self.use_somatosensory_feedback:
            self.best_synthesis_somatosensory = BestSynthesisSomatosensory(np.inf, np.inf, np.inf, initial_cp, initial_sig,
                                                                           initial_prod_tube,
                                                                           initial_pred_tube,
                                                                           initial_prod_tube_mel,
                                                                           initial_pred_tube_mel,
                                                                           initial_prod_tube_semvec,
                                                                           initial_pred_tube_semvec)

        # continue learning
        start_time = time.time()
        for ii_outer in tqdm(range(n_outer)):
            # imagine and plan
            pred_mel_steps_ii = list()
            prod_mel_steps_ii = list()
            cp_steps_ii = list()
            pred_semvec_steps_ii = list()
            prod_semvec_steps_ii = list()

            if self.use_somatosensory_feedback:
                pred_tube_steps_ii = list()
                prod_tube_steps_ii = list()
                pred_tube_mel_steps_ii = list()
                prod_tube_mel_steps_ii = list()
                pred_tube_semvec_steps_ii = list()
                prod_tube_semvec_steps_ii = list()

            for ii in range(n_inner):
                optimizer.zero_grad()

                pred_mel = self.pred_model(xx_new)
                if self.use_speech_classifier:
                    pred_speech_classifier = self.speech_classifier(pred_mel)
                if self.use_somatosensory_feedback:
                    pred_tube = self.cp_tube_model(xx_new)
                    pred_tube.retain_grad()
                    pred_tube_mel = self.tube_mel_model(pred_tube)

                if objective in ('semvec', 'acoustic_semvec'):
                    seq_length = pred_mel.shape[1]
                    self.embedder = self.embedder.train()
                    pred_semvec = self.embedder(pred_mel, (torch.tensor(seq_length),))
                    pred_semvec_steps_ii.append(pred_semvec[-1, :].detach().cpu().numpy().copy())

                    if self.use_somatosensory_feedback:
                        tube_seq_length = pred_tube.shape[1]
                        self.tube_embedder = self.tube_embedder.train()
                        pred_tube_semvec = self.tube_embedder(pred_tube, (torch.tensor(tube_seq_length),))
                        pred_tube_semvec_steps_ii.append(pred_tube_semvec[-1, :].detach().cpu().numpy().copy())

                if objective == 'acoustic':
                    if self.use_speech_classifier:
                        discrepancy, sub_loss = criterion(pred_mel, target_mel, xx_new, pred_speech_classifier)
                    elif self.use_somatosensory_feedback:
                        discrepancy, sub_loss = criterion(pred_mel, target_mel, xx_new, pred_tube_mel)
                    else:
                        discrepancy, sub_loss = criterion(pred_mel, target_mel, xx_new)

                    if (ii + 1) % log_ii == 0:
                        planned_loss_steps.append(float(discrepancy.item()))
                        planned_mel_loss_steps.append(float(sub_loss.mel_loss.item()))
                        vel_loss_steps.append(float(sub_loss.velocity_loss.item()))
                        jerk_loss_steps.append(float(sub_loss.jerk_loss.item()))
                        if self.use_speech_classifier:
                            pred_speech_classifier_loss_steps.append(float(sub_loss.speech_classifier_loss.item()))
                        if self.use_somatosensory_feedback:
                            pred_tube_mel_loss_steps.append(float(sub_loss.tube_mel_loss.item()))

                        if log_semantics:
                            seq_length = pred_mel.shape[1]
                            pred_semvec = self.embedder(pred_mel, (torch.tensor(seq_length),))
                            pred_semvec_steps_ii.append(pred_semvec[-1, :].detach().cpu().numpy().copy())
                            semvec_loss = SEMANTIC_WEIGHT * float(rmse_loss(pred_semvec, target_semvec).item())
                            pred_semvec_loss_steps.append(semvec_loss)
                            if self.use_somatosensory_feedback:
                                tube_seq_length = pred_tube.shape[1]
                                pred_tube_semvec = self.tube_embedder(pred_tube, (torch.tensor(tube_seq_length),))
                                pred_tube_semvec_steps_ii.append(pred_tube_semvec[-1, :].detach().cpu().numpy().copy())
                                tube_semvec_loss = TUBE_SEMANTIC_WEIGHT * float(rmse_loss(pred_tube_semvec, target_semvec).item())
                                pred_tube_semvec_loss_steps.append(tube_semvec_loss)

                    if verbose:
                        print("Iteration %d" % ii)
                        print("Planned Loss: ", float(discrepancy.item()))
                        print("Mel Loss: ", float(sub_loss.mel_loss.item()))
                        print("Vel Loss: ", float(sub_loss.velocity_loss.item()))
                        print("Jerk Loss: ", float(sub_loss.jerk_loss.item()))
                        print("Local Linear Loss: ", float(sub_loss.local_linear_loss.item()))
                        if self.use_speech_classifier:
                            print("Speech Classifier Loss: ", float(sub_loss.speech_classifier_loss.item()))
                        if self.use_somatosensory_feedback:
                            print("Tube Mel Loss: ", float(sub_loss.tube_mel_loss.item()))
                        if log_semantics:
                            print("Semvec Loss: ", float(semvec_loss))
                            if self.use_somatosensory_feedback:
                                print("Tube Semvec Loss: ", float(tube_semvec_loss))

                elif objective == 'acoustic_semvec':
                    if self.use_speech_classifier:
                        discrepancy, sub_loss = criterion(pred_mel, target_mel, pred_semvec, target_semvec, xx_new, pred_speech_classifier)
                    elif self.use_somatosensory_feedback:
                        discrepancy, sub_loss = criterion(pred_mel, target_mel, pred_semvec, target_semvec, xx_new, pred_tube_mel, pred_tube_semvec)
                    else:
                        discrepancy, sub_loss = criterion(pred_mel, target_mel, pred_semvec, target_semvec, xx_new)
                    if (ii + 1) % log_ii == 0:
                        planned_loss_steps.append(float(discrepancy.item()))
                        planned_mel_loss_steps.append(float(sub_loss.mel_loss.item()))
                        vel_loss_steps.append(float(sub_loss.velocity_loss.item()))
                        jerk_loss_steps.append(float(sub_loss.jerk_loss.item()))
                        pred_semvec_loss_steps.append(float(sub_loss.semvec_loss.item()))
                        if self.use_speech_classifier:
                            pred_speech_classifier_loss_steps.append(float(sub_loss.speech_classifier_loss.item()))
                        if self.use_somatosensory_feedback:
                            pred_tube_mel_loss_steps.append(float(sub_loss.tube_mel_loss.item()))
                            pred_tube_semvec_loss_steps.append(float(sub_loss.tube_semvec_loss.item()))

                    if verbose:
                        print("Iteration %d" % ii)
                        print("Planned Loss: ", float(discrepancy.item()))
                        print("Mel Loss: ", float(sub_loss.mel_loss.item()))
                        print("Vel Loss: ", float(sub_loss.velocity_loss.item()))
                        print("Jerk Loss: ", float(sub_loss.jerk_loss.item()))
                        print("Local Linear Loss: ", float(sub_loss.local_linear_loss.item()))
                        print("Semvec Loss: ", float(sub_loss.semvec_loss.item()))
                        if self.use_speech_classifier:
                            print("Speech Classifier Loss: ", float(sub_loss.speech_classifier_loss.item()))
                        if self.use_somatosensory_feedback:
                            print("Tube Mel Loss: ", float(sub_loss.tube_mel_loss.item()))
                            print("Tube Semvec Loss: ", float(sub_loss.tube_semvec_loss.item()))

                elif objective == 'semvec':
                    if self.use_speech_classifier:
                        discrepancy, sub_loss = criterion(pred_semvec, target_semvec, xx_new, pred_speech_classifier)
                    elif self.use_somatosensory_feedback:
                        discrepancy, sub_loss = criterion(pred_semvec, target_semvec, xx_new, pred_tube_semvec)
                        tube_mel_loss = TUBE_MEL_WEIGHT * rmse_loss(pred_tube_mel, target_mel)
                    else:
                        discrepancy, sub_loss = criterion(pred_semvec, target_semvec, xx_new)
                    mel_loss = MEL_WEIGHT * rmse_loss(pred_mel, target_mel)

                    if (ii + 1) % log_ii == 0:
                        planned_loss_steps.append(float(discrepancy.item()))
                        vel_loss_steps.append(float(sub_loss.velocity_loss.item()))
                        jerk_loss_steps.append(float(sub_loss.jerk_loss.item()))
                        pred_semvec_loss_steps.append(float(sub_loss.semvec_loss.item()))
                        planned_mel_loss_steps.append(float(mel_loss.item()))

                        if self.use_speech_classifier:
                            pred_speech_classifier_loss_steps.append(float(sub_loss.speech_classifier_loss.item()))
                        if self.use_somatosensory_feedback:
                            pred_tube_semvec_loss_steps.append(float(sub_loss.tube_semvec_loss.item()))
                            pred_tube_mel_loss_steps.append(float(tube_mel_loss.item()))

                    if verbose:
                        print("Iteration %d" % ii)
                        print("Planned Loss: ", float(discrepancy.item()))
                        print("Mel Loss: ", float(mel_loss.item()))
                        print("Vel Loss: ", float(sub_loss.velocity_loss.item()))
                        print("Jerk Loss: ", float(sub_loss.jerk_loss.item()))
                        print("Semvec Loss: ", float(sub_loss.semvec_loss.item()))
                        if self.use_speech_classifier:
                            print("Speech Classifier Loss: ", float(sub_loss.speech_classifier_loss.item()))
                        if self.use_somatosensory_feedback:
                            print("Tube Mel Loss: ", float(tube_mel_loss.item()))
                            print("Tube Semvec Loss: ", float(sub_loss.tube_semvec_loss.item()))

                else:
                    raise ValueError(f'unkown objective {objective}')

                discrepancy.backward()

                # if verbose:
                # print(f"grad.abs().max() {grad.abs().max()}")
                if verbose:
                    if xx_new.grad.max() > 10:
                        print("WARNING: gradient is larger than 10")
                    if xx_new.grad.min() < -10:
                        print("WARNING: gradient is smaller than -10")

                if log_gradients:
                    grad_steps.append(xx_new.grad.detach().clone())

                if (ii + 1) % log_ii == 0:
                    xx_new_numpy = xx_new[-1, :, :].detach().cpu().numpy().copy()
                    cp_steps_ii.append(xx_new_numpy)

                    if self.use_somatosensory_feedback:
                        sig, sr, tube_info = speak_and_extract_tube_information(inv_normalize_cp(xx_new_numpy))
                        area_within_oral_cavity = get_area_info_within_oral_cavity(tube_info["tube_length_cm"], tube_info["tube_area_cm2"])
                        prod_tube = np.concatenate([area_within_oral_cavity,
                                                np.expand_dims(tube_info["incisor_pos_cm"], axis=1),
                                                np.expand_dims(tube_info["tongue_tip_side_elevation"], axis=1),
                                                np.expand_dims(tube_info["velum_opening_cm2"],axis=1)], axis=1)
                        prod_tube = normalize_tube(prod_tube)
                        prod_tube_steps_ii.append(prod_tube.copy())
                        prod_tube.shape = pred_tube.shape
                        prod_tube = torch.from_numpy(prod_tube)
                        prod_tube = prod_tube.to(self.device)

                        with torch.no_grad():
                            pred_tube = self.cp_tube_model(xx_new)
                            prod_tube_mel = self.tube_mel_model(prod_tube)

                        pred_tube_steps_ii.append(pred_tube[-1, :, :].detach().cpu().numpy().copy())

                        prod_tube_loss = rmse_loss(pred_tube, prod_tube)
                        prod_tube_loss_steps.append(float(prod_tube_loss.item()))

                        prod_tube_mel_loss = TUBE_MEL_WEIGHT * rmse_loss(prod_tube_mel, target_mel)
                        prod_tube_mel_loss_steps.append(float(prod_tube_mel_loss.item()))

                        pred_tube_mel_steps_ii.append(pred_tube_mel[-1, :, :].detach().cpu().numpy().copy())
                        prod_tube_mel_steps_ii.append(prod_tube_mel[-1, :, :].detach().cpu().numpy().copy())


                    else:
                        sig, sr = speak(inv_normalize_cp(xx_new_numpy))

                    if log_signals:
                        sig_steps.append(sig)

                    prod_mel = librosa_melspec(sig, sr)
                    prod_mel = normalize_mel_librosa(prod_mel)
                    prod_mel_steps_ii.append(prod_mel.copy())

                    prod_mel.shape = pred_mel.shape
                    prod_mel = torch.from_numpy(prod_mel)
                    prod_mel = prod_mel.to(self.device)
                    pred_mel_steps_ii.append(pred_mel[-1, :, :].detach().cpu().numpy().copy())

                    prod_loss = rmse_loss(prod_mel, target_mel)
                    prod_loss *= MEL_WEIGHT
                    prod_loss_steps.append(float(prod_loss.item()))

                    if self.use_speech_classifier:
                        prod_speech_classifier = self.speech_classifier(prod_mel)
                        prod_speech_classifier_loss = bce_loss(
                                prod_speech_classifier,
                                torch.zeros_like(prod_speech_classifier, dtype=prod_speech_classifier.dtype))
                        prod_speech_classifier_loss *= SPEECH_CLASSIFIER_WEIGHT

                        prod_speech_classifier_loss_steps.append(float(
                            prod_speech_classifier_loss.item()))

                    if verbose:
                        print("Produced Mel Loss: ", float(prod_loss.item()))
                        if self.use_speech_classifier:
                            print("Produced Speech Classifier Loss: ", float(prod_speech_classifier_loss.item()))
                        if self.use_somatosensory_feedback:
                            print("Produced Tube Loss: ", float(prod_tube_loss.item()))

                    if objective in ('semvec', 'acoustic_semvec') or log_semantics:
                        self.embedder = self.embedder.eval()
                        prod_semvec = self.embedder(prod_mel, (torch.tensor(prod_mel.shape[1]),))
                        prod_semvec_steps_ii.append(prod_semvec[-1, :].detach().cpu().numpy().copy())

                        prod_semvec_loss = rmse_loss(prod_semvec, target_semvec)
                        prod_semvec_loss *= SEMANTIC_WEIGHT
                        prod_semvec_loss_steps.append(float(prod_semvec_loss.item()))

                        if self.use_somatosensory_feedback:
                            self.tube_embedder = self.tube_embedder.eval()
                            prod_tube_semvec = self.tube_embedder(prod_tube, (torch.tensor(prod_tube.shape[1]),))
                            prod_tube_semvec_steps_ii.append(prod_tube_semvec[-1, :].detach().cpu().numpy().copy())

                            prod_tube_semvec_loss = rmse_loss(prod_tube_semvec, target_semvec)
                            prod_tube_semvec_loss *= TUBE_SEMANTIC_WEIGHT
                            prod_tube_semvec_loss_steps.append(float(prod_tube_semvec_loss.item()))

                        if verbose:
                            print("Produced Semvec Loss: ", float(prod_semvec_loss.item()))
                            if self.use_somatosensory_feedback:
                                print("Produced Tube Semvec Loss: ", float(prod_tube_semvec_loss.item()))
                            print("")

                        new_synthesis_acoustic = BestSynthesisAcoustic(float(prod_loss.item()), xx_new_numpy, sig, prod_mel[-1, :, :].detach().cpu().numpy().copy(), pred_mel[-1, :, :].detach().cpu().numpy().copy())
                        new_synthesis_semantic = BestSynthesisSemantic(float(prod_semvec_loss.item()), xx_new_numpy, sig, prod_semvec[-1, :].detach().cpu().numpy().copy(), pred_semvec[-1, :].detach().cpu().numpy().copy())

                        if self.best_synthesis_acoustic.mel_loss > new_synthesis_acoustic.mel_loss:
                            self.best_synthesis_acoustic = new_synthesis_acoustic
                        if self.best_synthesis_semantic.semvec_loss > new_synthesis_semantic.semvec_loss:
                            self.best_synthesis_semantic = new_synthesis_semantic

                        if self.use_somatosensory_feedback:
                            new_synthesis_somatosensory = BestSynthesisSomatosensory(float(prod_tube_loss.item()), float(prod_tube_mel_loss.item()), float(prod_tube_semvec_loss.item()),
                                                                                           xx_new_numpy, sig,
                                                                                           prod_tube,
                                                                                           pred_tube[-1, :].detach().cpu().numpy().copy(),
                                                                                           prod_tube_mel[-1, :].detach().cpu().numpy().copy(),
                                                                                           pred_tube_mel[-1, :].detach().cpu().numpy().copy(),
                                                                                           prod_tube_semvec[-1, :].detach().cpu().numpy().copy(),
                                                                                           pred_tube_semvec[-1, :].detach().cpu().numpy().copy())
                            if self.best_synthesis_somatosensory.tube_loss > new_synthesis_somatosensory.tube_loss:
                                self.best_synthesis_somatosensory = new_synthesis_somatosensory


                    else:
                        new_synthesis_acoustic = BestSynthesisAcoustic(float(prod_loss.item()), xx_new_numpy, sig, prod_mel[-1, :, :].detach().cpu().numpy().copy(),pred_mel[-1, :, :].detach().cpu().numpy().copy())
                        if self.best_synthesis_acoustic.mel_loss > new_synthesis_acoustic.mel_loss:
                            self.best_synthesis_acoustic = new_synthesis_acoustic

                        if self.use_somatosensory_feedback:
                            new_synthesis_somatosensory = BestSynthesisSomatosensory(float(prod_tube_loss.item()), float(prod_tube_mel_loss.item()), np.inf,
                                                                                           xx_new_numpy, sig,
                                                                                           prod_tube,
                                                                                           pred_tube[-1, :].detach().cpu().numpy().copy(),
                                                                                           prod_tube_mel[-1, :].detach().cpu().numpy().copy(),
                                                                                           pred_tube_mel[-1, :].detach().cpu().numpy().copy(),
                                                                                           None,
                                                                                           None)
                            if self.best_synthesis_somatosensory.tube_loss > new_synthesis_somatosensory.tube_loss:
                                self.best_synthesis_somatosensory = new_synthesis_somatosensory

                    if verbose:
                        print("")

                optimizer.step()

                with torch.no_grad():
                    xx_new.data = xx_new.data.clamp(-1.05, 1.05) # clamp between -1.05 and 1.05
                    if self.smiling:
                        #Vocal tract parameters: "HX HY JX JA LP LD VS VO TCX TCY TTX TTY TBX TBY TRX TRY TS1 TS2 TS3"
                        #Glottis parameters: "f0 pressure x_bottom x_top chink_area lag rel_amp double_pulsing pulse_skewness flutter aspiration_strength "
                        # keep LP and HY at the maximum
                        xx_new.data[:, :, 4] = -1.0  # "LP"
                        xx_new.data[:, :, 1] = 1.0  # "HY"

                    if not past_cp is None:
                        xx_new.data[:, 0:past_cp_torch.shape[0], :] = past_cp_torch


            if plot:
                target_mel_ii = target_mel[-1, :, :].detach().cpu().numpy().copy()
                prod_mel_ii = prod_mel[-1, :, :].detach().cpu().numpy().copy()
                pred_mel_ii = pred_mel[-1, :, :].detach().cpu().numpy().copy()

                if plot is True:
                    visualize.plot_mels(True, target_mel_ii, initial_pred_mel,
                            initial_prod_mel, pred_mel_ii, prod_mel_ii)
                else:
                    visualize.plot_mels(f"{plot}_{ii_outer:03d}.png",
                            target_mel_ii, initial_pred_mel, initial_prod_mel,
                            pred_mel_ii, prod_mel_ii)

            prod_mel_steps.append(prod_mel_steps_ii)
            if log_cps:
                cp_steps.append(cp_steps_ii)
            pred_mel_steps.append(pred_mel_steps_ii)
            pred_semvec_steps.append(pred_semvec_steps_ii)
            prod_semvec_steps.append(prod_semvec_steps_ii)

            if self.use_somatosensory_feedback:
                prod_tube_steps.append(prod_tube_steps_ii)
                pred_tube_steps.append(pred_tube_steps_ii)
                prod_tube_mel_steps.append(prod_tube_mel_steps_ii)
                pred_tube_mel_steps.append(pred_tube_mel_steps_ii)
                pred_tube_semvec_steps.append(pred_tube_semvec_steps_ii)
                prod_tube_semvec_steps.append(prod_tube_semvec_steps_ii)


            # execute and continue learning
            if continue_learning:
                produced_data_ii = pd.DataFrame(columns=['vector', 'cp_norm', 'melspec_norm_synthesized', "tube_norm", 'segment_data'])
                produced_data_ii["cp_norm"] = cp_steps_ii
                produced_data_ii["melspec_norm_synthesized"] = prod_mel_steps_ii
                produced_data_ii["vector"] = [target_semvec[0].detach().cpu().numpy().copy() for _ in range(len(produced_data_ii))]
                produced_data_ii["segment_data"] = False
                if self.use_somatosensory_feedback:
                    produced_data_ii["tube_norm"] = prod_tube_steps_ii

                if add_training_data_pred or add_training_data_inv:
                    # update with new sample
                    if len(produced_data_ii) < int(0.5 * batch_size) * n_batches: # too few samples synthesized for filling 50% of batch_size * n_batches
                        batch_train_produced = random.sample(range(len(produced_data_ii)), k=len(produced_data_ii))
                        batch_train = random.sample(range(len(self.continue_data)), k=len(produced_data_ii))
                        n_train_batches = int(np.ceil(2*len(produced_data_ii)/batch_size))
                        reduced_last_batch = (2*len(produced_data_ii))%batch_size
                        print("Enhanced training data")
                        print(f"Not enough data produced to fill 50% of {n_batches} batches...")
                        if n_train_batches < n_batches:
                            print(f"Training on {n_train_batches} batches instead...")
                        if reduced_last_batch > 0:
                            print(f"Last batch reduced to {reduced_last_batch} samples instead of {batch_size}...")
                        print(" ")
                    else:
                        batch_train = random.sample(range(len(self.continue_data)), k=int(0.5 * batch_size) * n_batches)
                        batch_train_produced = random.sample(range(len(produced_data_ii)), k=int(0.5 * batch_size) * n_batches)
                        n_train_batches = n_batches

                    if self.use_somatosensory_feedback:
                        train_data_samples = self.continue_data[['vector','cp_norm', 'melspec_norm_synthesized', 'tube_norm','segment_data']].iloc[
                            batch_train].reset_index(drop=True)
                    else:
                        train_data_samples = self.continue_data[['vector', 'cp_norm', 'melspec_norm_synthesized', 'segment_data']].iloc[
                            batch_train].reset_index(drop=True)

                    produced_data_ii_samples = produced_data_ii.iloc[batch_train_produced].reset_index(drop=True)
                    continue_data_ii = pd.concat([train_data_samples, produced_data_ii_samples])


                    continue_data_ii["lens_input"] = np.array(continue_data_ii["cp_norm"].apply(len), dtype=int)
                    continue_data_ii["lens_output"] = np.array(continue_data_ii["melspec_norm_synthesized"].apply(len),dtype=int)
                    continue_data_ii.sort_values(by="lens_input", inplace=True)

                    del train_data_samples

                if len(produced_data_ii) <  batch_size * n_batches: # too few samples synthesized for filling batch_size * n_batches
                    batch_train_produced = random.sample(range(len(produced_data_ii)), k=len(produced_data_ii))
                    n_train_batches = int(np.ceil(len(produced_data_ii)/batch_size))
                    reduced_last_batch = len(produced_data_ii)%batch_size
                    print("Produced training data")
                    print(f"Not enough data produced to fill {n_batches} batches...")
                    if n_train_batches < n_batches:
                        print(f"Training on {n_train_batches} batches instead...")
                    if reduced_last_batch > 0:
                        print(f"Last batch reduced to {reduced_last_batch} samples instead of {batch_size}...")
                    print(" ")
                else:
                    batch_train_produced = random.sample(range(len(produced_data_ii)), k= batch_size * n_batches)
                    n_train_batches = n_batches

                produced_data_ii_samples = produced_data_ii.iloc[batch_train_produced].reset_index(drop=True)
                produced_data_ii_samples["lens_input"] = np.array(produced_data_ii_samples["cp_norm"].apply(len), dtype=int)
                produced_data_ii_samples["lens_output"] = np.array(produced_data_ii_samples["melspec_norm_synthesized"].apply(len),
                                                           dtype=int)
                produced_data_ii_samples.sort_values(by="lens_input", inplace=True)

                if add_training_data_pred:
                    training_data_pred = continue_data_ii
                else:
                    training_data_pred = produced_data_ii_samples

                training_length_dict_pred = {}
                lens_training_cps_pred = np.asarray(training_data_pred.lens_input)
                lengths, counts = np.unique(lens_training_cps_pred, return_counts=True)
                sorted_training_length_keys_pred = np.sort(lengths)
                for length in sorted_training_length_keys_pred:
                    training_length_dict_pred[length] = np.where(lens_training_cps_pred == length)[0]

                inps = training_data_pred["cp_norm"]
                tgts = training_data_pred["melspec_norm_synthesized"]

                lens_input = torch.tensor(np.array(training_data_pred.lens_input)).to(self.device)
                lens_output = torch.tensor(np.array(training_data_pred.lens_output)).to(self.device)

                if self.use_somatosensory_feedback:
                    tgts_tube = training_data_pred["tube_norm"]
                    lens_output_tube = torch.tensor(np.array(training_data_pred.lens_input)).to(self.device)

                if continue_learning_inv:
                    if add_training_data_inv:
                        training_data_inv = continue_data_ii
                    else:
                        training_data_inv = produced_data_ii_samples

                    training_length_dict_inv = {}
                    lens_training_cps_inv = np.asarray(training_data_inv.lens_input)
                    lengths, counts = np.unique(lens_training_cps_inv, return_counts=True)
                    sorted_training_length_keys_inv = np.sort(lengths)
                    for length in sorted_training_length_keys_inv:
                        training_length_dict_inv[length] = np.where(lens_training_cps_inv == length)[0]

                    inps_inv = training_data_inv["melspec_norm_synthesized"]
                    tgts_inv = training_data_inv["cp_norm"]

                    lens_input_inv = torch.tensor(np.array(training_data_inv.lens_output)).to(self.device)
                    lens_output_inv = torch.tensor(np.array(training_data_inv.lens_input)).to(self.device)



                for e in range(n_epochs):
                    epoch_ii = self.create_epoch_batches(training_data_pred, batch_size, shuffle=True,
                                                         same_size_batching=True,
                                                         training_length_dict=training_length_dict_pred)
                    avg_loss = list()
                    if continue_learning_tube and self.use_somatosensory_feedback:
                        avg_loss_tube = list()
                        avg_loss_tube_mel = list()

                    for j in epoch_ii:
                        lens_input_j = lens_input[j]
                        batch_input = inps.iloc[j]
                        batch_input = pad_batch_online(lens_input_j, batch_input, self.device)

                        lens_output_j = lens_output[j]
                        batch_output = tgts.iloc[j]
                        batch_output = pad_batch_online(lens_output_j, batch_output, self.device)


                        Y_hat = self.pred_model(batch_input, lens_input_j)

                        self.pred_optimizer.zero_grad()
                        pred_loss = self.pred_criterion(Y_hat, batch_output)
                        pred_loss.backward()
                        self.pred_optimizer.step()

                        avg_loss.append(float(pred_loss.item()))

                        if continue_learning_tube and self.use_somatosensory_feedback:
                            lens_output_tube_j = lens_output_tube[j]
                            batch_output_tube = tgts_tube.iloc[j]
                            batch_output_tube = pad_batch_online(lens_output_tube_j, batch_output_tube, self.device)

                            # learn cp_tube_model
                            Y_hat = self.cp_tube_model(batch_input, lens_input_j)

                            self.tube_optimizer.zero_grad()
                            tube_loss = self.tube_criterion(Y_hat, batch_output_tube)
                            tube_loss.backward()
                            self.tube_optimizer.step()

                            avg_loss_tube.append(float(tube_loss.item()))

                            # learn tube_mel_model
                            Y_hat = self.tube_mel_model(batch_output_tube)

                            self.tube_mel_optimizer.zero_grad()
                            tube_mel_loss = self.tube_mel_criterion(Y_hat, batch_output)
                            tube_mel_loss.backward()
                            self.tube_mel_optimizer.step()

                            avg_loss_tube_mel.append(float(tube_mel_loss.item()))

                    pred_model_loss.append(np.mean(avg_loss))

                    if continue_learning_tube and self.use_somatosensory_feedback:
                        tube_model_loss.append(np.mean(avg_loss_tube))
                        tube_mel_model_loss.append(np.mean(avg_loss_tube_mel))

                if continue_learning_inv:
                    for e in range(n_epochs):
                        epoch_ii = self.create_epoch_batches(training_data_inv, batch_size, shuffle=True,
                                                             same_size_batching=True,
                                                             training_length_dict=training_length_dict_inv)
                        avg_loss_inv = list()
                        for j in epoch_ii:
                            lens_input_j = lens_input_inv[j]
                            batch_input = inps_inv.iloc[j]
                            batch_input = pad_batch_online(lens_input_j, batch_input, self.device)

                            lens_output_j = lens_output_inv[j]
                            batch_output = tgts_inv.iloc[j]
                            batch_output = pad_batch_online(lens_output_j, batch_output, self.device)


                            Y_hat = self.inv_model(batch_input, lens_input_j)

                            self.inv_optimizer.zero_grad()
                            inv_loss = self.inv_criterion(Y_hat, batch_output)
                            inv_sub_losses = inv_loss[1:]  # sublosses
                            inv_loss = inv_loss[0]  # total loss
                            inv_loss.backward()
                            self.inv_optimizer.step()

                            avg_loss_inv.append(float(inv_loss.item()))

                        inv_model_loss.append(np.mean(avg_loss_inv))


                if not self.continue_data is None:
                    self.continue_data = pd.concat([self.continue_data, produced_data_ii]).reset_index(drop=True)
                    if len(self.continue_data) > self.continue_data_limit:
                        random_sample = random.sample(range(len(self.continue_data)), k=self.continue_data_limit)
                        self.continue_data = self.continue_data.iloc[random_sample].reset_index(drop=True)

                del produced_data_ii_samples
                del produced_data_ii
                del training_data_pred
                if continue_learning_inv:
                    del training_data_inv
                if add_training_data_pred or add_training_data_inv:
                    del continue_data_ii

        planned_cp = xx_new[-1, :, :].detach().cpu().numpy()
        prod_sig = sig
        prod_sr = sr

        with torch.no_grad():
            pred_mel = self.pred_model(xx_new)
            self.embedder = self.embedder.eval()
            pred_semvec = self.embedder(pred_mel, (torch.tensor(pred_mel.shape[1]),))
            prod_semvec = self.embedder(prod_mel, (torch.tensor(prod_mel.shape[1]),))

        target_mel = target_mel[-1, :, :].detach().cpu().numpy()
        prod_mel = prod_mel[-1, :, :].detach().cpu().numpy()
        pred_mel = pred_mel[-1, :, :].detach().cpu().numpy()
        prod_semvec = prod_semvec[-1, :].detach().cpu().numpy()
        pred_semvec = pred_semvec[-1, :].detach().cpu().numpy()

        if self.use_somatosensory_feedback:
            with torch.no_grad():
                pred_tube = self.cp_tube_model(xx_new)
                prod_tube_mel = self.tube_mel_model(prod_tube)
                pred_tube_mel = self.tube_mel_model(pred_tube)

                self.tube_embedder = self.tube_embedder.eval()
                prod_tube_semvec = self.tube_embedder(prod_tube, (torch.tensor(prod_tube.shape[1]),))
                pred_tube_semvec = self.tube_embedder(pred_tube, (torch.tensor(prod_tube.shape[1]),))

            prod_tube = prod_tube[-1, :, :].detach().cpu().numpy()
            pred_tube = pred_tube[-1, :, :].detach().cpu().numpy()
            prod_tube_mel = prod_tube_mel[-1, :, :].detach().cpu().numpy()
            pred_tube_mel = pred_tube_mel[-1, :, :].detach().cpu().numpy()
            prod_tube_semvec = prod_tube_semvec[-1, :].detach().cpu().numpy()
            pred_tube_semvec = pred_tube_semvec[-1, :].detach().cpu().numpy()


        print("--- %.2f min ---" % ((time.time() - start_time) / 60))

        #  0. planned_cp
        #  1. initial_cp
        #  3. initial_sig
        #  4. initial_sr
        #  5. initial_prod_mel
        #  6. initial_pred_mel
        #  7. target_sig
        #  8. target_sr
        #  9. target_mel
        # 10. prod_sig
        # 11. prod_sr
        # 12. prod_mel
        # 13. pred_mel
        # 14. initial_prod_semvec
        # 15. initial_pred_semvec
        # 16. prod_semvec
        # 17. pred_semvec
        # 18. prod_loss_steps
        # 19. planned_loss_steps
        # 20. planned_mel_loss_steps
        # 21. vel_loss_steps
        # 22. jerk_loss_steps
        # 23. pred_semvec_loss_steps
        # 24. prod_semvec_loss_steps
        # 25. cp_steps
        # 26. pred_semvec_steps
        # 27. prod_semvec_steps
        # 28. grad_steps
        # 29. sig_steps
        # 30. prod_mel_steps
        # 31. pred_mel_steps
        # 32. pred_model_loss
        # 33. inv_model_loss
        if self.use_speech_classifier:
            return PlanningResultsWithSpeechClassifier(planned_cp, initial_cp, initial_sig, initial_sr, initial_prod_mel, initial_pred_mel,
                    target_sig, target_sr, target_mel, prod_sig, prod_sr, prod_mel,
                    pred_mel, initial_prod_semvec, initial_pred_semvec, prod_semvec, pred_semvec, prod_loss_steps, planned_loss_steps,
                    planned_mel_loss_steps, vel_loss_steps, jerk_loss_steps,
                    pred_semvec_loss_steps, prod_semvec_loss_steps, pred_speech_classifier_loss_steps, prod_speech_classifier_loss_steps,
                    cp_steps, pred_semvec_steps, prod_semvec_steps, grad_steps, sig_steps,
                    prod_mel_steps, pred_mel_steps, pred_model_loss, inv_model_loss)
        elif self.use_somatosensory_feedback:
            return PlanningResultsWithSomatosensory(planned_cp, initial_cp, initial_sig, initial_sr, initial_prod_mel, initial_pred_mel, initial_prod_tube, initial_pred_tube, initial_prod_tube_mel, initial_pred_tube_mel,
                                   target_sig, target_sr, target_mel, prod_sig, prod_sr, prod_mel,
                                   pred_mel, prod_tube, pred_tube, prod_tube_mel, pred_tube_mel, initial_prod_semvec, initial_pred_semvec, initial_prod_tube_semvec, initial_pred_tube_semvec, prod_semvec, pred_semvec, prod_tube_semvec, pred_tube_semvec,
                                   prod_loss_steps, planned_loss_steps,
                                   planned_mel_loss_steps, vel_loss_steps, jerk_loss_steps,
                                   pred_semvec_loss_steps, prod_semvec_loss_steps, prod_tube_loss_steps, pred_tube_mel_loss_steps, prod_tube_mel_loss_steps, pred_tube_semvec_loss_steps, prod_tube_semvec_loss_steps, cp_steps,
                                   pred_semvec_steps, prod_semvec_steps, grad_steps, sig_steps,
                                   prod_mel_steps, pred_mel_steps, prod_tube_steps, pred_tube_steps, prod_tube_mel_steps, pred_tube_mel_steps, prod_tube_semvec_steps, pred_tube_semvec_steps, pred_model_loss, inv_model_loss, tube_model_loss, tube_mel_model_loss)

        else:
            return PlanningResults(planned_cp, initial_cp, initial_sig, initial_sr, initial_prod_mel, initial_pred_mel,
                    target_sig, target_sr, target_mel, prod_sig, prod_sr, prod_mel,
                    pred_mel, initial_prod_semvec, initial_pred_semvec, prod_semvec, pred_semvec, prod_loss_steps, planned_loss_steps,
                    planned_mel_loss_steps, vel_loss_steps, jerk_loss_steps,
                    pred_semvec_loss_steps, prod_semvec_loss_steps, cp_steps,
                    pred_semvec_steps, prod_semvec_steps, grad_steps, sig_steps,
                    prod_mel_steps, pred_mel_steps, pred_model_loss, inv_model_loss)

