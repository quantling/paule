#!/usr/bin/env python3

# pylint: disable=C0111, redefined-outer-name

import os
#import tempfile

import torch
import numpy as np
import pytest

from paule import paule, util

TEST_ROOT = os.path.join(os.path.pardir, os.path.dirname(__file__))
TARGET_ACOUSTICS = os.path.join(TEST_ROOT, "resources/target_acoustics.flac")
TARGET_SEMVEC = np.zeros((300,))

#TMP_PATH = tempfile.mkdtemp()


@pytest.fixture(scope='module')
def paule_model():
    util.download_pretrained_weights()
    return paule.Paule(device=torch.device('cpu'))

@pytest.fixture(scope='module')
def cp_11zeros():
    return np.zeros((11, 30))


def test_exceptions(paule_model, cp_11zeros):
    with pytest.raises(ValueError) as e_info:
        paule_model.plan_resynth(target_acoustic=None, target_semvec=None)
        assert e_info == 'Either target_acoustic or target_semvec has to be not None.'

    with pytest.raises(ValueError) as e_info:
        paule_model.plan_resynth(target_acoustic=TARGET_ACOUSTICS, target_semvec=None, n_inner=5, log_ii=10)
        assert e_info == 'results can only be logged between first and last planning step'

    with pytest.raises(ValueError) as e_info:
        paule_model.plan_resynth(target_acoustic=None, target_semvec=TARGET_SEMVEC)
        assert e_info == 'if target_acoustic is None you need to give a target_seq_length and a target_semvec'

    with pytest.raises(ValueError) as e_info:
        paule_model.plan_resynth(target_acoustic=TARGET_ACOUSTICS, initialize_from='ERROR')
        assert e_info == "initialize_from has to be either 'acoustic' or 'semvec'"

    with pytest.raises(ValueError) as e_info:
        paule_model.plan_resynth(target_acoustic=TARGET_ACOUSTICS, initial_cp=cp_11zeros, initialize_from='ERROR')
        assert e_info == "one of initial_cp and initialize_from has to be None"

    with pytest.raises(ValueError) as e_info:
        paule_model.plan_resynth(target_acoustic=TARGET_ACOUSTICS, initial_cp=cp_11zeros)
        assert e_info == "initialize_from has to be either 'acoustic' or 'semvec'"

    with pytest.raises(ValueError) as e_info:
        paule_model.plan_resynth(target_acoustic=TARGET_ACOUSTICS, past_cp=cp_11zeros)
        assert e_info == "past_cp have to be None or the sequence length has to be an even number"

    with pytest.raises(ValueError) as e_info:
        paule_model.plan_resynth(target_acoustic=TARGET_ACOUSTICS, objective="ERROR")
        assert e_info == "objective has to be one of 'acoustic_semvec', 'acoustic' or 'semvec'"


def test_plan_resynth(paule_model):
    results = paule_model.plan_resynth(target_acoustic=TARGET_ACOUSTICS,
                                       objective='acoustic_semvec',
                                       initialize_from='acoustic', n_outer=2,
                                       n_inner=2, n_batches=1, batch_size=2,
                                       n_epochs=2, verbose=True)


def clock(func, args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    stop = time.time()

    duration = stop - start

    return result, duration

