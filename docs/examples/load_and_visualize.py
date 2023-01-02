"""
Load and visualize results.

"""

import pickle

from paule import visualize, util

with open('results/test3.pkl', 'rb') as pfile:
    results = pickle.load(pfile)

visualize.visualize_results(results, 'test3-vis', 'results')

# set f0 to 110 Hz
planned_cp = results.planned_cp.copy()
planned_cp = util.inv_normalize_cp(planned_cp)

planned_cp[:, 19:20] = 110.
sig, sr = util.speak(planned_cp)
sf.write('test3_f0110Hz.flac', 4 * sig, sr)

