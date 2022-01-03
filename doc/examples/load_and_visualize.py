"""
Load and visualize results.

"""

import pickle

from paule import visualize

with open('results/test3-continued.pkl', 'rb') as pfile:
    results = pickle.load(pfile)

visualize.visualize_result(results, 'test42', 'results')
