# coding: utf-8
# Echo Chamber Model
# analysis.py
# Last Update: 20190410
# by Kazutoshi Sasahara

import numpy as np
import scipy.stats as stats
import peakutils
from scipy.linalg import LinAlgError


def screen_diversity(content_values, bins):
    h, w = np.histogram(content_values, range=(-1, 1), bins=bins)
    return stats.entropy(h + 1, base=2)


def num_opinion_peaks(opinions):
    try:
        nparam_density = stats.kde.gaussian_kde(opinions)
    except LinAlgError:
        return np.nan
    x = np.linspace(-1, 1, 100)
    density = nparam_density(x)
    indexes = peakutils.indexes(density, thres=0, min_dist=10)
    # try:
    #     print(x[indexes], density[indexes])
    # except IndexError:
    #     print(f"No peaks, {indexes}")
    return len(indexes)
