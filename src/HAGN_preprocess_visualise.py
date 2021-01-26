"""Script to plot a correlation and a histogram matrix for HAGN data."""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

plt.switch_backend('Agg')


match = np.load('/mnt/zfsusers/rstiskalek/hydro/data/HAGN_matched_catalog.npy')

data = {}
pars = ['logMvir', 'Rvir', 'concentration', 'rs', 'rho0', 'Reff', 'logMS']
for p in pars:
    if p == 'Rvir':
        data[p] = match['rs'] * match['concentration']
    elif p in ['Reff']:
        data[p] = match[p] * 1000  # in kpc
    elif p == 'rho0':
        data[p] = np.log10(match[p])
    else:
        data[p] = match[p]
data = pd.DataFrame(data)

# Histograms
fig = plt.figure(figsize=(10, 10))
ax = fig.gca()
data.hist(ax=ax, bins='auto', log=True)
fig.savefig('/mnt/zfsusers/rstiskalek/hydro/plots/HAGN_histograms.png')


# Correlation matrix
fig = plt.figure()
sns.heatmap(data.corr(), annot=True)
fig.savefig('/mnt/zfsusers/rstiskalek/hydro/plots/HAGN_correlations.png')
