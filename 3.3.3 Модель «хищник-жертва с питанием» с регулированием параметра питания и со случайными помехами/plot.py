#%%
"""
Code for plotting NAS. Goal - x1 - x1c
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml
import pprint
import os
import sys
import numpy as np

sys.path.append("..")
from models import PreyPredatorBigPhaseNASFactory

mpl.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 20})
plt.rcParams['svg.fonttype'] = 'none'
np.random.seed(20)

with open("params_x1c.yaml") as stream:
    try:
        params = yaml.safe_load(stream)
        pprint.pprint(params)
        params['x1c'] = params['alpha2'] / params['beta2']
    except yaml.YAMLError as exc:
        print(exc)

factory = PreyPredatorBigPhaseNASFactory()
model = factory.create_model()
model.set_parameters(
    beta1=params['beta1'],
    alpha1=params['alpha1'],
    alpha2=params['alpha2'],
    beta2=params['beta2'],
    c=params['c'],
    sigma=params['sigma'],
    T1=params['T1'],
    T2=params['T2'],
    N=params['N'],
    h=params['h'],
    x=params['x'],
    type_goal='x1c'
)
x, u = model.calculate(x1c=params['x1c'])
model.plot(x, u, x1c=params['x1c'], save_fig=True, name_fig1="3.26а", name_fig2="3.26б", name_fig3="3.27")