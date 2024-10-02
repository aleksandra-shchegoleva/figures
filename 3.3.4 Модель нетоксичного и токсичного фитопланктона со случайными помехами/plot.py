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
from models import NTP_NAS_Factory

mpl.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 20})
plt.rcParams['svg.fonttype'] = 'none'
np.random.seed(20)

with open("params_x2c_1.yaml") as stream:
    try:
        params = yaml.safe_load(stream)
        pprint.pprint(params)
    except yaml.YAMLError as exc:
        print(exc)

factory = NTP_NAS_Factory()
model = factory.create_model()
model.set_parameters(
    r1=params['r1'],
    r2=params['r2'],
    K1=params['K1'],
    K2=params['K2'],
    alpha1=params['alpha1'],
    alpha2=params['alpha2'],
    w1=params['w1'],
    w2=params['w2'],
    d1=params['d1'],
    d2=params['d2'],
    b1=params['b1'],
    gamma1=params['gamma1'],
    gamma2=params['gamma2'],
    m=params['m'],
    m1=params['m1'],
    c=params['c'],
    sigma=params['sigma'],
    T=params['T'],
    N=params['N'],
    h=params['h'],
    x=params['x'],
    type_goal='x2c'
)
x, u = model.calculate(x2c=params['x2c'])
model.plot(x, u, x2c=params['x2c'], save_fig=True, name_fig1="3.28", name_fig2="3.28б")
