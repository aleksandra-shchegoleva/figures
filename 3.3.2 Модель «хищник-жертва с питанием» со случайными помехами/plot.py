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
from models import PreyPredatorNASFactory

mpl.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 20})
plt.rcParams['svg.fonttype'] = 'none'
np.random.seed(20)

with open("params_x1c_add.yaml") as stream:
    try:
        params = yaml.safe_load(stream)
        pprint.pprint(params)
        params['x1c'] = params['alpha2'] / params['beta2']
    except yaml.YAMLError as exc:
        print(exc)

factory = PreyPredatorNASFactory()
model = factory.create_model()
model.set_parameters(
    beta1=params['beta1'],
    alpha1=params['alpha1'],
    alpha2=params['alpha2'],
    beta2=params['beta2'],
    c=params['c'],
    sigma=params['sigma'],
    T=params['T'],
    N=params['N'],
    h=params['h'],
    x=params['x'],
    type_goal='x1c_multi'
)
x, u = model.calculate(x1c=params['x1c'])
model.plot(x, u, x1c=params['x1c'], save_fig=True, name_fig1="3.22а", name_fig2="3.22б", name_fig3="3.25а")

#%%
"""
Code for plotting NAS. Goal - x1 + rho * x2 - d
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml
import pprint
import os
import sys
import numpy as np

sys.path.append("..")
from models import PreyPredatorNASFactory

mpl.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 20})
plt.rcParams['svg.fonttype'] = 'none'
np.random.seed(20)

with open("params_rho_d.yaml") as stream:
    try:
        params = yaml.safe_load(stream)
        pprint.pprint(params)
    except yaml.YAMLError as exc:
        print(exc)

factory = PreyPredatorNASFactory()
model = factory.create_model()
model.set_parameters(
    beta1=params['beta1'],
    alpha1=params['alpha1'],
    alpha2=params['alpha2'],
    beta2=params['beta2'],
    c=params['c'],
    sigma=params['sigma'],
    T=params['T'],
    N=params['N'],
    h=params['h'],
    x=params['x'],
    type_goal='rho_d'
)
x, u = model.calculate(rho=params['rho'], d=params['d'])
model.plot(x, u, rho=params['rho'], d=params['d'], save_fig=True, name_fig1="3.24а", name_fig2="3.24б", name_fig3="3.25б")