#%%
"""
Code for plotting NPZ. Goal - x2 - x2c
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml
import pprint
import os
import sys
import numpy as np

sys.path.append("..")
from models import NPZFactory

mpl.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 20})
plt.rcParams['svg.fonttype'] = 'none'

with open("params_x2c.yaml") as stream:
    try:
        params = yaml.safe_load(stream)
        pprint.pprint(params)
        params['x2c'] = params['b'] / params['d']
    except yaml.YAMLError as exc:
        print(exc)

factory = NPZFactory()
model = factory.create_model()
model.set_parameters(
    a=params['a'],
    b=params['b'],
    c=params['c'],
    d=params['d'],
    T1=params['T1'],
    T2=params['T2'],
    N=params['N'],
    h=params['h'],
    x=params['x'],
    type_goal=params['type_goal']
)
x, u = model.calculate(x2c=params['x2c'])
model.plot(x, u, x2c=params['x2c'], save_fig=True, name_fig1="3.18а", name_fig2="3.18б", name_fig3="3.19")

#%%
"""
Code for plotting NPZ model. Goal - x1 + rho * x2 - d
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml
import pprint
import os
import sys
import numpy as np

sys.path.append("..")
from models import NPZFactory

mpl.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 20})
plt.rcParams['svg.fonttype'] = 'none'

with open("params_rho_d.yaml") as stream:
    try:
        params = yaml.safe_load(stream)
        pprint.pprint(params)
    except yaml.YAMLError as exc:
        print(exc)

factory = NPZFactory()
model = factory.create_model()
model.set_parameters(
    a=params['a'],
    b=params['b'],
    c=params['c'],
    d=params['d'],
    T1=params['T1'],
    T2=params['T2'],
    N=params['N'],
    h=params['h'],
    x=params['x'],
    type_goal=params['type_goal']
)
x, u = model.calculate(rho=params['rho'], q=params['q'])
model.plot(x, u, rho=params['rho'], q=params['q'], save_fig=True, name_fig1="3.20а", name_fig2="3.20б", name_fig3="3.21")