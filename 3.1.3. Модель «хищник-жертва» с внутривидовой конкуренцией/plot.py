#%%
"""
Code for plotting prey-predator model intraspecifi concuration. Goal - x1 - x1c
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml
import pprint
import os
import sys

sys.path.append("..")
from models import PreyPredatorIntrComFactory

mpl.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 20})
plt.rcParams['svg.fonttype'] = 'none'

with open("params_x1c.yaml") as stream:
    try:
        params = yaml.safe_load(stream)
        pprint.pprint(params)
    except yaml.YAMLError as exc:
        print(exc)

factory = PreyPredatorIntrComFactory()
model = factory.create_model()
model.set_parameters(
    alpha1=params['alpha1'],
    alpha2=params['alpha2'],
    beta1=params['beta1'],
    beta2=params['beta2'],
    gamma1=params['gamma1'],
    gamma2=params['gamma2'],
    T=params['T'],
    N=params['N'],
    h=params['h'],
    x=params['x'],
    type_goal='x1c'
)
x, u = model.calculate(x1c=params['x1c'])
model.plot(x, u, x1c=params['x1c'], save_fig=True, name_fig1="3.6а", name_fig2="3.6б", name_fig3="3.8а")

#%%
"""
Code for plotting prey-predator model intraspecifi concuration. Goal - x1 + rho * x2 - d
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml
import pprint
import os
import sys

sys.path.append("..")
from models import PreyPredatorIntrComFactory

mpl.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 20})
plt.rcParams['svg.fonttype'] = 'none'

with open("params_rho_d.yaml") as stream:
    try:
        params = yaml.safe_load(stream)
        pprint.pprint(params)
    except yaml.YAMLError as exc:
        print(exc)

factory = PreyPredatorIntrComFactory()
model = factory.create_model()
model.set_parameters(
    alpha1=params['alpha1'],
    alpha2=params['alpha2'],
    beta1=params['beta1'],
    beta2=params['beta2'],
    gamma1=params['gamma1'],
    gamma2=params['gamma2'],
    T=params['T'],
    N=params['N'],
    h=params['h'],
    x=params['x'],
    type_goal='rho_d'
)
x, u = model.calculate(rho=params['rho'], d=params['d'])
model.plot(x, u, rho=params['rho'], d=params['d'], save_fig=True, name_fig1="3.7а", name_fig2="3.7б", name_fig3="3.8б")