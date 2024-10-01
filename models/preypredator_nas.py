from models.models import AbstractModel, AbstractFactory
import matplotlib.pyplot as plt
from typing import List, Tuple, NoReturn
import numpy as np
import seaborn as sns
import random

class PreyPredatorNASFactory(AbstractFactory):
    """
    Factory class for discret systems 
    """
    def create_model(self):
        return PreyPredatorNAS()

class PreyPredatorNAS(AbstractModel):
    """
    Class for model
    """
    def set_parameters(
        self,
        beta1: float,
        alpha1: None | float,
        alpha2: float,
        beta2: float,
        c: float,
        sigma: float,
        T: float,
        N: float,
        h: float,
        x: List[float],
        type_goal: str
        ) -> NoReturn:
        
        self.beta1 = beta1
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta2 = beta2
        self.c = c
        self.sigma = sigma
        self.T = T
        self.M = np.arange(0, N, h)
        self.N = N
        self.h = h
        self.x = np.empty((0, 2), dtype=np.float16)
        self.x = np.vstack((self.x, x))
        self.type_goal = type_goal

    def calculate(self, **kwargs) -> Tuple[List[float], List[float]]:
        u = [0]
        xi = [random.normalvariate(0, self.sigma) for _ in range(len(self.M) + 1)]
        
        # first 
        if self.type_goal == "x1c_multi":
            psi = [self.x[0, 0] - kwargs['x1c']]
            f1 = u[0] * self.x[0, 0] - self.beta1 * self.x[0, 0] * self.x[0, 1]
            f2 = -self.alpha2 * self.x[0, 1] + self.beta2 * self.x[0, 0] * self.x[0, 1]
            x1 = self.x[0, 0] + self.h * f1
            x2 = self.x[0, 1] + self.h * f2 + xi[1] + self.c * xi[0]
            self.x = np.vstack((self.x, [x1, x2]))
        elif self.type_goal == "x1c_add":
            psi = [self.x[0, 0] - kwargs['x1c']]
            f1 = self.alpha1 * self.x[0, 0] - self.beta1 * self.x[0, 0] * self.x[0, 1] + u[0]
            f2 = -self.alpha2 * self.x[0, 1] + self.beta2 * self.x[0, 0] * self.x[0, 1]
            x1 = self.x[0, 0] + self.h * f1
            x2 = self.x[0, 1] + self.h * f2 + xi[1] + self.c * xi[0]
            self.x = np.vstack((self.x, [x1, x2]))
        elif self.type_goal == "rho_d":
            psi = [self.x[0, 0] + kwargs['rho'] * self.x[0, 1] - kwargs['d']]
            f1 = u[0] * self.x[0, 0] - self.beta1 * self.x[0, 0] * self.x[0, 1]
            f2 = -self.alpha2 * self.x[0, 1] + self.beta2 * self.x[0, 0] * self.x[0, 1]
            x1 = self.x[0, 0] + self.h * f1
            x2 = self.x[0, 1] + self.h * f2 + xi[1] + self.c * xi[0]
            self.x = np.vstack((self.x, [x1, x2]))

        for i in range(1, len(self.M)-1):
            if self.type_goal == "x1c_multi":
                psi.append(self.x[i, 0] - kwargs['x1c'])
                u.append((-psi[i] * (1 + self.T) - self.c * (psi[i] + self.T * psi[i - 1])) / (self.h * self.x[i, 0]) + self.beta1 * self.x[i, 1])
                f1 = u[i] * self.x[i, 0] - self.beta1 * self.x[i, 0] * self.x[i, 1]
                f2 = -self.alpha2 * self.x[i, 1] + self.beta2 * self.x[i, 0] * self.x[i, 1]

            elif self.type_goal == "x1c_add":
                psi.append(self.x[i, 0] - kwargs['x1c'])
                u.append((-psi[i] * (1 + self.T) - self.c * (psi[i] + self.T * psi[i - 1])) / (self.h * self.x[i, 0]) + self.alpha1 * self.x[i, 0] + self.beta1 * self.x[i, 0] * self.x[i, 1])
                f1 = self.alpha1 * self.x[i, 0] - self.beta1 * self.x[i, 0] * self.x[i, 1] + u[i]
                f2 = -self.alpha2 * self.x[i, 1] + self.beta2 * self.x[i, 0] * self.x[i, 1]

            elif self.type_goal == "rho_d":
                psi.append(self.x[i, 0] + kwargs['rho'] * self.x[i, 1] - kwargs['d'])
                u.append((-self.x[i, 0] - self.c * (psi[i] + self.T * psi[i - 1]) - kwargs['rho'] * (self.x[i, 1] + self.h * f2) + kwargs['d'] - self.T * psi[i]) / (self.h * self.x[i, 0]) + self.beta1 * self.x[i, 1])
                f1 = u[i] * self.x[i, 0] - self.beta1 * self.x[i, 0] * self.x[i, 1]
                f2 = -self.alpha2 * self.x[i, 1] + self.beta2 * self.x[i, 0] * self.x[i, 1]

            x1 = self.x[i, 0] + self.h * f1  + xi[i + 1] + self.c * xi[i]
            x2 = self.x[i, 1] + self.h * f2
            self.x = np.vstack((self.x, [x1, x2]))
            

        return self.x, u

    def plot(self, x: List[List[float]], u: List[float], **kwargs) -> NoReturn:
        plt.figure(figsize=(10, 7))
        plt.grid(visible=True)
        plt.plot(self.M, x[:, 0], 'g', label=r'$x_{1}$')
        plt.plot(self.M, x[:, 1], 'b', label=r'$x_{2}$')
        if self.type_goal == "x1c_multi" or self.type_goal == "x1c_add":
            plt.plot(self.M, kwargs['x1c'] * np.ones(len(self.M)), 'k--', label=r"$x_{1}^{*}$")
        sns.set_style('whitegrid')
        plt.xlim(0, self.N)
        plt.ylim(0)
        plt.legend(loc="best")
        plt.xlabel('Время, дни')
        plt.ylabel('Популяция, ед/л')

        if "save_fig" in kwargs:
            plt.savefig(f"{kwargs['name_fig1']}.png")
            plt.savefig(f"{kwargs['name_fig1']}.svg")
            plt.savefig(f"{kwargs['name_fig1']}.eps")

        plt.figure(figsize=(10, 7))
        plt.plot(self.M[:-1], u, 'k', label=r'$u(t)$')
        sns.set_style('whitegrid')
        plt.xlim(0, self.N)
        plt.legend(loc="best")
        plt.xlabel('Время, дни')
        plt.ylabel('Управление')
        
        if "save_fig" in kwargs:
            plt.savefig(f"{kwargs['name_fig2']}.png")
            plt.savefig(f"{kwargs['name_fig2']}.svg")
            plt.savefig(f"{kwargs['name_fig2']}.eps")

        plt.figure(figsize=(10, 7))
        plt.plot(self.x[0, 0], self.x[0, 1], 'bo', zorder=3, label="Начальное состояние")
        plt.plot(self.x[-1, 0], self.x[-1, 1], 'ro', zorder=2, label='Конечное состояние')
        plt.plot(self.x[:, 0], self.x[:, 1], 'r-', zorder=1, linewidth=3)
        plt.legend(loc="upper right")
        plt.xlabel(r'$x_{1}$')
        plt.ylabel(r'$x_{2}$')
        # plt.xlim(left=0)
        # plt.ylim(bottom=0)

        if "save_fig" in kwargs:
            plt.savefig(f"{kwargs['name_fig3']}.png")
            plt.savefig(f"{kwargs['name_fig3']}.svg")
            plt.savefig(f"{kwargs['name_fig3']}.eps")

        plt.show()