from models.models import AbstractModel, AbstractFactory
import matplotlib.pyplot as plt
from typing import List, Tuple, NoReturn
import numpy as np
import seaborn as sns
import random

random.seed(20)

class PreyPredatorBigPhaseNASFactory(AbstractFactory):
    """
    Factory class for discret systems 
    """
    def create_model(self):
        return PreyPredatorBigPhaseNAS()

class PreyPredatorBigPhaseNAS(AbstractModel):
    """
    Class for model
    """
    def set_parameters(
        self,
        beta1: float,
        alpha1: float,
        alpha2: float,
        beta2: float,
        c: float,
        sigma: float,
        T1: float,
        T2: float,
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
        self.T1 = T1
        self.T2 = T2
        self.M = np.arange(0, N, h)
        self.N = N
        self.h = h
        self.x = np.empty((0, 3), dtype=np.float16)
        self.x = np.vstack((self.x, x))
        self.type_goal = type_goal

    def calculate(self, **kwargs) -> Tuple[List[float], List[float]]:
        u = [0]
        xi = [random.normalvariate(0, self.sigma) for _ in range(len(self.M) + 1)]
        
        # first 
        psi = [self.x[0, 0] - kwargs['x1c']]
        f1 = self.x[0, 2] * self.x[0, 0] - self.beta1 * self.x[0, 0] * self.x[0, 1]
        f2 = -self.alpha2 * self.x[0, 1] + self.beta2 * self.x[0, 0] * self.x[0, 1]
        x1 = self.x[0, 0] + self.h * f1
        x2 = self.x[0, 1] + self.h * f2
        x3 = self.x[0, 2] + self.h * u[0] + xi[1] + self.c * xi[0]
        self.x = np.vstack((self.x, [x1, x2, x3]))

        phi = [- (psi[0] * (1 + self.T2)) / (self.h * self.x[0, 0]) + self.beta1 * self.x[0, 1]]
        psi.append(self.x[1, 0] - kwargs['x1c'])
        phi.append(- (psi[1] * (1 + self.T2)) / (self.h * self.x[1, 0]) + self.beta1 * self.x[1, 1])
        psi1 = [self.x[0, 2] - phi[0]]

        for i in range(1, len(self.M)-1):
            f1 = self.x[i, 2] * self.x[i, 0] - self.beta1 * self.x[i, 0] * self.x[i, 1]
            f2 = -self.alpha2 * self.x[i, 1] + self.beta2 * self.x[i, 0] * self.x[i, 1]
            x1 = self.x[i, 0] + self.h * f1
            x2 = self.x[i, 1] + self.h * f2

            psi.append(x1 - kwargs['x1c'])
            psi1.append(self.x[i, 2] - phi[i])
            phi.append(- (psi[i + 1] * (1 + self.T2)) / (self.h *x1) + self.beta1 * x2)
            u.append((phi[i + 1] - self.T1 * psi1[i] - self.x[i, 2] - self.c * (psi1[i] + self.T1 * psi1[i - 1])) / self.h)

            x3 = self.x[i, 2] + self.h * u[i] + xi[i + 1] + self.c * xi[i]
            self.x = np.vstack((self.x, [x1, x2, x3]))
            

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

        if "save_fig" in kwargs:
            plt.savefig(f"{kwargs['name_fig3']}а.png")
            plt.savefig(f"{kwargs['name_fig3']}а.svg")
            plt.savefig(f"{kwargs['name_fig3']}а.eps")

        plt.figure(figsize=(10, 7))
        plt.plot(self.x[0, 0], self.x[0, 2], 'bo', zorder=3, label="Начальное состояние")
        plt.plot(self.x[-1, 0], self.x[-1, 2], 'ro', zorder=2, label='Конечное состояние')
        plt.plot(self.x[:, 0], self.x[:, 2], 'r-', zorder=1, linewidth=3)
        plt.legend(loc="lower left")
        plt.xlabel(r'$x_{1}$')
        plt.ylabel(r'$x_{3}$')

        if "save_fig" in kwargs:
            plt.savefig(f"{kwargs['name_fig3']}б.png")
            plt.savefig(f"{kwargs['name_fig3']}б.svg")
            plt.savefig(f"{kwargs['name_fig3']}б.eps")

        plt.figure(figsize=(10, 7))
        plt.plot(self.x[0, 1], self.x[0, 2], 'bo', zorder=3, label="Начальное состояние")
        plt.plot(self.x[-1, 1], self.x[-1, 2], 'ro', zorder=2, label='Конечное состояние')
        plt.plot(self.x[:, 1], self.x[:, 2], 'r-', zorder=1, linewidth=3)
        plt.legend(loc="lower right")
        plt.xlabel(r'$x_{2}$')
        plt.ylabel(r'$x_{3}$')

        if "save_fig" in kwargs:
            plt.savefig(f"{kwargs['name_fig3']}в.png")
            plt.savefig(f"{kwargs['name_fig3']}в.svg")
            plt.savefig(f"{kwargs['name_fig3']}в.eps")

        plt.show()