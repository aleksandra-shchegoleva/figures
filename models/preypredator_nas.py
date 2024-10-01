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
        print(xi)
        
        # first step
        psi = [self.x[0, 0] - kwargs['x1c']]

        for i in range(len(self.M)-1):
            if self.type_goal == "x1c":
                f1 = u[i] * self.x[i, 0] - self.beta1 * self.x[i, 0] * self.x[i, 1]
                f2 = -self.alpha2 * self.x[i, 1] + self.beta2 * self.x[i, 0] * self.x[i, 1]
                x1 = self.x[i, 0] + self.h * f1  + xi[i + 1] + self.c * xi[i]
                # x1 = self.x[i, 0] + self.h * f1 
                x2 = self.x[i, 1] + self.h * f2
                self.x = np.vstack((self.x, [x1, x2]))

                psi.append(self.x[i, 0] - kwargs['x1c'])
                u.append((-psi[i] * (1 + self.T) - self.c * (psi[i] + self.T * psi[i - 1])) / (self.h * self.x[i, 0]) + self.beta1 * self.x[i, 1])

            elif self.type_goal == "rho_d":
                f1 = self.r * self.x[i, 0] * (1 - self.x[i, 0] / self.K) - self.alpha * TPP.f(self.x[i, 0], self.gamma) * self.x[i, 1] + u[i] 
                f2 = self.beta * TPP.f(self.x[i, 0], self.gamma) * self.x[i, 1] - self.mu * self.x[i, 1] - self.theta * TPP.f(self.x[i, 0], self.gamma) * self.x[i, 1]
                psi = self.x[i, 0] + kwargs['rho'] * self.x[i, 1] - kwargs['d']
                u.append(-psi / self.T - kwargs['rho']*f2 - self.r * self.x[i, 0] * (1 - self.x[i, 0] / self.K) + self.alpha * TPP.f(self.x[i, 0], self.gamma) * self.x[i, 1])

            # x1 = self.x[i, 0] + self.h * f1 + xi[i + 1] + self.c * xi[i]
            # x2 = self.x[i, 1] + self.h * f2
            

        return self.x, u

    def plot(self, x: List[List[float]], u: List[float], **kwargs) -> NoReturn:
        plt.figure(figsize=(10, 7))
        plt.plot(self.M, x[:, 0], 'g', label=r'$x_{1}$')
        plt.plot(self.M, x[:, 1], 'b', label=r'$x_{2}$')
        if self.type_goal == "x1c":
            plt.plot(self.M, kwargs['x1c'] * np.ones(len(self.M)), 'k--', label=r"$x_{1}^{*}$")
        if self.type_goal == "rho_d":
            x1s = self.gamma * self.mu / (self.beta - self.mu - self.theta)
            x2s = (kwargs['d'] * self.mu + self.gamma * self.mu + kwargs['d'] * self.theta - self.beta * kwargs['d']) / (self.mu * kwargs['rho'] - self.beta * kwargs['rho'] + kwargs['rho'] * self.theta)
            plt.plot(self.M, x1s * np.ones(len(self.M)), 'k-.', label=r"$x_{1*}$")
            plt.plot(self.M, x2s * np.ones(len(self.M)), 'k--', label=r"$x_{2*}$")
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
        plt.plot(self.M, u, 'k', label=r'$u(t)$')
        sns.set_style('whitegrid')
        plt.xlim(0, self.N)
        plt.legend(loc="best")
        plt.xlabel('Время, дни')
        plt.ylabel('Управление')
        
        if "save_fig" in kwargs:
            plt.savefig(f"{kwargs['name_fig2']}.png")
            plt.savefig(f"{kwargs['name_fig2']}.svg")
            plt.savefig(f"{kwargs['name_fig2']}.eps")

        plt.show()