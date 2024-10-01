from models.models import AbstractModel, AbstractFactory
import matplotlib.pyplot as plt
from typing import List, Tuple, NoReturn
import numpy as np
import seaborn as sns
import random

class NTP_NAS_Factory(AbstractFactory):
    """
    Factory class for NTP model with NAS control
    """
    def create_model(self):
        return NTP_NAS()

class NTP_NAS(AbstractModel):
    """
    Class for model
    """
    def set_parameters(
        self,
        r1: float,
        r2: float,
        K1: float,
        K2: float,
        alpha1: float,
        alpha2: float,
        w1: float,
        w2: float,
        d1: float,
        d2: float,
        b1: float,
        gamma1: float,
        gamma2: float,
        m: float,
        m1: float,
        c: float,
        sigma: float,
        T: float,
        N: float,
        h: float,
        x: List[float],
        type_goal: str
        ) -> NoReturn:
        
        self.r1 = r1
        self.r2 = r2
        self.K1 = K1
        self.K2 = K2
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.w1 = w1
        self.w2 = w2
        self.d1 = d1
        self.d2 = d2
        self.b1 = b1
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.m = m
        self.m1 = m1
        self.c = c
        self.sigma = sigma
        self.T = T
        self.M = np.arange(0, N, h)
        self.N = N
        self.h = h
        self.x = np.empty((0, 3), dtype=np.float32)
        self.x = np.vstack((self.x, x))
        self.type_goal = type_goal

    def calculate(self, **kwargs) -> Tuple[List[float], List[float]]:
        u = [0]
        psi = []
        xi = [random.normalvariate(0, self.sigma) for _ in range(len(self.M) + 1)]

        # first step
        f1 = self.r1 * self.x[0, 0] * (1 - (self.x[0, 0] + self.alpha1 * self.x[0, 1]) / self.K1) - (self.w1 * self.x[0, 0] * self.x[0, 2]) / (self.d1 + self.x[0, 0])
        f2 = self.r2 * self.x[0, 1] * (1 - (self.x[0, 1] + self.alpha2 * self.x[0, 0]) / self.K2) - (self.w2 * self.x[0, 1] * self.x[0, 2]) / (
                    self.d2 + self.b1 * self.x[0, 1] ** 2)
        f3 = (self.gamma1 * self.x[0, 0] * self.x[0, 2]) / (self.d1 + self.x[0, 0]) - (self.gamma2 * self.x[0, 1] * self.x[0, 2]) / (
                    self.d2 + self.b1 * self.x[0, 1] ** 2) - self.m * self.x[0, 2] - self.m1 * self.x[0, 2] ** 2
        psi.append(self.x[0, 1] - kwargs['x2c'])
        x1 = self.x[0, 0] + self.h * f1
        x2 = self.x[0, 1] + self.h * f2 + self.h * u[0] + xi[1] + self.c * xi[0]
        x3 = self.x[0, 2] + self.h * f3
        self.x = np.vstack((self.x, [x1, x2, x3]))

        for i in range(1, len(self.M)-1):
            f1 = self.r1 * self.x[i, 0] * (1 - (self.x[i, 0] + self.alpha1 * self.x[i, 1]) / self.K1) - (self.w1 * self.x[i, 0] * self.x[i, 2]) / (self.d1 + self.x[i, 0])
            f2 = self.r2 * self.x[i, 1] * (1 - (self.x[i, 1] + self.alpha2 * self.x[i, 0]) / self.K2) - (self.w2 * self.x[i, 1] * self.x[i, 2]) / (
                        self.d2 + self.b1 * self.x[i, 1] ** 2)
            f3 = (self.gamma1 * self.x[i, 0] * self.x[i, 2]) / (self.d1 + self.x[i, 0]) - (self.gamma2 * self.x[i, 1] * self.x[i, 2]) / (
                        self.d2 + self.b1 * self.x[i, 1] ** 2) - self.m * self.x[i, 2] - self.m1 * self.x[i, 2] ** 2
            psi.append(self.x[i, 1] - kwargs['x2c'])
            x1 = self.x[i, 0] + self.h * f1
            u.append((-self.x[i, 1] - self.c * (psi[i] + self.T * psi[i - 1]) + kwargs['x2c'] - self.T * psi[i]) / self.h - f2)
            x2 = self.x[i, 1] + self.h * f2 + self.h * u[i] + xi[i + 1] + self.c * xi[i]
            x3 = self.x[i, 2] + self.h * f3
            self.x = np.vstack((self.x, [x1, x2, x3]))
            

        return self.x, u

    def plot(self, x: List[List[float]], u: List[float], **kwargs) -> NoReturn:
        plt.figure(figsize=(10, 7))
        plt.grid(visible=True)
        plt.plot(self.M, x[:, 0], 'g', label=r'$P_{1}$')
        plt.plot(self.M, x[:, 1], 'b', label=r'$P_{2}$')
        plt.plot(self.M, x[:, 2], 'r', label=r'$Z$')
        if self.type_goal == "x2c":
            plt.plot(self.M, kwargs['x2c'] * np.ones(len(self.M)), 'k--', label=r"$P_{2}^{*}$")
        if self.type_goal == "rho_d":
            x1s = self.gamma * self.mu / (self.beta - self.mu - self.theta)
            x2s = (kwargs['d'] * self.mu + self.gamma * self.mu + kwargs['d'] * self.theta - self.beta * kwargs['d']) / (self.mu * kwargs['rho'] - self.beta * kwargs['rho'] + kwargs['rho'] * self.theta)
            plt.plot(self.M, x1s * np.ones(len(self.M)), 'k-.', label=r"$x_{1*}$")
            plt.plot(self.M, x2s * np.ones(len(self.M)), 'k--', label=r"$x_{2*}$")
        sns.set_style('whitegrid')
        plt.xlim(0, self.N)
        print(np.min(self.x[:, 1]))
        if np.min(self.x[:, 1]) < 0:
            plt.ylim(round(np.min(self.x[:, 1])))
        else:
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

        plt.show()