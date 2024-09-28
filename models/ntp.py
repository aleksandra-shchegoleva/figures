from models.models import AbstractModel, AbstractFactory
import matplotlib.pyplot as plt
from typing import List, Tuple, NoReturn
import numpy as np
import seaborn as sns

class NTPFactory(AbstractFactory):
    """
    Factory class for continuous systems 
    """
    def create_model(self):
        return NTPModel()

class NTPModel(AbstractModel):
    """
    Class for NTP model
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
        self.T = T
        self.M = np.arange(0, N, h)
        self.N = N
        self.h = h
        self.x = np.empty((0, 3), dtype=np.float32)
        self.x = np.vstack((self.x, x))
        self.type_goal = type_goal

    def calculate(self, **kwargs) -> Tuple[List[float], List[float]]:
        u = [0]

        for i in range(len(self.M)-1):
            if self.type_goal == "x2c_add":
                f1 = self.r1 * self.x[i, 0] * (1 - (self.x[i, 0] + self.alpha1 * self.x[i, 1]) / self.K1) - (self.w1 * self.x[i, 0] * self.x[i, 2]) / (self.d1 + self.x[i, 0])
                f2 = self.r2 * self.x[i, 1] * (1 - (self.x[i, 1] + self.alpha2 * self.x[i, 0]) / self.K2) - (self.w2 * self.x[i, 1] * self.x[i, 2]) / (self.d2 + self.b1 * self.x[i, 1] ** 2) + u[i]
                f3 = (self.gamma1 * self.x[i, 0] * self.x[i, 2]) / (self.d1 + self.x[i, 0]) - (self.gamma2 * self.x[i, 1] * self.x[i, 2]) / (self.d2 + self.b1 * self.x[i, 1] ** 2) - self.m * self.x[i, 2] - self.m1 * self.x[i, 2] ** 2
                psi = self.x[i, 1] - kwargs['x2c']
                u.append(-psi / self.T - self.r2 * self.x[i, 1] * (1 - (self.x[i, 1] + self.alpha2 * self.x[i, 0]) / self.K2) + (self.w2 * self.x[i, 1] * self.x[i, 2]) / (self.d2 + self.b1 * self.x[i, 1] ** 2))

            elif self.type_goal == "rho_d":
                f1 = self.r1 * self.x[i, 0] * (1 - (self.x[i, 0] + self.alpha1 * self.x[i, 1]) / self.K1) - (self.w1 * self.x[i, 0] * self.x[i, 2]) / (self.d1 + self.x[i, 0]) + u[i]
                f2 = self.r2 * self.x[i, 1] * (1 - (self.x[i, 1] + self.alpha2 * self.x[i, 0]) / self.K2) - (self.w2 * self.x[i, 1] * self.x[i, 2]) / (self.d2 + self.b1 * self.x[i, 1] ** 2)
                f3 = (self.gamma1 * self.x[i, 0] * self.x[i, 2]) / (self.d1 + self.x[i, 0]) - (self.gamma2 * self.x[i, 1] * self.x[i, 2]) / (self.d2 + self.b1 * self.x[i, 1] ** 2) - self.m * self.x[i, 2] - self.m1 * self.x[i, 2] ** 2
                psi = self.x[i, 0] - kwargs['rho'] * self.x[i, 1] + kwargs['d']
                u.append(-psi / self.T - self.r1 * self.x[i, 0] * (1 - (self.x[i, 0] + self.alpha1 * self.x[i, 1]) / self.K1) + (self.w1 * self.x[i, 0] * self.x[i, 2]) / (self.d1 + self.x[i, 0]) + kwargs['rho'] * f2)
            elif self.type_goal == "x2c_multi":
                f1 = self.r1 * self.x[i, 0] * (1 - (self.x[i, 0] + self.alpha1 * self.x[i, 1]) / self.K1) - (self.w1 * self.x[i, 0] * self.x[i, 2]) / (self.d1 + self.x[i, 0])
                f2 = u[i] * self.x[i, 1] * (1 - (self.x[i, 1] + self.alpha2 * self.x[i, 0]) / self.K2) - (self.w2 * self.x[i, 1] * self.x[i, 2]) / (self.d2 + self.b1 * self.x[i, 1] ** 2)
                f3 = (self.gamma1 * self.x[i, 0] * self.x[i, 2]) / (self.d1 + self.x[i, 0]) - (self.gamma2 * self.x[i, 1] * self.x[i, 2]) / (self.d2 + self.b1 * self.x[i, 1] ** 2) - self.m * self.x[i, 2] - self.m1 * self.x[i, 2] ** 2
                psi = self.x[i, 1] - kwargs['x2c']
                u.append(self.x[i, 1] ** (-1) * (1 - (self.x[i, 1] + self.alpha2 * self.x[i, 0]) / self.K2) ** (-1) * (-psi / self.T + self.w2 * self.x[i, 1] * self.x[i, 2] / (self.d2 + self.b1 * self.x[i, 1] ** 2)))

            x1 = self.x[i, 0] + self.h * f1
            x2 = self.x[i, 1] + self.h * f2
            x3 = self.x[i, 2] + self.h * f3
            self.x = np.vstack((self.x, [x1, x2, x3]))

        return self.x, u

    def plot(self, x: List[List[float]], u: List[float], **kwargs) -> NoReturn:
        plt.figure(figsize=(10, 7))
        plt.grid(visible=True)
        plt.plot(self.M, x[:, 0], 'g', label=r'$P_{1}$')
        plt.plot(self.M, x[:, 1], 'b', label=r'$P_{2}$')
        plt.plot(self.M, x[:, 2], 'r', label=r'$Z$')
        if "x2c" in kwargs:
            plt.plot(self.M, kwargs['x2c'] * np.ones(len(self.M)), 'k--', label=r"$P_{2}^{*}$")
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