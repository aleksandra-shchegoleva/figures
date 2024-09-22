from models.models import AbstractModel, AbstractFactory
import matplotlib.pyplot as plt
from typing import List, Tuple, NoReturn
import numpy as np
import seaborn as sns

class TPPFactory(AbstractFactory):
    """
    Factory class for continuous systems 
    """
    def create_model(self):
        return TPP()

class TPP(AbstractModel):
    """
    Class for TPP-model
    """
    def set_parameters(
        self,
        r: float,
        K: float,
        alpha: float,
        gamma: float,
        beta: float,
        mu: float,
        theta: float,
        T: float,
        N: float,
        h: float,
        x: List[float],
        type_goal: str
        ) -> NoReturn:
        
        self.r = r
        self.K = K
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.mu = mu
        self.theta = theta
        self.T = T
        self.M = np.arange(0, N, h)
        self.N = N
        self.h = h
        self.x = np.empty((0, 2), dtype=np.float32)
        self.x = np.vstack((self.x, x))
        self.type_goal = type_goal

    def calculate(self, **kwargs) -> Tuple[List[float], List[float]]:
        u = [0]
        for i in range(len(self.M)-1):
            if self.type_goal == "x1c":
                f1 = self.r * self.x[i, 0] * (1 - self.x[i, 0] / self.K) - self.alpha * TPP.f(self.x[i, 0], self.gamma) * self.x[i, 1] + u[i]
                f2 = self.beta * TPP.f(self.x[i, 0], self.gamma) * self.x[i, 1] - self.mu * self.x[i, 1] - self.theta * TPP.f(self.x[i, 0], self.gamma) * self.x[i, 1]
                psi = self.x[i, 0] - kwargs['x1c']
                u.append(-psi / self.T - self.r * self.x[i, 0] * (1 - self.x[i, 0] / self.K) + self.alpha * TPP.f(self.x[i, 0], self.gamma) * self.x[i, 1])

            elif self.type_goal == "rho_d":
                f1 = self.r * self.x[i, 0] * (1 - self.x[i, 0] / self.K) - self.alpha * TPP.f(self.x[i, 0], self.gamma) * self.x[i, 1] + u[i]
                f2 = self.beta * TPP.f(self.x[i, 0], self.gamma) * self.x[i, 1] - self.mu * self.x[i, 1] - self.theta * TPP.f(self.x[i, 0], self.gamma) * self.x[i, 1]
                psi = self.x[i, 0] + kwargs['rho'] * self.x[i, 1] - kwargs['d']
                u.append(-psi / self.T - kwargs['rho']*f2 - self.r * self.x[i, 0] * (1 - self.x[i, 0] / self.K) + self.alpha * TPP.f(self.x[i, 0], self.gamma) * self.x[i, 1])

            x1 = self.x[i, 0] + self.h * f1
            x2 = self.x[i, 1] + self.h * f2
            self.x = np.vstack((self.x, [x1, x2]))

        return self.x, u
    
    @staticmethod
    def f(x, gamma):
        return x / (gamma + x)

    def plot(self, x: List[List[float]], u: List[float], **kwargs) -> NoReturn:
        plt.figure(figsize=(10, 7))
        plt.plot(self.M, x[:, 0], 'g', label=r'$x_{1}$')
        plt.plot(self.M, x[:, 1], 'b', label=r'$x_{2}$')
        if self.type_goal == "x1c":
            plt.plot(self.M, kwargs['x1c'] * np.ones(len(self.M)), 'k--', label=r"$x_{1*}$")
        if self.type_goal == "rho_d":
            x1s = self.gamma * self.mu / (self.beta - self.mu - self.theta)
            x2s = (kwargs['d'] * self.mu + self.gamma * self.mu + kwargs['d'] * self.theta - self.beta * kwargs['d']) / (self.mu * kwargs['rho'] - self.beta * kwargs['rho'] + kwargs['rho'] * self.theta)
            plt.plot(self.M, x1s * np.ones(len(self.M)), 'k--', label=r"$x_{1*}$")
            plt.plot(self.M, x2s * np.ones(len(self.M)), 'k--', label=r"$x_{2*}$")
        sns.set_style('whitegrid')
        plt.xlim(0, self.N)
        plt.legend(loc="upper right")
        plt.xlabel('Время, дни')
        plt.ylabel('Популяция, ед/л')

        if "save_fig" in kwargs:
            plt.savefig(f"{kwargs['name_fig1']}.png")
            plt.savefig(f"{kwargs['name_fig1']}.svg")
            plt.savefig(f"{kwargs['name_fig1']}.eps")

        plt.figure(figsize=(10, 7))
        plt.plot(self.M, u, 'g', label=r'$u(t)$')
        sns.set_style('whitegrid')
        plt.xlim(0, self.N)
        plt.legend(loc="upper right")
        plt.xlabel('Время, дни')
        plt.ylabel('Управление')
        
        if "save_fig" in kwargs:
            plt.savefig(f"{kwargs['name_fig2']}.png")
            plt.savefig(f"{kwargs['name_fig2']}.svg")
            plt.savefig(f"{kwargs['name_fig2']}.eps")

        plt.show()