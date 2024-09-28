from models.models import AbstractModel, AbstractFactory
import matplotlib.pyplot as plt
from typing import List, Tuple, NoReturn
import numpy as np
import seaborn as sns

class PreyPredatorBigPhaseFactory(AbstractFactory):
    """
    Factory class for continuous systems 
    """
    def create_model(self):
        return PreyPredatorBigPhaseModel()

class PreyPredatorBigPhaseModel(AbstractModel):
    """
    Class for base prey-predator model with big phase
    """
    def set_parameters(
        self,
        alpha1: float,
        alpha2: float,
        beta1: float,
        beta2: float,
        T1: float,
        T2: float,
        N: float,
        h: float,
        x: List[float],
        x1c: float
        ) -> NoReturn:
        
        self.alpha1 = [alpha1]
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2
        self.T1 = T1
        self.T2 = T2
        self.M = np.arange(0, N, h)
        self.N = N
        self.h = h
        self.x = np.empty((0, 2), dtype=np.float32)
        self.x = np.vstack((self.x, x))
        self.x1c = x1c

    def calculate(self, **kwargs) -> Tuple[List[float], List[float]]:
        u = [0]
        for i in range(len(self.M)-1):
            f1 = self.alpha1[i] * self.x[i, 0] - self.beta1 * self.x[i, 0] * self.x[i, 1]
            f2 = -self.alpha2 * self.x[i, 1] + self.beta2 * self.x[i, 0] * self.x[i, 1]
            dphi_dt = - self.x1c * f1 / (self.T2 * self.x[i, 0] ** 2) + self.beta1 * f2
            phi = self.beta1 * self.x[i, 1] - 1 / self.T2 + self.x1c / (self.T2 * self.x[i, 0])
            psi1 = self.alpha1[i] - phi
            u.append(-psi1 / self.T1 + dphi_dt)

            x1 = self.x[i, 0] + self.h * f1
            x2 = self.x[i, 1] + self.h * f2
            x3 = self.alpha1[i] + self.h * u[i]
            self.x = np.vstack((self.x, [x1, x2]))
            self.alpha1.append(x3)

        return self.x, self.alpha1

    def plot(self, x: List[List[float]], alpha1: List[float], **kwargs) -> NoReturn:
        plt.figure(figsize=(10, 7))
        plt.grid()
        plt.plot(self.M, x[:, 0], 'g', label=r'$x_{1}$')
        plt.plot(self.M, x[:, 1], 'b', label=r'$x_{2}$')
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
        plt.plot(self.M, alpha1, 'k', label=r'$u(t)$')
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
        plt.plot(self.x[0, 0], self.x[0, 1], 'bo', label="Начальное состояние")
        plt.plot(self.x[-1, 0], self.x[-1, 1], 'ro', label='Конечное состояние')
        plt.plot(self.x[:, 0], self.x[:, 1], 'r-', linewidth=3)
        plt.legend(loc="best")
        plt.xlabel(r'$x_{1}$')
        plt.ylabel(r'$x_{2}$')
        plt.xlim(left=min(self.x[:, 0]))
        plt.ylim(bottom=0)

        if "save_fig" in kwargs:
            plt.savefig(f"{kwargs['name_fig3']}а.png")
            plt.savefig(f"{kwargs['name_fig3']}а.svg")
            plt.savefig(f"{kwargs['name_fig3']}а.eps")

        plt.figure(figsize=(10, 7))
        plt.plot(self.x[0, 0], alpha1[0], 'bo', label="Начальное состояние")
        plt.plot(self.x[-1, 0], alpha1[-1], 'ro', label='Конечное состояние')
        plt.plot(self.x[:, 0], alpha1, 'r-', linewidth=3)
        plt.legend(loc="best")
        plt.xlabel(r'$x_{1}$')
        plt.ylabel(r'$x_{3}$')
        plt.xlim(left=min(self.x[:, 0]))
        plt.ylim(bottom=0)

        if "save_fig" in kwargs:
            plt.savefig(f"{kwargs['name_fig3']}б.png")
            plt.savefig(f"{kwargs['name_fig3']}б.svg")
            plt.savefig(f"{kwargs['name_fig3']}б.eps")

        plt.figure(figsize=(10, 7))
        plt.plot(self.x[0, 1], alpha1[0], 'bo', label="Начальное состояние")
        plt.plot(self.x[-1, 1], alpha1[-1], 'ro', label='Конечное состояние')
        plt.plot(self.x[:, 1], alpha1, 'r-', linewidth=3)
        plt.legend(loc="best")
        plt.xlabel(r'$x_{2}$')
        plt.ylabel(r'$x_{3}$')
        plt.xlim(left=min(self.x[:, 1]))
        plt.ylim(bottom=0)

        if "save_fig" in kwargs:
            plt.savefig(f"{kwargs['name_fig3']}в.png")
            plt.savefig(f"{kwargs['name_fig3']}в.svg")
            plt.savefig(f"{kwargs['name_fig3']}в.eps")

        plt.show()