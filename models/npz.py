from models.models import AbstractModel, AbstractFactory
import matplotlib.pyplot as plt
from typing import List, Tuple, NoReturn
import numpy as np
import seaborn as sns

class NPZFactory(AbstractFactory):
    """
    Factory class for continuous systems 
    """
    def create_model(self):
        return NPZModel()

class NPZModel(AbstractModel):
    """
    Class for NTP model
    """
    def set_parameters(
        self,
        a: float,
        b: float,
        c: float,
        d: float,
        T1: float,
        T2: float,
        N: float,
        h: float,
        x: List[float],
        type_goal: str
        ) -> NoReturn:
        
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.T1 = T1
        self.T2 = T2
        self.M = np.arange(0, N, h)
        self.N = N
        self.h = h
        self.x = np.empty((0, 3), dtype=np.float32)
        self.x = np.vstack((self.x, x))
        self.type_goal = type_goal

    def calculate(self, **kwargs) -> Tuple[List[float], List[float]]:
        u = [0]

        for i in range(len(self.M)-1):
            if self.type_goal == "x2c":
                f1 = self.a * self.x[i, 1] + self.b * self.x[i, 2] - self.c * self.x[i, 0] * self.x[i, 1] + u[i]
                f2 = self.c * self.x[i, 0] * self.x[i, 1] - self.d * self.x[i, 1] * self.x[i, 2] - self.a * self.x[i, 1]
                f3 = self.d * self.x[i, 1] * self.x[i, 2] - self.b * self.x[i, 2]
                psi = self.x[i, 1] - kwargs['x2c']
                dphi_dt = ((-f2 / self.T2 + self.d * (f2 * self.x[i, 2] + self.x[i, 1] * f3) + self.a * f2) * self.x[i, 1] -(-psi / self.T2 + self.d * self.x[i, 1] * self.x[i, 2] + self.a * self.x[i, 1]) * f2) * self.c ** (-1) * self.x[i, 1] ** (-2)
                phi = (-psi / self.T2 + self.d * self.x[i, 1] * self.x[i, 2] + self.a * self.x[i, 1]) * self.c ** (-1) * self.x[i, 1] ** (-1)
                psi1 = self.x[i, 0] - phi
                u.append(-psi1 / self.T1 - self.a  * self.x[i, 1] - self.b * self.x[i, 2] + self.c * self.x[i, 0] * self.x[i, 1] + dphi_dt)

            elif self.type_goal == "rho_d":
                f1 = self.a * self.x[i, 1] + self.b * self.x[i, 2] - self.c * self.x[i, 0] * self.x[i, 1] + u[i]
                f2 = self.c * self.x[i, 0] * self.x[i, 1] - self.d * self.x[i, 1] * self.x[i, 2] - self.a * self.x[i, 1]
                f3 = self.d * self.x[i, 1] * self.x[i, 2] - self.b * self.x[i, 2]
                psi = self.x[i, 1] + kwargs['rho'] * self.x[i, 2] - kwargs['q']
                dphi_dt = (-(f2 + kwargs['rho'] * f3) / self.T2 + self.d * (f2 * self.x[i, 2] + self.x[i, 1] * f3) + self.a * f2 - kwargs['rho'] * (self.d * (f2 * self.x[i, 2] + self.x[i, 1] * f3) - self.b * f3)) * self.x[i, 1] / (self.c * self.x[i, 1] ** 2) - \
                    (-psi / self.T2 + self.d * self.x[i, 1] * self.x[i, 2] + self.a * self.x[i, 1] - kwargs['rho'] * (self.d * self.x[i, 1] * self.x[i, 2] - self.b * self.x[i, 2])) * f2 / (self.c * self.x[i, 1] ** 2)
                phi = (-psi / self.T2 + self.d * self.x[i, 1] * self.x[i, 2] + self.a * self.x[i, 1] - kwargs['rho'] * (self.d * self.x[i, 1] * self.x[i, 2] - self.b * self.x[i, 2])) / (self.c * self.x[i, 1])
                psi1 = self.x[i, 0] - phi
                u.append(-psi1 / self.T1 - self.a  * self.x[i, 1] - self.b * self.x[i, 2] + self.c * self.x[i, 0] * self.x[i, 1] + dphi_dt)

            x1 = self.x[i, 0] + self.h * f1
            x2 = self.x[i, 1] + self.h * f2
            x3 = self.x[i, 2] + self.h * f3
            self.x = np.vstack((self.x, [x1, x2, x3]))

        return self.x, u

    def plot(self, x: List[List[float]], u: List[float], **kwargs) -> NoReturn:
        plt.figure(figsize=(10, 7))
        plt.grid(visible=True)
        plt.plot(self.M, x[:, 0], 'g', label=r'$x_{1}$')
        plt.plot(self.M, x[:, 1], 'b', label=r'$x_{2}$')
        plt.plot(self.M, x[:, 2], 'r', label=r'$x_{3}$')
        if "x2c" in kwargs:
            A = self.x[-1, 0] + self.x[-1, 1] + self.x[-1, 2]
            x1s = (self.a - self.b + A * self.d) / (self.c + self.d)
            x3s = (A * self.c * self.d - self.a * self.d - self.b * self.c) / (self.d ** 2 + self.c * self.d)
            plt.plot(self.M, x1s * np.ones(len(self.M)), 'k--', label=r"$x_{1}^{*}$")
            plt.plot(self.M, kwargs['x2c'] * np.ones(len(self.M)), 'k-.', label=r"$x_{2}^{*}$")
            plt.plot(self.M, x3s * np.ones(len(self.M)), 'k:', label=r"$x_{3}^{*}$")
        elif self.type_goal == "rho_d":
            x1s = (self.a * kwargs['rho'] - self.b + self.d * kwargs['q']) / (self.c * kwargs['rho'])
            x2s = self.b / self.d
            x3s = (self.d * kwargs['q'] - self.b) / (self.d * kwargs['rho'])
            plt.plot(self.M, x1s * np.ones(len(self.M)), 'k--', label=r"$x_{1}^{*}$")
            plt.plot(self.M, x2s * np.ones(len(self.M)), 'k-.', label=r"$x_{2}^{*}$")
            plt.plot(self.M, x3s * np.ones(len(self.M)), 'k:', label=r"$x_{3}^{*}$")
        sns.set_style('whitegrid')
        plt.xlim(0, self.N)
        plt.ylim(min(0, round(0.2*np.floor(round(np.min(x)/ 0.2,2)),1)))
        # plt.legend(loc="best")
        plt.legend(loc="upper right")
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

        plt.figure(figsize=(10, 7))
        plt.plot(self.x[0, 0], self.x[0, 1], 'bo', zorder=3, label="Начальное состояние")
        plt.plot(self.x[-1, 0], self.x[-1, 1], 'ro', zorder=2, label='Конечное состояние')
        plt.plot(self.x[:, 0], self.x[:, 1], 'r-', zorder=1, linewidth=3)
        plt.legend(loc="lower right")
        plt.xlabel(r'$x_{1}$')
        plt.ylabel(r'$x_{2}$')
        # округляем до ближайшего меньшего значения, кратного 0.2; 
        # из-за неточного представления чисел с плавающей запятой используем двойное округление, чтобы избежать появления значений 0.5(9) вместо 0.6 или 9.600000000000001 вместо 9.6
        plt.xlim(left=min(0,round(0.2*np.floor(round(min(self.x[:, 0])/ 0.2,2)),1)))
        plt.ylim(bottom=0)

        if "save_fig" in kwargs:
            plt.savefig(f"{kwargs['name_fig3']}а.png")
            plt.savefig(f"{kwargs['name_fig3']}а.svg")
            plt.savefig(f"{kwargs['name_fig3']}а.eps")

        plt.figure(figsize=(10, 7))
        plt.plot(self.x[0, 1], self.x[0, 2], 'bo', zorder=3, label="Начальное состояние")
        plt.plot(self.x[-1, 1], self.x[-1, 2], 'ro', zorder=2, label='Конечное состояние')
        plt.plot(self.x[:, 1], self.x[:, 2], 'r-', zorder=1, linewidth=3)
        plt.legend(loc="lower right")
        plt.xlabel(r'$x_{2}$')
        plt.ylabel(r'$x_{3}$')
        plt.xlim(left=min(0,round(0.2*np.floor(round(min(self.x[:, 1])/ 0.2,2)),1)))
        plt.ylim(bottom=0)

        if "save_fig" in kwargs:
            plt.savefig(f"{kwargs['name_fig3']}б.png")
            plt.savefig(f"{kwargs['name_fig3']}б.svg")
            plt.savefig(f"{kwargs['name_fig3']}б.eps")

        plt.figure(figsize=(10, 7))
        plt.plot(self.x[0, 0], self.x[0, 2], 'bo', zorder=3, label="Начальное состояние")
        plt.plot(self.x[-1, 0], self.x[-1, 2], 'ro', zorder=2, label='Конечное состояние')
        plt.plot(self.x[:, 0], self.x[:, 2], 'r-', zorder=1, linewidth=3)
        plt.legend(loc="lower right")
        plt.xlabel(r'$x_{1}$')
        plt.ylabel(r'$x_{3}$')
        plt.xlim(left=min(0,round(0.2*np.floor(round(min(self.x[:, 0])/ 0.2,2)),1)))
        plt.ylim(bottom=0)

        if "save_fig" in kwargs:
            plt.savefig(f"{kwargs['name_fig3']}в.png")
            plt.savefig(f"{kwargs['name_fig3']}в.svg")
            plt.savefig(f"{kwargs['name_fig3']}в.eps")

        plt.show()