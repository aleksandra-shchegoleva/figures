import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sympy import exp, symbols, solve, sqrt
import random

random.seed(20)
np.random.seed(20)
mpl.rcParams['figure.dpi'] = 300

class NTP_model():
    def __init__(self, P1, P2, Z, r1, r2, a1, a2, K1, K2, w1, w2, y1, y2, d1, d2, b1, m, m1, N, h, x2c, T1=1, T2=1, T=1,
                 rho=None, d=None, c=None, sigma=None, c_xi=None, m_t=None, x2c_2=None, rho2=None, d_2=None):
        self.x = np.array([[P1, P2, Z]])
        self.r1, self.r2 = r1, r2
        self.a1, self.a2 = a1, a2
        self.K1, self.K2 = K1, K2
        self.w1, self.w2 = w1, w2
        self.y1, self.y2 = y1, y2
        self.d1, self.d2 = d1, d2
        self.b1, self.m, self.m1 = b1, m, m1
        self.N, self.h = N, h
        self.M = np.arange(0, self.N, self.h)
        self.x2c = x2c
        self.T = T
        self.T1 = T1
        self.T2 = T2
        self.rho, self.d = rho, d
        self.rho2, self.d_2 = rho2, d_2
        self.c = c
        self.sigma = sigma
        self.c_xi = c_xi
        self.m_t = m_t
        self.x2c_2 = x2c_2
        self.set_text()

    def set_text(self):
        self.TEXT = f"$P_{1}(0)$ = {self.x[0, 0]}\n" \
                    f"$P_{2}(0)$ = {self.x[0, 1]}\n" \
                    f"$Z(0)$ = {self.x[0, 2]}\n" \
                    f"$r_{1}$ = {self.r1}\n" \
                    f"$K_{1}$ = {self.K1}\n" \
                    f"$K_{2}$ = {self.K2}\n" \
                    f"$w_{1}$ = {self.w1}\n" \
                    f"$w_{2}$ = {self.w2}\n" \
                    f"$d_{1}$ = {self.d1}\n" \
                    f"$d_{2}$ = {self.d2}\n" \
                    f"$\\gamma_{1}$ = {self.y1}\n" \
                    f"$\\gamma_{2}$ = {self.y2}\n" \
                    f"$b_{1}$ = {self.b1}\n" \
                    f"$m$ = {self.m}\n" \
                    f"$m1$ = {self.m1}\n" \
                    f"$\\tau$ = {self.h}\n"

    def __repr__(self):
        return (f"r1 {self.r1} K1 {self.K1} K2 {self.K2} w1 {self.w1} w2 {self.w2} d1 {self.d1} d2 {self.d2} y1 {self.y1}"
                f" y2 {self.y2} b1 {self.b1} m {self.m} m1 {self.m1} x2c {self.x2c}")

    def state_points(self):
        x1, x2, x3 = symbols('x1 x2 x3', positive=True)
        a1, a2, r1, r2 = symbols('a1 a2 r1 r2', positive=True)
        K1, K2, w1, w2 = symbols('K1 K2 w1 w2', positive=True)
        d1, d2 = symbols('d1 d2', positive=True)
        beta, c = symbols('beta c', positive=True)
        rho, d = symbols('rho d', positive=True)

        f1 = r1 * x1 * (1 - (x1 + a1 * x2) / K1) - (w1 * x1 * x3) / (d1 + x1)
        f2 = r2 * x2 * (1 - (x2 + a2 * x1) / K2) - (w2 * x2 * x3) / (
                    d2 + x2 + beta * x1)
        f3 = (w1 * x1 * x3) / (d1 + x1) - (w2 * x2 * x3) / (
                    d2 + x2 + beta * x1) - c * x3
        phi = x1 - rho*x2 + d
        x = solve([f1, f2, f3, phi], [x1, x2, x3])
        print(x)

    def plot_solution(self, x, u, xc=None, **kwargs):
        plt.figure(figsize=(10, 7))
        plt.plot(self.M, x[0], 'g', label=r'$P_{1}$')
        plt.plot(self.M, x[1], 'b', label=r'$P_{2}$')
        plt.plot(self.M, x[2], 'r', label=r'$Z$')
        plt.grid(visible=True)
        if xc:
            for key, value in xc.items():
                plt.plot(self.M, value * np.ones(len(self.M)), 'k--', label=key)
        sns.set_style('whitegrid')

        if self.x2c_2:
            for key, value in xc.items():
                plt.plot(self.M, self.x2c_2 * np.ones(len(self.M)), 'k-.', label=key)
        if self.m_t:
            plt.axvline(x=self.m_t, color='purple', linestyle='--', label='Инициация смены\nсостояния')
        plt.xlim(0, self.N)
        plt.ylim(0)
        plt.legend(loc="upper right")
        plt.xlabel('Время, дни')
        plt.ylabel('Популяция, ед/л')
        # plt.annotate(
        #         self.TEXT,
        #         xy=(1, 0),
        #         textcoords='axes fraction',
        #         xytext=(0.99, 0.01),
        #         horizontalalignment='right',
        #         verticalalignment='bottom',
        #         fontsize=12
        #     )
        if "save_fig" in kwargs:
            plt.savefig(f"{kwargs['fig_name1']}.png")
            plt.savefig(f"{kwargs['fig_name1']}.svg")
            plt.savefig(f"{kwargs['fig_name1']}.eps")

        plt.figure(figsize=(10, 7))
        plt.plot(self.M, u, 'k', label=r'$u(t)$', zorder=2)
        sns.set_style('whitegrid')
        plt.xlim(0, self.N)
        if self.m_t:
            plt.axvline(x=self.m_t, color='purple', linestyle='--', label='Инициация смены\nсостояния', zorder=1)
        plt.legend(loc="best")
        plt.xlabel('Время, дни')
        plt.ylabel('Управление')

        if "save_fig" in kwargs:
            plt.savefig(f"{kwargs['fig_name2']}.png")
            plt.savefig(f"{kwargs['fig_name2']}.svg")
            plt.savefig(f"{kwargs['fig_name2']}.eps")

        plt.show()


    def calc(self, plot, get_value=False, **kwargs):
        x1, x2, x3 = [self.x[0, 0]].copy(), [self.x[0, 1]].copy(), [self.x[0, 2]].copy()
        u = [0]

        for i in range(len(self.M)-1):
            if self.m_t and self.m_t > self.M[i] and self.x2c_2 is None:
                u[i] = 0
            f1 = self.r1*x1[i]*(1 - (x1[i] + self.a1*x2[i])/self.K1) - (self.w1*x1[i]*x3[i])/(self.d1 + x1[i])
            f2 = self.r2*x2[i]*(1 - (x2[i] + self.a2*x1[i])/self.K2) - (self.w2*x2[i]*x3[i])/(self.d2 + self.b1*x2[i]**2) + u[i]
            f3 = (self.y1*x1[i]*x3[i])/(self.d1 + x1[i]) - (self.y2*x2[i]*x3[i])/(self.d2 + self.b1*x2[i]**2) - self.m * x3[i] - self.m1 * x3[i] ** 2
            
            if self.x2c_2 and self.m_t < self.M[i]:
                psi = x2[i] - self.x2c_2
            else:
                psi = x2[i] - self.x2c
            x1.append(x1[i] + self.h * f1)
            x2.append(x2[i] + self.h * f2)
            x3.append(x3[i] + self.h * f3)
            u.append(-psi / self.T - self.r2*x2[i]*(1 - (x2[i] + self.a2*x1[i])/self.K2) + (self.w2*x2[i]*x3[i])/(self.d2 + self.b1*x2[i]**2))

        if plot:
            self.plot_solution([x1, x2, x3], u, {r'$P_2^*$': self.x2c}, **kwargs)
        if get_value:
            return [x1, x2, x3]

    def cacl_NAS_add(self, plot, get_value=False):
        x1, x2, x3 = [self.x[0, 0]].copy(), [self.x[0, 1]].copy(), [self.x[0, 2]].copy()
        u = [0]
        psi = []
        xi = [random.normalvariate(0, self.sigma) for _ in range(len(self.M) + 1)]

        # first step
        f1 = self.r1 * x1[0] * (1 - (x1[0] + self.a1 * x2[0]) / self.K1) - (self.w1 * x1[0] * x3[0]) / (self.d1 + x1[0])
        f2 = self.r2 * x2[0] * (1 - (x2[0] + self.a2 * x1[0]) / self.K2) - (self.w2 * x2[0] * x3[0]) / (
                    self.d2 + self.b1 * x2[0] ** 2)
        f3 = (self.y1 * x1[0] * x3[0]) / (self.d1 + x1[0]) - (self.y2 * x2[0] * x3[0]) / (
                    self.d2 + self.b1 * x2[0] ** 2) - self.m * x3[0] - self.m1 * x3[0] ** 2
        psi.append(x2[0] - self.x2c)
        x1.append(x1[0] + self.h * f1)
        x2.append(x2[0] + self.h * f2 + self.h * u[0] + xi[1] + self.c_xi * xi[0])
        x3.append(x3[0] + self.h * f3)

        for i in range(1, len(self.M)-1):
            f1 = self.r1*x1[i]*(1 - (x1[i] + self.a1*x2[i])/self.K1) - (self.w1*x1[i]*x3[i])/(self.d1 + x1[i])
            f2 = self.r2*x2[i]*(1 - (x2[i] + self.a2*x1[i])/self.K2) - (self.w2*x2[i]*x3[i])/(self.d2 + self.b1*x2[i]**2)
            f3 = (self.y1*x1[i]*x3[i])/(self.d1 + x1[i]) - (self.y2*x2[i]*x3[i])/(self.d2 + self.b1*x2[i]**2) - self.m * x3[i] - self.m1 * x3[i] ** 2

            psi.append(x2[i] - self.x2c)
            x1.append(x1[i] + self.h * f1)
            u.append((-x2[i] - self.c_xi * (psi[i] + self.T1 * psi[i - 1]) + self.x2c - self.T1 * psi[i]) / self.h
                     - f2)
            x2.append(x2[i] + self.h * f2 + self.h * u[i] + xi[i + 1] + self.c_xi * xi[i])
            x3.append(x3[i] + self.h * f3)

        if plot:
            self.plot_solution([x1, x2, x3], u, {r'$P_2^*$': self.x2c})
        if get_value:
            return [x1, x2, x3]

    def calc_multi(self, plot, get_value=False, **kwargs):
        x1, x2, x3 = [self.x[0, 0]].copy(), [self.x[0, 1]].copy(), [self.x[0, 2]].copy()
        u = [0]

        for i in range(len(self.M)-1):
            if self.m_t and self.m_t > self.M[i] and self.x2c_2 is None:
                u[i] = self.r2
            f1 = self.r1*x1[i]*(1 - (x1[i] + self.a1*x2[i])/self.K1) - (self.w1*x1[i]*x3[i])/(self.d1 + x1[i])
            f2 = u[i]*x2[i]*(1 - (x2[i] + self.a2*x1[i])/self.K2) - (self.w2*x2[i]*x3[i])/(self.d2 + self.b1*x2[i]**2)
            f3 = (self.y1*x1[i]*x3[i])/(self.d1 + x1[i]) - (self.y2*x2[i]*x3[i])/(self.d2 + self.b1*x2[i]**2) - self.m*x3[i] - self.m1*x3[i]**2
            
            if self.x2c_2 and self.m_t < self.M[i]:
                psi = x2[i] - self.x2c_2
            else:
                psi = x2[i] - self.x2c
            x1.append(x1[i] + self.h*f1)
            x2.append(x2[i] + self.h * f2)
            x3.append(x3[i] + self.h * f3)
            u.append((-psi/self.T + self.w2*x2[i]*x3[i]/(self.d2 + self.b1*x2[i]**2)) / (x2[i]*(1 - (x2[i] + self.a2*x1[i])/self.K2)))

        if plot:
            self.plot_solution([x1, x2, x3], u, {r'$P_2^*$': self.x2c}, **kwargs)
        if get_value:
            return [x1, x2, x3]

    def calc_multi_big_phase(self, plot, get_value=False):
        x1, x2, x3 = [self.x[0, 0]].copy(), [self.x[0, 1]].copy(), [self.x[0, 2]].copy()
        u = [0]


        for i in range(len(self.M)-1):
            f1 = u[i]*x1[i]*(1 - (x1[i] + self.a1*x2[i])/self.K1) - (self.w1*x1[i]*x3[i])/(self.d1 + x1[i])
            f2 = self.r2*x2[i]*(1 - (x2[i] + self.a2*x1[i])/self.K2) - (self.w2*x2[i]*x3[i])/(self.d2 + self.b1*x2[i]**2)
            f3 = (self.y1*x1[i]*x3[i])/(self.d1 + x1[i]) - (self.y2*x2[i]*x3[i])/(self.d2 + self.b1*x2[i]**2) - self.m*x3[i] - self.m1*x3[i]**2

            psi = x2[i] - self.x2c
            phi = (self.K2*self.T2*self.b1*self.r2*x2[i]**3 + self.K2*self.T2*self.d2*self.r2*x2[i] - self.K2*self.T2*self.w2*x2[i]*x3[i]
                   + self.K2*self.b1*x2[i]**3 - self.K2*self.b1*x2[i]**2*self.x2c + self.K2*self.d2*x2[i] - self.K2*self.d2*self.x2c
                   - self.T2*self.b1*self.r2*x2[i]**4 - self.T2*self.d2*self.r2*x2[i]**2)\
                  /(self.T2*self.a2*self.r2*x2[i]*(self.b1*x2[i]**2 + self.d2))
            psi1 = x1[i] - phi
            dphi_dp2 = -2*self.b1*(self.K2*self.T2*self.b1*self.r2*x2[i]**3 + self.K2*self.T2*self.d2*self.r2*x2[i] -
                        self.K2*self.T2*self.w2*x2[i]*x3[i] + self.K2*self.b1*x2[i]**3 - self.K2*self.b1*x2[i]**2*self.x2c +
                        self.K2*self.d2*x2[i] - self.K2*self.d2*self.x2c - self.T2*self.b1*self.r2*x2[i]**4 -
                        self.T2*self.d2*self.r2*x2[i]**2)/(self.T2*self.a2*self.r2*(self.b1*x2[i]**2 + self.d2)**2) + \
                       (3*self.K2*self.T2*self.b1*self.r2*x2[i]**2 + self.K2*self.T2*self.d2*self.r2 -
                        self.K2*self.T2*self.w2*x3[i] + 3*self.K2*self.b1*x2[i]**2 - 2*self.K2*self.b1*x2[i]*self.x2c +
                        self.K2*self.d2 - 4*self.T2*self.b1*self.r2*x2[i]**3 - 2*self.T2*self.d2*self.r2*x2[i])\
                       /(self.T2*self.a2*self.r2*x2[i]*(self.b1*x2[i]**2 + self.d2)) - (self.K2*self.T2*self.b1*self.r2*x2[i]**3 +
                        self.K2*self.T2*self.d2*self.r2*x2[i] - self.K2*self.T2*self.w2*x2[i]*x3[i] + self.K2*self.b1*x2[i]**3 -
                        self.K2*self.b1*x2[i]**2*self.x2c + self.K2*self.d2*x2[i] - self.K2*self.d2*self.x2c -
                        self.T2*self.b1*self.r2*x2[i]**4 - self.T2*self.d2*self.r2*x2[i]**2)/\
                       (self.T2*self.a2*self.r2*x2[i]**2*(self.b1*x2[i]**2 + self.d2))

            dphi_dz = -self.K2*self.w2/(self.a2*self.r2*(self.b1*x2[i]**2 + self.d2))
            dphi_dt = dphi_dp2 * f2 + dphi_dz * f3
            x1.append(x1[i] + self.h * f1)
            x2.append(x2[i] + self.h * f2)
            x3.append(x3[i] + self.h * f3)
            u.append((-psi1 / self.T1 + dphi_dt + self.w1*x1[i]*x3[i] / (self.d1 + x1[i])) / (x1[i] * (1 - (x1[i] + self.a1*x2[i])/self.K1)))

        if plot:
            self.plot_solution([x1, x2, x3], u)
        if get_value:
            return [x1, x2, x3]

    def calc_big_phase(self, plot):
        x1, x2, x3 = [self.x[0, 0]].copy(), [self.x[0, 1]].copy(), [self.x[0, 2]].copy()
        u = [0]
        for i in range(len(self.M)-1):
            f1 = self.r1*x1[i]*(1 - (x1[i] + self.a1*x2[i])/self.K1) - (self.w1*x1[i]*x3[i])/(self.d1 + x1[i]) + u[i]
            f2 = self.r2*x2[i]*(1 - (x2[i] + self.a2*x1[i])/self.K2) - (self.w2*x2[i]*x3[i])/(self.d2 + self.b1*x2[i]**2)
            f3 = (self.y1*x1[i]*x3[i])/(self.d1 + x1[i]) - (self.y2*x2[i]*x3[i])/(self.d2 + self.b1*x2[i]**2) - self.m*x3[i] - self.m1*x3[i]**2

            psi = x2[i] - self.x2c
            phi = self.K2 / self.a2 * (1 - (-psi/self.T2 + self.w2*x2[i]*x3[i] / (self.d2 + self.b1*x2[i]**2)) / (self.r2*x2[i])) \
                  - x2[i]/self.a2
            psi1 = x1[i] - phi
            dphi_dp2 = self.K2 / self.a2 * (((-1/self.T2 + (self.w2*x3[i]*(self.d2 + self.b1*x2[i]**2) -
                        self.w2*x2[i]*x3[i]*2*self.b1*x2[i]) / (self.d2 + self.b1*x2[i]**2)**2) * x2[i] - (-psi/self.T2 +
                        self.w2*x2[i]*x3[i] / (self.d2 + self.b1 * x2[i] ** 2))) / (self.r1 * x2[i] ** 2)) - 1/self.a2
            dphi_dz = self.K2 / self.a2 * ((self.w2 * x2[i]) / (self.d2 + self.b1*x2[i]**2) / (self.r2*x2[i]))
            dphi_dt = dphi_dp2 * f2 + dphi_dz * f3
            x1.append(x1[i] + self.h*f1)
            x2.append(x2[i] + self.h * f2)
            x3.append(x3[i] + self.h * f3)
            u.append(-psi1 / self.T1 + dphi_dt - self.r1*x1[i]*(1 - (x1[i] + self.a1*x2[i])/self.K1) + (self.w1*x1[i]*x3[i])/(self.d1 + x1[i]))

        if plot:
            self.plot_solution([x1, x2, x3], u)

    def calc_add_rho_d(self, plot, get_value=False, get_psi=False, get_phi=False, **kwargs):
        x1, x2, x3 = [self.x[0, 0]].copy(), [self.x[0, 1]].copy(), [self.x[0, 2]].copy()
        u = [0]
        a1, a2 = self.a1, self.a2
        w1, w2 = self.w1, self.w2
        r1, r2 = self.r1, self.r2
        K1, K2 = self.K1, self.K2
        y1, y2 = self.y1, self.y2
        d1, d2 = self.d1, self.d2
        b1, m, m1 = self.b1, self.m, self.m1
        N, h = self.N, self.h
        M = np.arange(0, self.N, self.h)
        x2c = self.x2c
        T = self.T
        rho, d = self.rho, self.d
        rho2, d_2 = self.rho2, self.d_2
        if get_psi:
            psi_lst = []

        for i in range(len(self.M)-1):
            if self.m_t and self.m_t > self.M[i] and rho2 is None:
                u[i] = 0
            f1 = r1*x1[i]*(1 - (x1[i] + a1*x2[i])/K1) - (w1*x1[i]*x3[i])/(d1 + x1[i]) + u[i]
            f2 = r2*x2[i]*(1 - (x2[i] + a2*x1[i])/K2) - (w2*x2[i]*x3[i])/(d2 + b1*x2[i]**2)
            f3 = (y1*x1[i]*x3[i])/(d1 + x1[i]) - (y2*x2[i]*x3[i])/(d2 + b1*x2[i]**2) - m*x3[i] - m1*x3[i]**2

            if rho2 and d_2 and self.m_t < self.M[i]:
                psi = x1[i] - rho2*x2[i] + d_2
            else:
                psi = x1[i] - rho*x2[i] + d
            if get_psi:
                psi_lst.append(psi)

            x1.append(x1[i] + h * f1)
            x2.append(x2[i] + h * f2)
            x3.append(x3[i] + h * f3)
            u.append(-psi/T - r1*x1[i]*(1 - (x1[i] + a1*x2[i])/K1) + w1*x1[i]*x3[i]/(d1 + x1[i]) + rho*f2)
        if plot:
            self.TEXT += f"$\\rho$ = {self.rho}\n" \
                         f"$T$ = {self.T}\n" \
                         f"$d$ = {self.d}"
            self.plot_solution([x1, x2, x3], u, **kwargs)

        if get_value:
            return [x1, x2, x3]
        if get_psi:
            psi_lst.append(x1[-1] - rho*x2[-1] + d)
            return psi_lst

    def calc_big_phase_add_xc(self, plot):
        x1, x2, x3, r2 = [self.x[0, 0]].copy(), [self.x[0, 1]].copy(), [self.x[0, 2]].copy(), [0]
        u = [0]
        a1, a2 = self.a1, self.a2
        w1, w2 = self.w1, self.w2
        r1 = self.r1
        K1, K2 = self.K1, self.K2
        y1, y2 = self.y1, self.y2
        d1, d2 = self.d1, self.d2
        b1, m, m1 = self.b1, self.m, self.m1
        N, h = self.N, self.h
        M = np.arange(0, self.N, self.h)
        x2c = self.x2c
        T1, T2 = self.T1, self.T2

        for i in range(len(self.M)-1):
            f1 = r1*x1[i]*(1 - (x1[i] + a1*x2[i])/K1) - (w1*x1[i]*x3[i])/(d1 + x1[i])
            f2 = r2[i]*x2[i]*(1 - (x2[i] + a2*x1[i])/K2) - (w2*x2[i]*x3[i])/(d2 + b1*x2[i]**2)
            f3 = (y1*x1[i]*x3[i])/(d1 + x1[i]) - (y2*x2[i]*x3[i])/(d2 + b1*x2[i]**2) - m*x3[i] - m1*x3[i]**2
            dr2dt = u[i]

            psi = x2[i] - x2c
            phi = (-psi/T2 + w2*x2[i]*x3[i] / (d2 + b1*x2[i]**2)) / (x2[i] * (1 - (x2[i] + a2*x1[i])/K2))
            psi1 = r2[i] - phi

            dphi_dx1 = a2*(w2*x2[i]*x3[i]/(b1*x2[i]**2 + d2) + (-x2[i] + x2c)/T2)/(K2*x2[i]*(1 - (a2*x1[i] + x2[i])/K2)**2)
            dphi_dx2 = (-2*b1*w2*x2[i]**2*x3[i]/(b1*x2[i]**2 + d2)**2 + w2*x3[i]/(b1*x2[i]**2 + d2) - 1/T2)/(x2[i]*(1 -
                        (a2*x1[i] + x2[i])/K2)) - (w2*x2[i]*x3[i]/(b1*x2[i]**2 + d2) + (-x2[i] + x2c)/T2)/(x2[i]**2*(1 -
                        (a2*x1[i] + x2[i])/K2)) + (w2*x2[i]*x3[i]/(b1*x2[i]**2 + d2) + (-x2[i] + x2c)/T2)/(K2*x2[i]*(1 -
                        (a2*x1[i] + x2[i])/K2)**2)
            dphi_dx3 = w2/((1 - (a2*x1[i] + x2[i])/K2)*(b1*x2[i]**2 + d2))
            dphi = dphi_dx1*f1 + dphi_dx2*f2 + dphi_dx3*f3

            x1.append(x1[i] + h * f1)
            x2.append(x2[i] + h * f2)
            x3.append(x3[i] + h * f3)
            r2.append(r2[i] + h * dr2dt)

            u.append(-psi1/T1 + dphi)

        if plot:
            self.TEXT += f"$T_1$ = {T1}\n" \
                         f"$T_2$ = {T2}\n" \
                        f"$r_2(0)$ = {r2[0]}\n"
            self.plot_solution([x1, x2, x3], u, {r'$P_2^*$': x2c})

    def calc_big_phase_multi_xc(self, plot, get_value=False):
        x1, x2, x3, r2 = [self.x[0, 0]].copy(), [self.x[0, 1]].copy(), [self.x[0, 2]].copy(), [self.r2]
        u = [0]
        a1, a2 = self.a1, self.a2
        w1, w2 = self.w1, self.w2
        r1 = self.r1
        K1, K2 = self.K1, self.K2
        y1, y2 = self.y1, self.y2
        d1, d2 = self.d1, self.d2
        b1, m, m1 = self.b1, self.m, self.m1
        N, h = self.N, self.h
        M = np.arange(0, self.N, self.h)
        x2c = self.x2c
        T1, T2 = self.T1, self.T2

        for i in range(len(self.M) - 1):
            f1 = r1 * r2[i] * x1[i] * (1 - (x1[i] + a1 * x2[i]) / K1) - (w1 * x1[i] * x3[i]) / (d1 + x1[i])
            f2 = r2[i] * x2[i] * (1 - (x2[i] + a2 * x1[i]) / K2) - (w2 * x2[i] * x3[i]) / (d2 + b1 * x2[i] ** 2)
            f3 = (y1 * x1[i] * x3[i]) / (d1 + x1[i]) - (y2 * x2[i] * x3[i]) / (d2 + b1 * x2[i] ** 2) - m * x3[i] - m1 * \
                 x3[i] ** 2
            dr2dt = u[i]

            psi = x2[i] - x2c
            phi = (-psi / T2 + w2 * x2[i] * x3[i] / (d2 + b1 * x2[i] ** 2)) / (x2[i] * (1 - (x2[i] + a2 * x1[i]) / K2))
            psi1 = r2[i] - phi

            dphi_dx1 = a2*(w2*x2[i]*x3[i]/(b1*x2[i]**2 + d2) + (-x2[i] + x2c)/T2)/(K2*x2[i]*(1 - (a2*x1[i] + x2[i])/K2)**2)
            dphi_dx2 = (-2*b1*w2*x2[i]**2*x3[i]/(b1*x2[i]**2 + d2)**2 + w2*x3[i]/(b1*x2[i]**2 + d2) - 1/T2)/(x2[i]*(1 - (a2*x1[i] + x2[i])/K2)) - (w2*x2[i]*x3[i]/(b1*x2[i]**2 + d2) + (-x2[i] + x2c)/T2)/(x2[i]**2*(1 - (a2*x1[i] + x2[i])/K2)) + (w2*x2[i]*x3[i]/(b1*x2[i]**2 + d2) + (-x2[i] + x2c)/T2)/(K2*x2[i]*(1 - (a2*x1[i] + x2[i])/K2)**2)
            dphi_dx3 = w2/((1 - (a2*x1[i] + x2[i])/K2)*(b1*x2[i]**2 + d2))
            dphi = dphi_dx1 * f1 + dphi_dx2 * f2 + dphi_dx3 * f3

            x1.append(x1[i] + h * f1)
            x2.append(x2[i] + h * f2)
            x3.append(x3[i] + h * f3)
            r2.append(r2[i] + h * dr2dt)

            u.append(-psi1 / T1 + dphi)

        if plot:
            self.TEXT += f"$T_1$ = {T1}\n" \
                         f"$T_2$ = {T2}\n" \
                         f"$r_2(0)$ = {r2[0]}\n"
            self.plot_solution([x1, x2, x3], u, {r'$P_2^*$': x2c})

        if get_value:
            return [x1, x2, x3]

    def calc_big_phase_rho_d(self, plot=False):
        x1, x2, x3, r2 = [self.x[0, 0]].copy(), [self.x[0, 1]].copy(), [self.x[0, 2]].copy(), [0]
        u = [0]
        a1, a2 = self.a1, self.a2
        w1, w2 = self.w1, self.w2
        r1 = self.r1
        K1, K2 = self.K1, self.K2
        y1, y2 = self.y1, self.y2
        d1, d2 = self.d1, self.d2
        b1, m, m1 = self.b1, self.m, self.m1
        N, h = self.N, self.h
        M = np.arange(0, self.N, self.h)
        rho, d = self.rho, self.d
        T1, T2 = self.T1, self.T2

        for i in range(len(self.M) - 1):
            f1 = r1 * r2[i] * x1[i] * (1 - (x1[i] + a1 * x2[i]) / K1) - (w1 * x1[i] * x3[i]) / (d1 + x1[i])
            f2 = r2[i] * x2[i] * (1 - (x2[i] + a2 * x1[i]) / K2) - (w2 * x2[i] * x3[i]) / (d2 + b1 * x2[i] ** 2)
            f3 = (y1 * x1[i] * x3[i]) / (d1 + x1[i]) - (y2 * x2[i] * x3[i]) / (d2 + b1 * x2[i] ** 2) - m * x3[i] - m1 * \
                 x3[i] ** 2
            dr2dt = u[i]

            psi = x2[i] - rho*x1[i] + d
            phi = (-psi / T2 + w2 * x2[i] * x3[i] / (d2 + b1 * x2[i] ** 2)) / (x2[i] * (1 - (x2[i] + a2 * x1[i]) / K2))
            psi1 = r2[i] - phi

            dphi_dx1 = (rho*(r1*(1 - (a1*x2[i] + x1[i])/K1) + w1*x1[i]*x3[i]/(d1 + x1[i])**2 - w1*x3[i]/(d1 + x1[i]) - r1*x1[i]/K1) - 1/T2)/(x2[i]*(1 - (a2*x1[i] + x2[i])/K2)) + a2*(rho*(r1*x1[i]*(1 - (a1*x2[i] + x1[i])/K1) - w1*x1[i]*x3[i]/(d1 + x1[i])) + w2*x2[i]*x3[i]/(b1*x2[i]**2 + d2) + (-d + rho*x2[i] - x1[i])/T2)/(K2*x2[i]*(1 - (a2*x1[i] + x2[i])/K2)**2)
            dphi_dx2 = (-2*b1*w2*x2[i]**2*x3[i]/(b1*x2[i]**2 + d2)**2 + w2*x3[i]/(b1*x2[i]**2 + d2) + rho/T2 - a1*r1*rho*x1[i]/K1)/(x2[i]*(1 - (a2*x1[i] + x2[i])/K2)) - (rho*(r1*x1[i]*(1 - (a1*x2[i] + x1[i])/K1) - w1*x1[i]*x3[i]/(d1 + x1[i])) + w2*x2[i]*x3[i]/(b1*x2[i]**2 + d2) + (-d + rho*x2[i] - x1[i])/T2)/(x2[i]**2*(1 - (a2*x1[i] + x2[i])/K2)) + (rho*(r1*x1[i]*(1 - (a1*x2[i] + x1[i])/K1) - w1*x1[i]*x3[i]/(d1 + x1[i])) + w2*x2[i]*x3[i]/(b1*x2[i]**2 + d2) + (-d + rho*x2[i] - x1[i])/T2)/(K2*x2[i]*(1 - (a2*x1[i] + x2[i])/K2)**2)
            dphi_dx3 = (-rho*w1*x1[i]/(d1 + x1[i]) + w2*x2[i]/(b1*x2[i]**2 + d2))/(x2[i]*(1 - (a2*x1[i] + x2[i])/K2))
            dphi = dphi_dx1 * f1 + dphi_dx2 * f2 + dphi_dx3 * f3

            x1.append(x1[i] + h * f1)
            x2.append(x2[i] + h * f2)
            x3.append(x3[i] + h * f3)
            r2.append(r2[i] + h * dr2dt)

            u.append(-psi1 / T1 + dphi)

        if plot:
            self.TEXT += f"$T_1$ = {T1}\n" \
                         f"$T_2$ = {T2}\n" \
                         f"$r_2(0)$ = {r2[0]}\n"
            self.plot_solution([x1, x2, x3], u)