# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:04:00 2024

@author: Larrot
"""

import numpy as np
from A_Pringle_module import General_Pringle
from Diffusion_coefficient import Diff
from CONSTS import Dead, t_end, M, tau


# Степень вязкости
gamma = 0.375
# gamma = -1

# Коэффициент диффузии


# def Diff(u, x, t):
#     # D = x**(gamma)
#     # D = 100**(gamma)*x**(gamma)
#     D = 1
#     return D


# Начальные условия
def u_init(x):
    if x <= 100:
        u_init = 100*x**(-3/8)
        # u_init = 10**(-3/8)*100*x**(-3/8)
        # u_init = 100*x**(-3/8)
        # u_init = 100
    else:
        u_init = 0
    return u_init


# Граничные условия слева


def u_left(u, x0, x1, t):
    # u_left = (x1/x0)**(3/2)*u[1]
    # u_left = 50*0.1**(-3/8)
    u_left = 0
    # u_left = 10**(-5)
    # u_left = u[1]
    return u_left


# Граничные условия справа
def u_right(u, x, t):
    # u_right = u[N-2]
    # u_right = 0
    u_right = 10**(-5)
    return u_right


# Внутренняя граница
R_in = -2
# R_in = -2

# Внешнаяя граница
R_out = 3
# R_out = 3

# Предельная ошибка
# epsilon = 10**(-5)
epsilon = 10

# Количество узлов
N = 100
# N = 200






""" ANIMATION """
# t_end = 400       
# M = 51
# Dead = 0

# I = np.zeros((M, N))
# D = np.zeros((M, N))
# A = np.zeros((M, N))
# T = np.array([1/M*i for i in range(M)])
# Tau = np.array([T[i]/100 for i in range(len(T))])

# tau = t_end/(M-1)
# tau = 0.1

DATA = np.zeros((M, N))

    
DATA = General_Pringle(Diff, u_init, u_left, u_right,
                                  t_end, R_in, R_out, tau, N, epsilon, M)
#     t+=tau
# for i in range(len(T)):
#     t_end = T[i]
#     tau = Tau[i]
#     DATA[i][:] = General_Pringle(Diff, u_init, u_left, u_right,
#                                   t_end, R_in, R_out, tau, N, epsilon)

np.savetxt(f'DATA_{t_end}_{M}_{Dead}.txt', DATA)


