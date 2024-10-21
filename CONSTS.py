# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:27:42 2024

@author: Larrot
"""



"Constants for numerical calculations"
t_end = 30
tau = t_end/100
# tau = t_end/10
# t_end = 10*1000
M = int(t_end/tau)+1
Dead = 1

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

"""Physical constants"""

M_sun = 1.98*10**33

# Gravitational constant
G = 6.67*10**(-8)

# Boltzman's constant
K = 1.38*10**(-16)

# Molar mass
MU = 2.3

# astronomical unit
L = 1.5*10**13

# Avogadro constant
N_A = 6.02*10**(23)

# Amount of Potassium

Nu_K = 10**(-7)

""" Ionization rate """

Ksi_0 = 10**(-17)

Ksi_r = 2.6*10**(-19)

R_CR = 100

C1 = (N_A*G*M_sun/(K*MU))**(1/2)*L**(-3/2)

dust_size = 0.1

alpha_g0 = 4.5*10**(-17)*(dust_size/0.1)**(-1)

alpha_gm = 3*10**(-18)

sigma = 5.67*10**(-5)
