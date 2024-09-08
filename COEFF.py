# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 00:11:25 2024

@author: Larrot
"""


from scipy import integrate

from matplotlib import pyplot as plt
import numpy as np
from DATA_FOR_ANIMATION import DATA
from CONSTS import Dead, t_end, M
from Smooth_alpha import S_ALPHA


# t_end = 400
# M = 51
# Dead = 0
# V_r = np.loadtxt(f'New_V_r_{t_end}_{M}_{Dead}.txt')


data = DATA
""" GRID """

gamma = 0.375
# gamma = -1
#  Внутреняя граница
R_in = -2
# R_in = 0.1

# Внешняя граница
R_out = 3
# R_out = 1000

# Количество узлов
N = 100

x_s = np.logspace(R_in, R_out, N+1)
# x = np.logspace(R_in, R_out, N+1)

# x = np.logspace(R_in, R_out, N)
x = np.array([(x_s[i+1]+x_s[i])/2 for i in range(N)])


""" GLOBAL CONSTANTS """

# Solar mass
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

NU_0 = 10**(-4)*K*N_A/(MU*(G*M_sun)**(1/2))*280*(L)**(3/2)

"""B_z const"""

H_0 = (K*N_A/(MU*G*M_sun))**(1/2)*280**(1/2)*L**(3/2)

CC = K*N_A/(MU*G*M_sun)
# Cross-section
M_H = 1.67*10**(-24)
ETA_in= 30*2*10**(-9)/(2.3*32.3)*1/M_H


# C_Bz = 4*np.pi*ETA_in*L/H_0**2
C_H = (L**3*K*N_A*280/(MU*G*M_sun))**(1/2)


C_Bz = np.pi*ETA_in*L/C_H**2
# C_Bz = 3*np.pi*ETA_in*(G*M_sun)**(1/2)*L**(-3/2)*10**(-2)

""" Ionization rate """

Ksi_0 = 10**(-17)
R_CR = 100

""" Surface density"""
# T = [0, 0.01, 0.1, 1, 10]

# data0 = np.loadtxt(f"Pringle_Sigma_R_{gamma}_N_{N}_T_{T[0]}.txt")
# data1 = np.loadtxt(f"Pringle_Sigma_R_{gamma}_N_{N}_T_{T[1]}.txt")
# data2 = np.loadtxt(f"Pringle_Sigma_R_{gamma}_N_{N}_T_{T[2]}.txt")
# data3 = np.loadtxt(f"Pringle_Sigma_R_{gamma}_N_{N}_T_{T[3]}.txt")
# data4 = np.loadtxt(f"Pringle_Sigma_R_{gamma}_N_{N}_T_{T[4]}.txt")

# data = np.array([data0, data1, data2, data3, data4])

""" Arrays for calculation """

""" Ionization by cosmic rays """
Ksi = np.zeros((M, N))
for t in range(M):
    for i in range(N):
        Ksi[t][i] = Ksi_0*np.exp(-data[t][i]/R_CR)

""" Temperature profile """
T = np.zeros(N)
for i in range(N):
    T[i] = 280*x[i]**(-1/2)


""" concentration """
n = np.zeros((M, N))
C1 = 0.5*(N_A*G*M_sun/(K*MU))**(1/2)*L**(-3/2)
for t in range(M):
    for i in range(N):
        n[t][i] = C1*data[t][i]*T[i]**(-1/2)*x[i]**(-3/2)


""" Coefficient of radiative recombination """


alpha_r = np.zeros(N)
for i in range(N):
    if 0 <= T[i] <= 10**3:
        alpha_r[i] = 2.07*10**(-11)*T[i]**(-1/2)*3
    elif 10**3 <= T[i] <= 10**4:
        alpha_r[i] = 2.07*10**(-11)*T[i]**(-1/2)*1.5


""" Coefficient of dust recombination """
dust_size = 0.1

alpha_g0 = 4.5*10**(-17)*(dust_size/0.1)**(-1)

alpha_gm = 3*10**(-18)

alpha_g = np.zeros(N)

for i in range(N):
    if T[i] <= 150:
        alpha_g[i] = alpha_g0

    elif 150 <= T[i] <= 400:
        alpha_g[i] = -1/250*(alpha_g0-alpha_gm)*T[i] + \
            8/5*alpha_g0 - 3/5*alpha_gm

    elif 400 <= T[i] <= 1500:
        alpha_g[i] = alpha_gm

    elif 1500 <= T[i] <= 2000:
        alpha_g[i] = -1/500*alpha_gm*T[i]+4*alpha_gm


betta = np.zeros((M, N))
for t in range(M):
    for i in range(N):
        if n[t][i] == 0:
            betta[t][i] = 0
        else:
            betta[t][i] = (alpha_g[i]*n[t][i]+Ksi[t][i])/(2*alpha_r[i]*n[t][i])


gama = np.zeros((M, N))
for t in range(M):
    for i in range(N):
        if n[t][i] == 0:
            gama[t][i] = 0
        else:
            gama[t][i] = Ksi[t][i]/(alpha_r[i]*n[t][i])
            
betta1 = np.zeros((M, N))
for t in range(M):
    for i in range(N):
        if n[t][i] == 0:
            betta1[t][i] = 0
        else:
            betta1[t][i] = (alpha_g[i]*n[t][i]+2.6*10**(-19))/(2*alpha_r[i]*n[t][i])


gama1 = np.zeros((M, N))
for t in range(M):
    for i in range(N):
        if n[t][i] == 0:
            gama1[t][i] = 0
        else:
            gama1[t][i] = 2.6*10**(-19)/(alpha_r[i]*n[t][i])

Xi = np.zeros((M, N))
Xt = np.zeros((M, N))
X_xr = np.zeros((M, N))
for t in range(M):
    for i in range(N):
        X_xr[t][i] = 2.6*10**(-15)*x[i]**(-2)*np.exp(-data[t][i]/8)

# Ionization by radiactive elements
Xr = np.zeros((M, N))
for t in range(M):
    for i in range(N):
        # Xr[t][i] = 2.6*10**(-19)
        Xr[t][i] = -betta1[t][i] + (betta1[t][i]**2+gama1[t][i])**(1/2)


X = np.zeros((M, N))


for t in range(M):
    for i in range(N):
        if data[t][i] <= 10**(-5):
            Xi[t][i] = 0
        else:
            Xi[t][i] = -betta[t][i] + (betta[t][i]**2+gama[t][i])**(1/2)

for t in range(M):
    for i in range(N):
        if data[t][i] <= 10**(-5):
            Xt[t][i] = 0
        else:
            Xt[t][i] = 1.8*10**(-11)*(T[i]/1000)**(3/4)*(Nu_K/10**(-7))**(0.5) \
                * (n[t][i]/10**(13))**(-0.5) \
                * np.exp(-25000/T[i])/(1.15*10**(-11))

""" Only dust recombination """
# for t in range(5):
#     for i in range(N):
#         if data[t][i] <= 10**(-5):
#             Xt[t][i] = 0
#         else:
#             if alpha_g[i] == 0:
#                 X[t][i] = 1 + Xr[t][i] + Xt[t][i]
#             else:
#                 X[t][i] = Ksi[t][i]/(alpha_g[i]*n[t][i]+Ksi[t][i]) +Xr[t][i] + Xt[t][i]


for t in range(M):
    for i in range(N):
        if data[t][i] <= 10**(-5):
            Xt[t][i] = 0
        else:
            X[t][i] = Xi[t][i] + Xt[t][i] + Xr[t][i] + X_xr[t][i]

""" D(x) """

D = np.zeros((M, N))
if Dead == 0:
    ALPHA = 0.01
    for t in range(M):
        for i in range(N):
            D[t][i] = ALPHA/0.0001*x[i]
else:
    for t in range(M):
        for i in range(N):
            # if X[t][i] < 10**(-12):
            #     ALPHA = 10**(-4)
            # else:
            #     ALPHA = 0.01
            if X[t][i] == 0:
                ALPHA = 0
            else:
                ALPHA = S_ALPHA(np.log10(X[t][i]))
            D[t][i] = ALPHA/0.0001*x[i]
np.savetxt(f'D_{t_end}_{M}_{Dead}.txt', D)


A = np.zeros((M, N))
if Dead == 0:
    A[:][:] = 0.01
else:
    for t in range(M):
        for i in range(N):
            if X[t][i] < 10**(-12):
                ALPHA = 10**(-4)
            else:
                ALPHA = 0.01
            # if X[t][i] == 0:
            #     ALPHA = 0
            # else:
            #     ALPHA = S_ALPHA(np.log10(X[t][i]))
            A[t][i] = ALPHA

M_dot = np.zeros((M, N))
for t in range(M):
    for i in range(N):
        M_dot[t][i] = 3*np.pi*D[t][i]*DATA[t][i]

New_V_r = np.zeros((M, N))
for t in range(M):
    for i in range(N):
        # New_V_r[t][i] = - M_dot[t][i]/(2*np.pi*x[i]*DATA[t][i])
        # New_V_r[t][i] = -3/2*100*D[t][i]/x[i]
        # New_V_r[t][i] = -3/2*D[t][i]/x[i]
        New_V_r[t][i] = (0.034)*(-3/2)*D[t][i]/x[i]
        # New_V_r[t][i] = (-3/2)*D[t][i]/x[i]
V_r = New_V_r


B_zz = np.zeros((M,N))
for t in range(M):
    for i in range(N):
        B_zz[t][i] = 100*DATA[t][i]/540
B_z1 = B_zz
        
B_z_diff = np.zeros((M,N))
for t in range(M):
    for i in range(N):
        # B_z_diff[t][i] = DATA[t][i]*(C_Bz*X[t][i]*abs(V_r[t][i])*x[i]**(-3/2))**(1/2)
        B_z_diff[t][i] = DATA[t][i]*(0.034*1.5*C_Bz*X[t][i]*D[t][i]*x[i]**(-5/2))**(1/2)
        # B_z_diff[t][i] = DATA[t][i]*(0.034*1.5/0.0001*C_Bz*X[t][i]*A[t][i]*x[i]**(-3/2))**(1/2)
B_z2 = B_z_diff
        
B_z = np.zeros((M,N))
for t in range(M):
    for i in range(N):
        # if X[t][i] <= 10**(-12):
        #     B_z[t][i] = min(B_z1[t][i], B_z2[t][i])
        # else:
        #     B_z[t][i] = B_z1[t][i]
        B_z[t][i] = min(B_z1[t][i], B_z2[t][i])
            
# B_z = np.zeros((M,N))
# for t in range(M):
#     for i in range(N):
#         if X[t][i] <= 10**(-12):
#             B_z[t][i] = min(B_zz[t][i], B_z_diff[t][i])
#         else:
#             B_z[t][i] = B_zz[t][i]
            
B_z3 = B_z   

B_z1_i = np.zeros((M,N))

for t in range(M):
    for i in range(N):
        B_z1_i[t][i] = B_z1[t][i]*x[i]   

x_i = np.logspace(R_in, R_out, N)


Phi_z1 = np.zeros(M)
for t in range(M):
    Phi_z1[t]= 2*np.pi*2.25*10**(-3)*integrate.trapezoid(B_z1_i[t][:], x[:])
    
np.savetxt(f'Phi_1_{t_end}_{M}_{Dead}.txt', Phi_z1)

B_z2_i = np.zeros((M,N))

for t in range(M):
    for i in range(N):
        B_z2_i[t][i] = B_z2[t][i]*x[i]

Phi_z2 = np.zeros(M)
for t in range(M):
    Phi_z2[t]= 2*np.pi*2.25*10**(-3)*integrate.trapezoid(B_z2_i[t][1:], x[1:])
    
np.savetxt(f'Phi_2_{t_end}_{M}_{Dead}.txt', Phi_z2)



Phi_z3 = np.zeros(M)

B_z3_i = np.zeros((M,N))

for t in range(M):
    for i in range(N):
        B_z3_i[t][i] = B_z3[t][i]*x[i]
for t in range(M):
    Phi_z3[t]= 2*np.pi*2.25*10**(-3)*integrate.trapezoid(B_z3_i[t][:], x[:])
    
np.savetxt(f'Phi_3_{t_end}_{M}_{Dead}.txt', Phi_z3)


np.savetxt(f'A_{t_end}_{M}_{Dead}.txt', A)
np.savetxt(f'B_z_{t_end}_{M}_{Dead}.txt', B_z)
np.savetxt(f'X_{t_end}_{M}_{Dead}.txt', X)
# np.savetxt(f'Phi_{t_end}_{M}_{Dead}.txt', Phi_z)
# np.savetxt(f'B_z_{t_end}_{M}_{Dead}.txt', B_z)
np.savetxt(f'B_z1_{t_end}_{M}_{Dead}.txt', B_z1)
np.savetxt(f'B_z2_{t_end}_{M}_{Dead}.txt', B_z2)
np.savetxt(f'B_z3_{t_end}_{M}_{Dead}.txt', B_z3)

np.savetxt(f'New_V_r_{t_end}_{M}_{Dead}.txt', New_V_r)
np.savetxt(f'M_dot_{t_end}_{M}_{Dead}.txt', M_dot)
