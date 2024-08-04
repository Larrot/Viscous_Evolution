# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:53:16 2023

@author: Larrot
"""

from matplotlib import pyplot as plt
import numpy as np
from CONSTS import Dead, t_end, M, tau
import colorcet as cc

clrs = []
for i in range(4):
    """ 
    выбираем палитру fire
    
    всего в палитре 256 цветов, 
    поэтому если хотим сделать список из 3х цветов из этой палитры, 
    то 256/3 ~ 64 - с таким шагом надо выбрать из палитры цвета
    """
    clrs.append(cc.fire[::][i*64])
  
#список выбранных цветов для последующего использования в графиках
colors = [clrs[3], clrs[2], clrs[1], clrs[0]]


# A = np.loadtxt(f"A_{t_end}_{M}_{Dead}.txt")
# D = np.loadtxt(f"D_{t_end}_{M}_{Dead}.txt")
# X = np.loadtxt(f"X_{t_end}_{M}_{Dead}.txt")
# DATA = np.loadtxt(f"DATA_{t_end}_{M}_{Dead}.txt")
# B_z = np.loadtxt(f'B_z_{t_end}_{M}_{Dead}.txt')
# V_r = np.loadtxt(f'V_r_{t_end}_{M}_{Dead}.txt')
# New_V_r = np.loadtxt(f'New_V_r_{t_end}_{M}_{Dead}.txt')
# M_dot = np.loadtxt(f'M_dot_{t_end}_{M}_{Dead}.txt')

# B_z1 = np.loadtxt(f'B_z1_{t_end}_{M}_{Dead}.txt')
# B_z2 = np.loadtxt(f'B_z2_{t_end}_{M}_{Dead}.txt')
# B_z3 = np.loadtxt(f'B_z3_{t_end}_{M}_{Dead}.txt')
B_z1 = np.loadtxt(f'DUST_01_B_z3_{t_end}_{M}_{Dead}.txt')
B_z2 = np.loadtxt(f'DUST_1_B_z3_{t_end}_{M}_{Dead}.txt')
B_z3 = np.loadtxt(f'DUST_1000_B_z3_{t_end}_{M}_{Dead}.txt')
# B_z4 = np.loadtxt(f'B_z4_{t_end}_{M}_{Dead}.txt')
B_z = B_z3


T = np.array([tau*i for i in range(M)])

# t1 = 0

# t2 = 2

# t3 = 5

# t4 = 10

t1 = 0

t2 = 12

t3 = 50

t4 = 75

# Степень вязкости
#  Внутреняя граница
R_in = -2

# Внешняя граница
R_out = 3
# R_out = 1000


# Количество узлов
N = 100

# Время в миллионах лет
TAU = 1.39

D_0 = 5.11*10**(12)
V_0 = 1
M_sun = 1.9*10**(33)
M_0 = 1.8*10**(17)/M_sun*3.15*10**(7)

# Пространственная сетка
# x = np.linspace(R_in, R_out, N)
x_s = np.logspace(R_in, R_out, N+1)
# x = np.logspace(R_in, R_out, N+1)
x = [(x_s[i+1]+x_s[i])/2 for i in range(N)]


# fig = plt.figure(figsize=(8, 6))
fig = plt.figure(figsize=(10, 8))
# fig = plt.figure(figsize=(6, 4))
# fig.subplots_adjust(hspace=0.6, wspace=0.4)

# """График поверхностной плотности"""
plt.rcParams.update({'font.size': 17})

# ax_1 = fig.add_subplot(2, 2, 1)
# # ax_1 = fig.add_subplot(1, 1, 1)
# # # ax_1.text(0.05, 0.95, '(а)', transform=ax_1.transAxes, fontsize=16, fontweight='bold', va='top')

# ax_1.plot(x[1:], DATA[t1][1:], '-', color=colors[0],
#           label=f'T={T[t1]*TAU:.2f}')

# ax_1.plot(x[1:], DATA[t2][1:], ':', color=colors[1],
#           label=f'T={T[t2]*TAU:.2f} млн лет')

# ax_1.plot(x[1:], DATA[t3][1:], '--', color=colors[2],
#           label=f'T={T[t3]*TAU:.2f} млн лет')

# ax_1.set_ylim(10**(-5), 10**(6))
# ax_1.set_xlim(0.01, 1000)

# ax_1.set_xlabel('а.е.')
# ax_1.set_ylabel(r'$\Sigma,~{г}/{см}^2$ ')

# ax_1.set_yscale('log')
# ax_1.set_xscale('log')

# ax_1.set_title(fr'Поверхностная плотность $\Sigma$')
# ax_1.legend(loc='best')


# ax_2 = fig.add_subplot(2, 2, 2)
# # ax_2 = fig.add_subplot(1, 1, 1)
# # ax_2.text(0.05, 0.95, '(б)', transform=ax_2.transAxes, fontsize=16, fontweight='bold', va='top')


# ax_2.plot(x[1:], X[t1][1:], '-', color=colors[0],
#           label=f'T={T[t1]*TAU:.2f} ')

# ax_2.plot(x[1:], X[t2][1:], ':', color=colors[1],
#           label=f'T={T[t2]*TAU:.2f} млн лет')

# ax_2.plot(x[1:], X[t3][1:], '--', color=colors[2],
#           label=f'T={T[t3]*TAU:.2f} млн лет')

# ax_2.set_ylim(10**(-19), 10**(0))
# ax_2.set_xlim(0.01, 1000)

# ax_2.axhline(y=10**(-12), color='k', linestyle='-')

# ax_2.set_xlabel('а.е.')
# ax_2.set_ylabel(r'x')

# ax_2.set_yscale('log')
# ax_2.set_xscale('log')

# ax_2.set_title(fr'Степень ионизации x')
# ax_2.legend(loc='best')


# ax_3 = fig.add_subplot(2, 2, 3)

# ax_3.plot(x[1:], A[t1][1:], '--', color='r',
#           label=f'T={T[t1]*TAU:.2f} Myr')

# ax_3.plot(x[1:], A[t2][1:], '--', color='b',
#           label=f'T={T[t2]*TAU:.2f} Myr')

# ax_3.plot(x[1:], [t3][1:], '--', color='k',
#           label=f'T={T[t3]*TAU:.2f} Myr')

# ax_3.set_ylim(10**(9), 10**(17))
# ax_3.set_xlim(0.01, 1000)


# ax_3.set_xlabel('а.е.')
# ax_3.set_ylabel(r'$\nu,~{см}^2/{с}$')

# ax_3.set_yscale('log')
# ax_3.set_xscale('log')

# ax_3.set_title(fr'Турбулентная вязкость $\nu$')
# ax_3.legend(loc='best')



# ax_4 = fig.add_subplot(2, 2, 4)

# ax_4.plot(x[1:], B_z2[t1][1:], '--', color='r',
#           label=f'T={T[t1]*TAU:.2f} Myr')

# ax_4.plot(x[1:], B_z2[t2][1:], '--', color='b',
#           label=f'T={T[t2]*TAU:.2f} Myr')

# ax_4.plot(x[1:], B_z2[t3][1:], '--', color='k',
#           label=f'T={T[t3]*TAU:.2f} Myr')

# ax_4.set_ylim(0.5*10**(-4), 10**(-1))
# ax_4.set_xlim(0.01, 1000)


# ax_4.set_xlabel('а.е.')
# ax_4.set_ylabel(r'$\alpha$')

# ax_4.set_yscale('log')
# ax_4.set_xscale('log')

# ax_4.set_title(fr'Коэффициент $\alpha$')
# ax_4.legend(loc='best')

# ax_5 = fig.add_subplot(2, 2, 3)
# ax_5.plot(x[1:], V_0*New_V_r[t1][1:], '--', color='r',
#           label=f'T={T[t1]*TAU:.2f} Myr')

# ax_5.plot(x[1:], V_0*New_V_r[t2][1:], '--', color='b',
#           label=f'T={T[t2]*TAU:.2f} Myr')

# ax_5.plot(x[1:], V_0*New_V_r[t3][1:], '--', color='k',
#           label=f'T={T[t3]*TAU:.2f} Myr')

# # ax_5.set_ylim(-6, 0.1)
# ax_5.set_xlim(0.01, 1000)


# ax_5.set_xlabel('а.е.')
# ax_5.set_ylabel(r'$v_r,~{см}/{с}$')

# # ax_5.set_yscale('log')
# ax_5.set_xscale('log')

# ax_5.set_title(fr'Радиальная скорость $v_r$')
# ax_5.legend(loc='best')




# ax_6 = fig.add_subplot(2, 2, 4)
# ax_6.plot(x[1:], M_0*M_dot[t1][1:], '--', color='r',
#           label=f'T={T[t1]*TAU:.2f} Myr')

# ax_6.plot(x[1:], M_0*M_dot[t2][1:], '--', color='b',
#           label=f'T={T[t2]*TAU:.2f} Myr')

# ax_6.plot(x[1:], M_0*M_dot[t3][1:], '--', color='k',
#           label=f'T={T[t3]*TAU:.2f} Myr')

# # ax_6.set_ylim(10**(14), 10**(25))
# ax_6.set_xlim(0.01, 1000)


# ax_6.set_xlabel('а.е.')
# ax_6.set_ylabel(r'$\dot{M},~M_{\odot}/{год}$')

# ax_6.set_yscale('log')
# ax_6.set_xscale('log')

# ax_6.set_title(r'Темп аккреции $\dot{M}$')
# # ax_6.legend(loc='best')


# # ax_7 = fig.add_subplot(2, 2, 4)
# ax_7 = fig.add_subplot(1, 1, 1)
# # plt.rcParams.update({'font.size': 15})

# ax_7.text(0.05, 0.05, '(а)', transform=ax_7.transAxes, fontsize=16, fontweight='bold', va='bottom')

# ax_7.plot(x[1:], B_z[t1][1:], '-', color=colors[0],
#           label=f'T={T[t1]*TAU:.2f} млн лет')

# ax_7.plot(x[1:], B_z[t2][1:], ':', color=colors[1],
#           label=f'T={T[t2]*TAU:.2f} млн лет')

# # ax_7.plot(x[67], B_z[t2][67], 'o', color = 'r')
# # ax_7.plot(x[66], B_z[t2][66], 'x', color = 'k')
# # ax_7.plot(x[68], B_z[t2][68], 'o')
# # ax_7.plot(x[65], B_z[t2][65], 'x')
# # ax_7.plot(x[64], B_z[t2][64], 'o')
# # ax_7.plot(x[62:68], B_z[24][62:68], '--', color='b')

# ax_7.plot(x[1:], B_z[t3][1:], '--', color=colors[2],
#             label=f'T={T[t3]*TAU:.2f} млн лет')

# ax_7.plot(x[1:], B_z[10][1:], '-.', color='g',
#             label=f'T={T[10]*TAU:.2f} млн лет')
# # ax_7.set_ylim(10**(-3), 10**(3))
# ax_7.set_xlim(0.01, 1000)

# ax_7.plot(x[1:], B_z[50][1:], '-.', color='k',
#             label=f'T={T[50]*TAU:.2f} млн лет')
# ax_7.set_ylim(10**(-3), 10**(3))
# ax_7.set_xlim(0.01, 1000)


# ax_7.set_xlabel('а.е.')
# ax_7.set_ylabel(r'$B_z, {Гс}$')

# ax_7.set_yscale('log')
# ax_7.set_xscale('log')

# ax_7.set_title(r'$B_z$ компонента магнитного поля')
# ax_7.legend(loc='best')

# # """1"""
# ax_11 = fig.add_subplot(2, 2, 1)
# # ax_11 = fig.add_subplot(1, 1, 1)
# ax_11.plot(x[1:], DATA[t1][1:], '-', color='r',
#           label='F')

# ax_11.plot(x[1:], DATA[t2][1:], '-', color='y',
#           label=f'AD')

# ax_11.plot(x[1:], DATA[t3][1:], '--', color='k',
#           label='R')

# # ax_11.plot(x[1:], B_z4[t1][1:], '--', color='k',
# #           label=f'$B_z$ для min')

# # ax_11.set_ylim(10**(-6), 10**(6))
# ax_11.set_xlim(0.01, 1000)


# ax_11.set_xlabel('а.е.')
# ax_11.set_ylabel(r'\Sigma$')

# ax_11.set_yscale('log')
# ax_11.set_xscale('log')

# ax_11.set_title(fr'Поверхностная плотность')
# ax_11.legend(loc='best')
# """1"""

# ax_11 = fig.add_subplot(2, 2, 1)

""" Поврехностная плотность """
# ax_11 = fig.add_subplot(1, 1, 1)
# ax_11.plot(x[1:], DATA[t1][1:], '-', color=colors[0],
#           label=fr'$t=${TAU*T[t1]:.2f} млн лет')

# ax_11.plot(x[1:], DATA[t2][1:], '-.', color=colors[1],
#           label=fr'$t=${TAU*T[t2]:.2f} млн лет')

# ax_11.plot(x[1:], DATA[t3][1:], '--', color=colors[2],
#           label=fr'$t=${TAU*T[t3]:.2f} млн лет')

# ax_11.plot(x[1:], DATA[t4][1:], '-', color=colors[3],
#           label=fr'$t=${TAU*T[t4]:.2f} млн лет')

# # ax_11.plot(x[1:], B_z4[t1][1:], '--', color='k',
# #           label=f'$B_z$ для min')

# ax_11.set_ylim(10**(-6), 10**(6))
# ax_11.set_xlim(0.01, 1000)


# ax_11.set_xlabel('$r,~{а.е.}$')
# ax_11.set_ylabel(r'$\Sigma,~{г}/{см}^2$')

# ax_11.set_yscale('log')
# ax_11.set_xscale('log')

# ax_11.set_title(fr'Поверхностная плотность')
# ax_11.legend(loc='best')

""" Степень ионизации """

# ax_11 = fig.add_subplot(1, 1, 1)
# ax_11.plot(x[1:], X[t1][1:], '-', color=colors[0],
#           label=fr'$t=${TAU*T[t1]:.2f} млн лет')

# ax_11.plot(x[1:], X[t2][1:], '--', color=colors[1],
#           label=fr'$t=${TAU*T[t2]:.2f} млн лет')

# ax_11.plot(x[1:], X[t3][1:], '-.', color=colors[2],
#           label=fr'$t=${TAU*T[t3]:.2f} млн лет')

# # ax_11.plot(x[1:], X[t4][1:], '-', color=colors[3],
#           # label=fr'$t=${TAU*T[t4]:.2f} млн лет')
# ax_11.axhline(y=10**(-12), color='grey', linestyle='-')


# ax_11.set_xlim(0.01, 1000)


# ax_11.set_xlabel('$r,~{а.е.}$')
# ax_11.set_ylabel(r'x')

# ax_11.set_yscale('log')
# ax_11.set_xscale('log')

# ax_11.set_title(fr'Степень ионизации $d = 1~мм$')
# ax_11.legend(loc='best')


# """ Магнитное поле """
# ax_11 = fig.add_subplot(1, 1, 1)
# ax_11.plot(x[1:], B_z3[t1][1:], '-', color=colors[0],
#           label=fr'$t=${TAU*T[t1]:.2f} млн лет')

# ax_11.plot(x[1:], B_z3[t2][1:], '--', color=colors[1],
#           label=fr'$t=${TAU*T[t2]:.2f} млн лет')

# ax_11.plot(x[1:], B_z3[t3][1:], '-.', color=colors[2],
#           label=fr'$t=${TAU*T[t3]:.2f} млн лет')

# # ax_11.plot(x[1:], B_z3[t3][1:], '-', color=colors[3],
# #           label=fr'$t=${TAU*T[t4]:.2f} млн лет')

# # ax_11.plot(x[1:], B_z4[t1][1:], '--', color='k',
# #           label=f'$B_z$ для min')

# # ax_11.set_ylim(10**(-6), 10**(6))
# ax_11.set_xlim(0.01, 1000)


# ax_11.set_xlabel('$r,~{а.е.}$')
# ax_11.set_ylabel(r'$B_z, {Гс}$')

# ax_11.set_yscale('log')
# ax_11.set_xscale('log')

# ax_11.set_title(fr'Магнитное поле $d = 1~мм$')
# ax_11.legend(loc='best')

""" Magnetic Field """
# ax_11 = fig.add_subplot(1, 1, 1)

fig, (ax_11, ax_12) = plt.subplots(1, 2, sharey=True, figsize=(10, 6))
# fig = plt.figure(figsize=(10, 8))

# fig, (ax_11, ax_12) = plt.subplots(1, 2)
ax_11.plot(x[1:79], B_z1[t1][1:79], '-', color='k',
          label=fr'$a_g = 0.1~\mu m$')

ax_11.plot(x[1:79], B_z2[t1][1:79], '--', color=colors[1],
          label=fr'$a_g = 1~\mu m$')

ax_11.plot(x[1:79], B_z3[t1][1:79], '-.', color=colors[2],
          label=fr'$a_g = 10^3~\mu m$')

# ax_11.plot(x[1:], B_z3[t3][1:], '-', color=colors[3],
#           label=fr'$t=${TAU*T[t4]:.2f} млн лет')

# ax_11.plot(x[1:], B_z4[t1][1:], '--', color='k',
#           label=f'$B_z$ для min')

ax_11.set_ylim(10**(-6), 10**(4))
ax_11.set_xlim(0.01, 1000)


ax_11.set_xlabel('$r,~{AU}$')
ax_11.set_ylabel(r'$B_z, {G}$')

ax_11.set_yscale('log')
ax_11.set_xscale('log')

ax_11.set_title(fr'$t = {TAU*T[t1]:.2f}~Myr$')

ax_11.text(0.05, 0.1, '(a)', transform=ax_11.transAxes,
      fontsize=20, fontweight='bold', va='top')

ax_11.legend(loc='best')

# ax_12 = fig.add_subplot(1, 2, 2)
# fig, (ax_11, ax_12) = plt.subplots(1, 2)
ax_12.plot(x[1:], B_z1[t2][1:], '-', color='k',
          label=fr'$a_g = 0.1~\mu m$')

ax_12.plot(x[1:], B_z2[t2][1:], '--', color=colors[1],
          label=fr'$a_g = 1~\mu m$')

ax_12.plot(x[1:], B_z3[t2][1:], '-.', color=colors[2],
          label=fr'$a_g = 10^3~\mu m$')

# ax_11.plot(x[1:], B_z3[t3][1:], '-', color=colors[3],
#           label=fr'$t=${TAU*T[t4]:.2f} млн лет')

# ax_11.plot(x[1:], B_z4[t1][1:], '--', color='k',
#           label=f'$B_z$ для min')

ax_12.set_ylim(10**(-6), 10**(4))
ax_12.set_xlim(0.01, 1000)


ax_12.set_xlabel('$r,~{AU}$')
# ax_12.set_ylabel(r'$B_z, {G}$')

ax_12.set_yscale('log')
ax_12.set_xscale('log')

ax_12.set_title(fr'$t \approx {TAU*T[t2]:.2f}~Myr$')

ax_12.text(1.25, 0.1, '(b)', transform=ax_11.transAxes,
      fontsize=20, fontweight='bold', va='top')

# ax_12.legend(loc='best')

fig.subplots_adjust(wspace=0.17)


""" Профиль температуры """

# T = np.zeros(N)

# for i in range(N):
#     T[i] = 280*(x[i])**(-1/2)
    
# ax_11 = fig.add_subplot(1, 1, 1)
# ax_11.plot(x[1:], T[1:], '-', color=colors[2],
#           label=fr'T')


# # ax_11.plot(x[1:], B_z4[t1][1:], '--', color='k',
# #           label=f'$B_z$ для min')

# # ax_11.set_ylim(10**(-6), 10**(6))
# ax_11.set_xlim(0.01, 1000)


# ax_11.set_xlabel('$r,~{а.е.}$')
# ax_11.set_ylabel(r'$T, {К}$')

# # ax_11.set_yscale('log')
# ax_11.set_xscale('log')

# ax_11.set_title(fr'Температура')
# ax_11.legend(loc='best')

# """2"""
# ax_11 = fig.add_subplot(2, 2, 2)
# ax_11.plot(x[1:], X[t1][1:], '-', color='r'
#           )

# ax_11.plot(x[1:], X[t2][1:], '-', color='y'
#           )

# ax_11.plot(x[1:], X[t3][1:], '--', color='k'
#           )

# # ax_11.plot(x[1:], B_z4[t2][1:], '--', color='k',
# #           label=f'$B_z$ для min')

# # ax_11.set_ylim(10**(-6), 10**(6))
# ax_11.set_xlim(0.01, 1000)


# ax_11.set_xlabel('а.е.')
# ax_11.set_ylabel(r'x$')

# ax_11.set_yscale('log')
# ax_11.set_xscale('log')

# ax_11.set_title(fr'Степень ионизации')
# ax_11.legend(loc='best')

# """3"""
# ax_11 = fig.add_subplot(2, 2, 3)
# ax_11.plot(x[1:], A[t1][1:], '-', color='r')

# ax_11.plot(x[1:], A[t2][1:], '-', color='y')

# ax_11.plot(x[1:], A[t3][1:], '--', color='k')

# # ax_11.plot(x[1:], B_z4[t3][1:], '--', color='k',
# #            label=f'$B_z$ для min')

# # ax_11.set_ylim(10**(-6), 10**(6))
# ax_11.set_xlim(0.01, 1000)


# ax_11.set_xlabel('а.е.')
# ax_11.set_ylabel(r'$\alpha$')

# ax_11.set_yscale('log')
# ax_11.set_xscale('log')

# ax_11.set_title(fr'$\alpha$')
# ax_11.legend(loc='best')

# """4"""
# ax_11 = fig.add_subplot(2, 2, 4)
# # ax_11.plot(x[1:], B_z2[t1][1:], '-', color='r')

# ax_11.plot(x[1:], B_z2[t2][1:], '.', color='k')

# # ax_11.plot(x[1:], B_z2[t3][1:], '--', color='k')

# # ax_11.plot(x[1:], B_z4[t4][1:], '--', color='k',
# #           label=f'$B_z$ для min')

# # ax_11.set_ylim(10**(-6), 10**(6))
# ax_11.set_xlim(0.01, 1000)


# ax_11.set_xlabel('а.е.')
# ax_11.set_ylabel(r'$B_z, {Гс}$')

# ax_11.set_yscale('log')
# ax_11.set_xscale('log')

# ax_11.set_title(fr'Магнитное поле')
# ax_11.legend(loc='best')

# ax.legend(loc='best')
fig.show()
# plt.tight_layout()
# fig.savefig(f"Pringle_Sigma", orientation='landscape', dpi=300)
# fig.savefig(f"Pringle_X_1000.png", orientation='landscape', dpi=300)
# fig.savefig(f"2_DUST_Pringle_Bz.pdf", orientation='landscape', dpi=300)
fig.savefig(f"DUST_Pringle_Bz.pdf", orientation='landscape', dpi=300)
