# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 19:30:17 2023

@author: Larrot
"""

from scipy import integrate

from matplotlib import pyplot as plt
import numpy as np
from CONSTS import t_end, M, Dead, tau
# Время в миллионах лет
TAU = 1.39

DATA = np.loadtxt(f"DATA_{t_end}_{M}_{Dead}.txt")

Phi_z1 = np.loadtxt(f'Phi_1_{t_end}_{M}_{Dead}.txt')
Phi_z2 = np.loadtxt(f'Phi_2_{t_end}_{M}_{Dead}.txt')
Phi_z3 = np.loadtxt(f'Phi_3_{t_end}_{M}_{Dead}.txt')

# Степень вязкости

gamma = 1
#  Внутреняя граница
R_in = -2
# R_in = 0.1

# Внешняя граница
R_out = 3
# R_out = 1000

M_sun = 1.98*10**33

L = 1.5*10**13

# Количество узлов
N = 100

# Пространственная сетка
# x = np.linspace(R_in, R_out, N)
x = np.logspace(R_in, R_out, N)

N = 100

x_s = np.logspace(R_in, R_out, N+1)

# x = np.logspace(R_in, R_out, N)
x = np.array([(x_s[i+1]+x_s[i])/2 for i in range(N)])

Mass = np.zeros(M)


for t in range(M):
    for i in range(N):
        DATA[t][i] = DATA[t][i]*x[i]
    


C = 2*3.1415*L**2/M_sun
for t in range(M):
    Mass[t] = C*integrate.trapezoid(DATA[t][:], x[:])



fig = plt.figure(figsize=(6, 6))
# fig = plt.figure()
# fig.subplots_adjust(hspace=0.6, wspace=0.4)

"""График поверхностной плотности"""
plt.rcParams.update({'font.size': 15})
# fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# ax.text(0.05, 0.05, '(б)', transform=ax.transAxes, fontsize=16, fontweight='bold', va='bottom')


# fig = plt.figure()
# fig.subplots_adjust(hspace=0.6, wspace=0.4)



# # ax.set_xlim(0.1, 1000)
# ax.set_ylim(6.5*10**(-2), 8*10**(-2))
# # ax.set_ylim(10**(1), 3*10**(2))
# # ax.set_ylim(0.5*10**(-2), 2*10**(-1))
# # ax.set_ylim(0.5*10**(-2), 2*10**(1))
# # ax.set_xlabel(r'$r_{ а.е.}$ ')

ax.set_xlabel(r'T, млн. лет')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_ylabel(r'M/$M_{\odot}$')
# ax.set_title(fr'Масса диска для различных $t$. $N={N}$, $\nu \sim$'+ '$r^{%s}$' %gamma)
# ax.set_title(fr'Масса диска')
# ax.set_title(fr'Масса: Поток (амбиполярная диффузия)')
# ax.set_title(fr'Масса: Поток (вмороженное поле)')

# ax.set_xlim(0.1, 1000)
# # ax.set_ylim(10**(-2), 3*10**(1))
# # ax.set_ylim(10**(1), 3*10**(2))
# # ax.set_ylim(0.5*10**(-2), 2*10**(-1))
# # ax.set_ylim(0.5*10**(-2), 2*10**(1))
# # ax.set_xlabel(r'$r_{ а.е.}$ ')
# ax.set_xlabel(r't, млн лет')
# # ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel(r'$\Phi, 10^{29}~Мкс$')
ax.set_title(fr'Магнитный поток')

T = np.array([tau*i for i in range(M)])


# ax2 = ax.twiny()
# # ax2.set_xscale('log')
# ax2.set_xlabel(r'Myr')

T_Myr = np.array([T[i]*TAU for i in range(M)])


MP3 = np.zeros(M)

for i in range(M):
    MP3[i] = round(Mass[i]/Phi_z3[i], 10)


MP1 = np.zeros(M)

for i in range(M):
    MP1[i] = round(Mass[i]/Phi_z1[i], 10)

# ax.plot(T_Myr, Mass, '-',  color='k')

# ax.plot(T_Myr, MP1, '-',  color='k')
# ax.plot(T_Myr[2:], MP3[2:], '-',  color='k')

# ax.plot(T, Mass, '-',  color='k')
ax.plot(T_Myr[1:], Phi_z1[1:], '--',  color='r', label = 'F')
# ax.plot(T_Myr[1:], Phi_z2[1:], '--',  color='k', label = 'AD')
ax.plot(T_Myr[1:], Phi_z3[1:], '-',  color='k', label = 'AD')
# ax2.plot(T_Myr, np.zeros(M),)






ax.legend(loc='best')
fig.show()
plt.tight_layout()
# fig.savefig(f"Pringle_Sigma_R_{gamma}_N_{N}.png", orientation='landscape', dpi=300)
# fig.savefig(f"Pringle_Phi_{Dead}_t_{t_end}.png", orientation='landscape', dpi=300)
# fig.savefig(f"Pringle_Phi_{Dead}_t_{t_end}.tif", orientation='landscape', dpi=300)
# fig.savefig(f"Pringle_Phi.png", orientation='landscape', dpi=300)
# fig.savefig(f"Pringle_M.Phi.eps", orientation='landscape', dpi=300)
fig.savefig(f"Pringle_M.Phi.pdf", orientation='landscape')
