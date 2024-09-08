# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:37:39 2024

@author: Larrot
"""
import numpy as np
from CONSTS import Dead, t_end, M
from Smooth_alpha import S_ALPHA
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

# Global TIME constant
#Seconds
# TAU1 = (100*L)**2/(0.01*K*N_A/(MU*(G*M_sun)**(1/2))*100**(-1/2)*280*(100*L)**(3/2))
# TAU2 = (1*L)**2/(0.0001*K*N_A/(MU*(G*M_sun)**(1/2))*1**(-1/2)*280*(1*L)**(3/2))
#Years
# TAU  = TAU1/(3.15*10**7)
""" Ionization rate """

Ksi_0 = 10**(-17)
R_CR = 100

C1 = (N_A*G*M_sun/(K*MU))**(1/2)*L**(-3/2)

dust_size = 0.1

alpha_g0 = 4.5*10**(-17)*(dust_size/0.1)**(-1)

alpha_gm = 3*10**(-18)

def Diff(u, x, t):
    
    def T(x, u):
        T = 280*x**(-1/2)
        # T = 280*x**(-1/2)*u**(1/8)
    
        return T
    
    def Ksi(u):
        Ksi = Ksi_0*np.exp(-u/R_CR)
        return Ksi

    def n(u, x):
        n = C1*u*T(x, u)**(-1/2)*x**(-3/2)
        # n = C1*u*T(x)**(-1/2)*x**(-3/2)
        return n
    
    def alpha_r(x, u):
        if 0 <= T(x, u) <= 10**3:
            alpha_r = 2.07*10**(-11)*T(x, u)**(-1/2)*3
        elif 10**3 <= T(x, u) <= 10**4:
            alpha_r = 2.07*10**(-11)*T(x, u)**(-1/2)*1.5
        else:
            alpha_r = 0
        
        return alpha_r
            
    def alpha_g(x, u):
        if T(x, u) <= 150:
            alpha_g = alpha_g0
    
        elif 150 <= T(x, u) <= 400:
            alpha_g = -1/250*(alpha_g0-alpha_gm)*T(x, u) + \
                8/5*alpha_g0 - 3/5*alpha_gm
    
        elif 400 <= T(x, u) <= 1500:
            alpha_g = alpha_gm
    
        elif 1500 <= T(x, u) <= 2000:
            alpha_g = -1/500*alpha_gm*T(x, u)+4*alpha_gm
        else:
            alpha_g = 0
            
        return alpha_g
    
    
    def betta(u, x):
        if u >= 10**(-5):
            betta = (alpha_g(x, u)*n(u, x)+Ksi(u))/(2*alpha_r(x, u)*n(u, x))
        else:
            betta = 0
        return betta
        
    def gama(u,x):
        if u >= 10**(-5):
            gama = Ksi(u)/(alpha_r(x, u)*n(u, x))
        else:
            gama = 0
        return gama
        
    
    Xi = -betta(u,x) + (betta(u,x)**2+gama(u,x))**(1/2)
    
    if u >= 10**(-5):
        Xt = 1.8*10**(-11)*(T(x, u)/1000)**(3/4)*(Nu_K/10**(-7))**(0.5) \
            * (n(u,x)/10**(13))**(-0.5) \
            * np.exp(-25000/T(x, u))/(1.15*10**(-11))
    else:
        Xt = 0
        
    # Radiactive elements ionization
    Ksi_r = 2.6*10**(-19)

    def betta1(u, x):
        if u >= 10**(-5):
            betta = (alpha_g(x, u)*n(u, x)+Ksi_r)/(2*alpha_r(x, u)*n(u, x))
        else:
            betta = 0
        return betta
        
    def gama1(u,x):
        if u >= 10**(-5):
            gama = Ksi_r/(alpha_r(x, u)*n(u, x))
        else:
            gama = 0
        return gama
        
    
    Xi_r = -betta1(u,x) + (betta1(u,x)**2+gama1(u,x))**(1/2)
    
    
    X = Xi + Xt + Xi_r
    if Dead == 0:
        ALPHA = 0.01
    else:
        if X < 10**(-12):
            ALPHA = 10**(-4)
        else:
            ALPHA = 0.01
        # ALPHA = S_ALPHA(np.log10(X))
    # D = ALPHA/0.01*x**(2/2)    
    D = ALPHA/0.0001*x    
    
    return D
