# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:37:39 2024

@author: Larrot
"""
import numpy as np
from CONSTS import Dead, t_end, M, tau
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

sigma = 5.67*10**(-5)



def Diff(u, x, t):
    
    if t == 0:
        def T():
            T = 280*x**(-1/2)
            # T = 280*x**(-1/2)*u**(1/8)
        
            return T
    else:
        def T():
            D_0 = Diff(u, x, t-tau)    
            # T_eff= 9/8*D_0*5.11*10**(12)/sigma*u*G*M_sun/(x*L)**3
            T_eff= 9/8*D_0*10**6/sigma*u*G*M_sun/(x*L)**3
            
            # T = (3/8*0.1*u*T_eff)**(1/4)
            # T = (3/8*0.1*u*T_eff)**(1/4)+280*x**(-1/2)
            T = (3/8*0.1*u*T_eff)**(1/4)+10
            
            return T
            
        
    
    def Ksi():
        Ksi = Ksi_0*np.exp(-u/R_CR)
        return Ksi

    def n():
        n = C1*u*T()**(-1/2)*x**(-3/2)
        # n = C1*u*T(x)**(-1/2)*x**(-3/2)
        return n
    
    def alpha_r():
        if 0 <= T() <= 10**3:
            alpha_r = 2.07*10**(-11)*T()**(-1/2)*3
        elif 10**3 <= T() <= 10**4:
            alpha_r = 2.07*10**(-11)*T()**(-1/2)*1.5
        else:
            alpha_r = 0
        
        return alpha_r
            
    def alpha_g():
        if T() <= 150:
            alpha_g = alpha_g0
    
        elif 150 <= T() <= 400:
            alpha_g = -1/250*(alpha_g0-alpha_gm)*T() + \
                8/5*alpha_g0 - 3/5*alpha_gm
    
        elif 400 <= T() <= 1500:
            alpha_g = alpha_gm
    
        elif 1500 <= T() <= 2000:
            alpha_g = -1/500*alpha_gm*T()+4*alpha_gm
        else:
            alpha_g = 0
            
        return alpha_g
    
    
    def betta():
        if u >= 10**(-5):
            betta = (alpha_g()*n()+Ksi())/(2*alpha_r()*n())
        else:
            betta = 0
        return betta
        
    def gama():
        if u >= 10**(-5):
            gama = Ksi()/(alpha_r()*n())
        else:
            gama = 0
        return gama
        
    
    Xi = -betta() + (betta()**2+gama())**(1/2)
    if u >= 10**(-5):
        Xt = 1.8*10**(-11)*(T()/1000)**(3/4)*(Nu_K/10**(-7))**(0.5) \
            * (n()/10**(13))**(-0.5) \
            * np.exp(-25000/T())/(1.15*10**(-11))
    else:
        Xt = 0
        
    # Radiactive elements ionization
    Xr = 2.6*10**(-19)

    
    
    X = Xi + Xt + Xr
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






