# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 21:04:25 2024

@author: Larrot
"""
import CONSTS as C
import numpy as np
from scipy.optimize import fsolve


def T(x, u, t, nu):
    # T_star = 280*x**(-1/2)
    # T = T_star


    D_0 = nu    
    # T_eff= 9/8*D_0*5.11*10**(12)/sigma*u*G*M_sun/(x*L)**3
    T_eff= 9/8*D_0*10**6/C.sigma*u*C.G*C.M_sun/(x*C.L)**3
    
    # T = (3/8*0.1*u*T_eff)**(1/4)
    T_irr = 280*x**(-1/2)
    
    T = (3/8*0.1*u*T_eff)**(1/4) + T_irr
    

    return T

def Ksi(u):
    Ksi = C.Ksi_0*np.exp(-u/C.R_CR)
    return Ksi

def n(x, u, t, nu):
    n = C.C1*u*T(x, u, t, nu)**(-1/2)*x**(-3/2)
    # n = C1*u*T(x)**(-1/2)*x**(-3/2)
    return n

def alpha_r(x, u, t, nu):
    if 0 <= T(x, u, t, nu) <= 10**3:
        alpha_r = 2.07*10**(-11)*T(x, u, t, nu)**(-1/2)*3
    elif 10**3 <= T(x, u, t, nu) <= 10**4:
        alpha_r = 2.07*10**(-11)*T(x, u, t, nu)**(-1/2)*1.5
    else:
        alpha_r = 0
    
    return alpha_r
        
def alpha_g(x, u, t, nu):
    if T(x, u, t, nu) <= 150:
        alpha_g = C.alpha_g0

    elif 150 <= T(x, u, t, nu) <= 400:
        alpha_g = -1/250*(C.alpha_g0-C.alpha_gm)*T(x, u, t, nu) + \
            8/5*C.alpha_g0 - 3/5*C.alpha_gm

    elif 400 <= T(x, u, t, nu) <= 1500:
        alpha_g = C.alpha_gm

    elif 1500 <= T(x, u, t, nu) <= 2000:
        alpha_g = -1/500*C.alpha_gm*T(x, u, t, nu)+4*C.alpha_gm
    else:
        alpha_g = 0
        
    return alpha_g


def betta(u, x, t, nu):
    if u >= 10**(-5):
        betta = (alpha_g(x, u, t, nu)*n(x, u, t, nu)+Ksi(u))/(2*alpha_r(x, u, t, nu)*n(x, u, t, nu))
    else:
        betta = 0
    return betta
    
def gama(x, u, t, nu):
    if u >= 10**(-5):
        gama = Ksi(u)/(alpha_r(x, u, t, nu)*n(x, u, t, nu))
    else:
        gama = 0
    return gama

# Radiactive elements ionization

def betta1(x, u, t, nu):
    if u >= 10**(-5):
        betta = (alpha_g(x, u, t, nu)*n(x, u, t, nu)+C.Ksi_r)/(2*alpha_r(x, u, t, nu)*n(x, u, t, nu))
    else:
        betta = 0
    return betta
    
def gama1(x, u, t, nu):
    if u >= 10**(-5):
        gama = C.Ksi_r/(alpha_r(x, u, t, nu)*n(x, u, t, nu))
    else:
        gama = 0
    return gama

def Ionization_Rate(x, u, t, nu):
    

    Xi = -betta(x, u, t, nu) + (betta(x, u, t, nu)**2+gama(x, u, t, nu))**(1/2)
    
    if u >= 10**(-5):
        Xt = 1.8*10**(-11)*(T(x, u, t, nu)/1000)**(3/4)*(C.Nu_K/10**(-7))**(0.5) \
            * (n(x, u, t, nu)/10**(13))**(-0.5) \
            * np.exp(-25000/T(x, u, t, nu))/(1.15*10**(-11))
    else:
        Xt = 0
        
    
    Xi_r = -betta1(x, u, t, nu) + (betta1(x, u, t, nu)**2+gama1(x, u, t, nu))**(1/2)
    
    
    return Xi + Xt + Xi_r

def alpha(x, u, t, nu):
    if Ionization_Rate(x, u, t, nu) < 10**(-12):
        ALPHA = 10**(-4)
    else:
        ALPHA = 0.01
    D = ALPHA/0.0001*x    

    return D


def f(nu):
    #Example values:
    # Radial distance
    x = 10
    # Surface density
    u = 100
    # Time
    t = 0.1
    return nu - alpha(x, u, t, nu)*T(x, u, t, nu)


# Zero approximation
x0 = 100

# Solve the equation
solution = fsolve(f, x0)
print("Solution:", solution)