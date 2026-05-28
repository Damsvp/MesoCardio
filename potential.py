import numpy as np


# ---- Kernel functions ------------------------------
def uniforme(x,d):
    if -d/2 < x < d/2: return 1/d
    else: return 0.0

def gaussienne(x, sigma):
    return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-x**2/(2*sigma**2))

# ---- Energy functions and derivatives ------------------------------

#Constant potential
def constant(k):
    return k 

def dconstant(k):
    return 0.0

#double well potential
def double_puits(x):
    return 0.25*x**4 - 0.5*x**2

def ddouble_puits(x):
    return x**3-x

#simple well potential
def simple_puits(x):
    return 0.5*x**2

def dsimple_puits(x):
    return x



