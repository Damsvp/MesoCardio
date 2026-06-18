import numpy as np

def uniforme(valeur):
    return lambda x, y: 1/valeur if -valeur/2 < x < valeur/2 else 0.0
 
def gaussienne(mu, sigma):
    return lambda x, y: np.exp(-((x - mu) / sigma) ** 2)

def lineaire(a, b):
    return lambda x, y: a * x + b

def sinus(A, omega, phi):
    return lambda x, y: A * np.sin(omega * x + phi)

def double_well(y, kpre, kpost, lsep, ypre, ypost, v) :
  if y < lsep :
    return (kpre/2)*(y - ypre)**2 + v
  else :
    return (kpost/2)*(y - ypost)**2
  
def d_double_well(y, kpre, kpost, lsep, ypre, ypost, v) :
  if y < lsep :
    return kpre*(y - ypre)
  else :
    return kpost*(y - ypost)
  


