import numpy as np

def uniforme(valeur):
    return lambda x, y: 1/valeur if -valeur/2 < x < valeur/2 else 0.0
 
def gaussienne(mu, sigma):
    return lambda x, y: np.exp(-((x - mu) / sigma) ** 2)

def lineaire(a, b):
    return lambda x, y: a * x + b

def sinus(A, omega, phi):
    return lambda x, y: A * np.sin(omega * x + phi)


