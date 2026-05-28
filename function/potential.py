import numpy as np

def uniforme(valeur):
    return lambda x: 1/valeur if -valeur/2 < x < valeur/2 else 0.0

def gaussienne(mu, sigma):
    return lambda x: np.exp(-((x - mu) / sigma) ** 2)

def lineaire(a, b):
    return lambda x: a * x + b

def sinus(A, omega, phi):
    return lambda x: A * np.sin(omega * x + phi)


