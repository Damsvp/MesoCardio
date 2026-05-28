import numpy as np

def uniforme(valeur):
    return lambda x: 1/valeur if -valeur/2 < x < valeur/2 else 0.0

def exponentiel(lam):
    return lambda x: np.exp(-lam * x)