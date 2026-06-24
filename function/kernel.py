import numpy as np

def uniforme(valeur):
    return lambda x, y, s : 1/valeur if -valeur/2 < x < valeur/2 and -valeur/2 < s < valeur/2 else 0.0

def exponentiel(lam):
    return lambda x, y, s: np.exp(-lam * x)