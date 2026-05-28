def uniforme(x,d):
    if -d/2 < x < d/2: return 1/d
    else: return 0.0

def gaussienne(x, sigma):
    return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-x**2/(2*sigma**2))

