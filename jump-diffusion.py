import numpy as np
from matplotlib import pyplot as plt
import random
import scipy.integrate as integrate

N = 10 #number of heads we are going to simulate
n = 1 #viscosity of the sarcomere's surroundings
F = 1 #force exerted on the actin filament

d = 100 #length of the interval in which the myosin head can move around the actin attachment site
dx = 0.1 #length discretization
npos = int(d/dx) #number of possible positions
positions = [k*dx for k in range(npos)]   #all possible positions

b = 0.00001 #inverse temperature
ny = 1 #viscosity coefficients
nx = 1

T = 10  #max time of the simulation
dt = 0.1    #time step
nsteps = int(T/dt)    #number of steps in the simulation

K01 = lambda x, y, s : np.exp(-((s-50)/50)**2)    #direct transition rates
K10 = lambda x, y, s : 1

w0 = lambda x, y : 0.1     #energy landscape and derivatives for detached head
dxw0 = lambda x, y : 0
dyw0 = lambda x, y : 0

w1 = lambda x, y : 10*((x-50)/50)**2      #energy landscape and derivatives for attached head
dxw1 = lambda x, y : 0
dyw1 = lambda x, y : 0

muT = 0.05   #shift due to ATP consumption

h = 1       #caracteristic length scale used to make the expression of reverse transition rates homogeneous

K01rev = lambda x, y, s : (1/h)*K01(x, y, s)*np.exp(b*(w1(s, y) - w0(x, y)))     #reverse transition rates
K10rev = lambda x, y, s : h*K10(x, y, s)*np.exp(b*((w0(x, y) - muT) - w1(s, y)))



alpha = [0 for t in range(nsteps)]      #actual description of the stochastic process
X = [0 for t in range(nsteps)]
Y = [0 for t in range(nsteps)]
s = [0 for t in range(nsteps)]



for t in range(nsteps - 1):
  #generating random numbers to decide if a jump takes place between t and t+dt
  x = random.random()
  y = random.random()
  #X and Y brownian motions
  Bx = random.gauss(0, 1)
  By = random.gauss(0, 1)
    
  if alpha[t] == 0 :
    #Y dynamics
    Y[t + 1] = Y[t] - ny*dyw0(X[t], Y[t]) + np.sqrt(2*ny*dt/b)*By

    #alpha and X dynamics
    if x < K01(X[t], Y[t], s[t])*dt :
      alpha[t + 1] = 1
      X[t + 1] = s[t + 1]
    elif y < K10rev(X[t], Y[t], s[t])*dt :
      alpha[t + 1] = 1
      X[t + 1] = s[t + 1]
    else :
      alpha[t + 1] = 0
      X[t + 1] = X[t] - nx*dxw0(X[t], Y[t]) + np.sqrt(2*nx*dt/b)*Bx

  if alpha[t] == 1 :
    #Y dynamics
    Y[t + 1] = Y[t] - ny*dyw1(X[t], Y[t]) + np.sqrt(2*ny*dt/b)*By

    #alpha and X dynamics
    prob = [(K10(j*dx, Y[t], s[t]) + K01rev(j*dx, Y[t], s[t]))*dx for j in range(npos)] #space discretized probabilities of detachment
    k = sum(prob)  #calculate the overall detachment rate
    if x < k*dt :
      alpha[t + 1] = 0
      X[t + 1] = -(d/2) + np.random.choice(positions, p = (1/k)*np.array(prob))
    else :
      alpha[t + 1] = 1
      X[t + 1] = s[t + 1]

print(alpha)

plt.plot([t for t in range(nsteps)], X)
plt.show()