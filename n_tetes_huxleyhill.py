import numpy as np
from matplotlib import pyplot as plt
import random

N = 10 #number of heads we are going to simulate
delta = [5*(random.random() - 0.5) for i in range(N)] #random position shifts for the heads
v = 1 #viscosity of the sarcomere's surroundings
F = 0 #force exerted on the actin filament
d = 100 #distance between two actin attachment sites
b = 0.1 #inverse temperature

T = 1000  #max time of the simulation
dt = 0.1    #time step
nsteps = int(T/dt)    #number of steps in the simulation

k01 = lambda s : np.exp(-(s/50)**2)    #direct transition rates
k10 = lambda s : 1

w0 = lambda s : 0.1     #energy landscape and derivatives for detached head
dw0 = lambda s : 0

w1 = lambda s : 5*(s/50)**2      #energy landscape and derivatives for attached head
dw1 = lambda s : 0.1*s/50

muT = 0.05   #shift due to ATP hydrolysis

k01rev = lambda s : k01(s)*np.exp(b*(w1(s) - w0(s)))     #reverse transition rates
k10rev = lambda s : k10(s)*np.exp(b*((w0(s) - muT) - w1(s)))



alpha = np.array([[0 for t in range(nsteps)] for i in range(N)])      #actual description of the stochastic process
s = [0 for t in range(nsteps)]



for t in range(nsteps - 1):

  #s dynamics
  s[t + 1] = s[t] + F*dt - (dt/v)*sum([alpha[i, t]*dw1(s[t] + delta[i]) for i in range(N)])
  #torus condition on s
  if s[t + 1] < -d/2 :
    s[t + 1] += d
  elif s[t + 1] > d/2 :
    s[t + 1] -= d

  #alpha dynamics
  for i in range(N) :  
    #generating random numbers to decide if a jump takes place between t and t+dt
    x = random.random()
    y = random.random()

    if alpha[i, t] == 0 :
      if x < k01(s[t])*dt :
        alpha[i, t + 1] = 1
      elif y < k10rev(s[t])*dt :
        alpha[i, t + 1] = 1
      else :
        alpha[i, t + 1] = 0

    if alpha[i, t] == 1 :
      if x < k10(s[t])*dt :
        alpha[i, t + 1] = 0
      elif y < k01rev(s[t])*dt :
        alpha[i, t + 1] = 0
      else :
        alpha[i, t + 1] = 1
      

print(alpha[0, :])

#Visualization of the results
plt.plot([dt*t for t in range(nsteps)], s)
#plt.plot([dt*t for t in range(nsteps)], [sum(alpha[i, t] for i in range(N)) for t in range(nsteps)])
plt.show()