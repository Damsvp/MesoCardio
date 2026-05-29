import numpy as np
from matplotlib import pyplot as plt
import random



N = 2 #number of heads we are going to simulate
delta = [5*(random.random() - 0.5) for i in range(N)] #random position shifts for the heads
v = 2 #viscosity of the sarcomere's surroundings
F = 5 #force exerted on the actin filament
d = 100 #distance between two actin attachment sites
b = 0.1 #inverse temperature


l = 20 #length of the interval in which the myosin head can detach
dx = 0.1 #length discretization
npos = int(l/dx) #number of possible positions
positions = [k*dx - l/2 for k in range(npos)]   #all possible positions


Temp = 10
b = 1/Temp #inverse temperature
ny = 1 #viscosity coefficients
nx = 1

T = 100  #max time of the simulation
dt = 0.1    #time step
nsteps = int(T/dt)    #number of steps in the simulation

K01 = lambda x, y, s : np.exp(-(s/50)**2)    #direct transition rates
K10 = lambda x, y, s : 0.04
#note that K10 is supposed to vanish outside of [-l/2, l/2]

w0 = lambda x, y : 0.1     #energy landscape and derivatives for detached head
dxw0 = lambda x, y : 0
dyw0 = lambda x, y : 0

w1 = lambda x, y : 10*(x/50)**2      #energy landscape and derivatives for attached head
dxw1 = lambda x, y : 0.4*x/50
dyw1 = lambda x, y : 0

muT = 0.05   #shift due to ATP consumption

h = 10       #caracteristic length scale used to make the expression of reverse transition rates homogeneous
#Modifier à terme pour que l'intégrale fasse 1 je crois NONONONONON
#c'est un taux de transition, pas une probabilité

K01rev = lambda x, y, s : 0*(1/h)*K01(x, y, s)*np.exp(b*(w1(s, y) - w0(x, y)))     #reverse transition rates
K10rev = lambda x, y, s : h*K10(x, y, s)*np.exp(b*((w0(x, y) - muT) - w1(s, y)))


alpha = np.array([[0 for t in range(nsteps)] for i in range(N)])      #actual description of the stochastic process
s = np.array([5 for t in range(nsteps)])
X = np.array([[s[0] + delta[i] for t in range(nsteps)] for i in range(N)])
Y = np.array([[0 for t in range(nsteps)] for i in range(N)])

for t in range(nsteps - 1):

  #s dynamics
  s[t + 1] = s[t] + F*dt - (dt/v)*sum([alpha[i, t]*dxw1(s[t] + delta[i], Y[i, t]) for i in range(N)])
  #torus condition on s
  print((dt/v)*sum([alpha[i, t]*dxw1(s[t] + delta[i], Y[i, t]) for i in range(N)]))
  if s[t + 1] < -d/2 :
    s[t + 1] += d
  elif s[t + 1] > d/2 :
    s[t + 1] -= d

  for i in range(N) :
    #generating random numbers to decide if a jump takes place between t and t+dt
    x = random.random()
    y = random.random()
    #X and Y brownian motions
    Bx = random.gauss(0, 1)
    By = random.gauss(0, 1)

    
    if alpha[i, t] == 0 :
      #Y dynamics
      Y[i, t + 1] = Y[i, t] - ny*dyw0(X[i, t], Y[i, t]) + np.sqrt(2*ny*dt/b)*By

      #alpha and X dynamics
      if x < K01(X[i, t], Y[i, t], s[t])*dt :
        alpha[i, t + 1] = 1
        X[i, t + 1] = s[t + 1] + delta[i]
      elif y < K10rev(X[i, t], Y[i, t], s[t] + delta[i])*dt :
        alpha[i, t + 1] = 1
        X[i, t + 1] = s[t + 1] + delta[i]
      else :
        alpha[i, t + 1] = 0
        X[i, t + 1] = X[i, t] - nx*dxw0(X[i, t], Y[i, t]) + np.sqrt(2*nx*dt/b)*Bx

    if alpha[i, t] == 1 :
      #Y dynamics
      Y[i, t + 1] = Y[i, t] - ny*dyw1(X[i, t], Y[i, t]) + np.sqrt(2*ny*dt/b)*By

      #alpha and X dynamics
      prob = [(K10(j*dx, Y[i, t], s[t] + delta[i]) + K01rev(j*dx, Y[i, t], s[t] + delta[i]))*dx for j in range(npos)] #space discretized probabilities of detachment
      k = sum(prob)  #calculate the overall detachment rate
      if x < k*dt :
        alpha[i, t + 1] = 0
        X[i, t + 1] = s[t] + delta[i] + np.random.choice(positions, p = (1/k)*np.array(prob))
      else :
        alpha[i, t + 1] = 1
        X[i, t + 1] = s[t + 1]


#Visualization of the results
fig, axs = plt.subplots(nrows=1, ncols=2)

axs[0].plot([dt*t for t in range(nsteps)], s)
#axs[0].plot([dt*t for t in range(nsteps)], X[0, :])
axs[1].plot([dt*t for t in range(nsteps)], Y[0, :])

plt.show()