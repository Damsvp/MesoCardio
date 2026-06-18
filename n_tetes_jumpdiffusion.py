import numpy as np
from matplotlib import pyplot as plt
import random



N = 1 #number of heads we are going to simulate
delta = [0 for i in range(N)] #random position shifts for the heads
v = 2 #viscosity of the sarcomere's surroundings
F = 0 #force exerted on the actin filament
d = 38 #distance between two actin attachment sites

k1pre = 5.6 #stiffness pre power stroke (pN/nm)
k1post = 1.4 #stiffness post power stroke
l1 = 1.42 #separation between the two wells (nm)
y1pre = 0 #position of the minimum of the pre power stroke well
y1post = 11 #position of the minimum of the post power stroke well
v1 = (k1post/2)*(l1 - y1post)**2 - (k1pre/2)*(l1 - y1pre)**2 #energy difference between the two states at the position of the minimum of the post power stroke well

k0pre = 11.45 #stiffness pre power stroke
k0post = 0.45 #stiffness post power stroke
l0 = 1.42 #separation between the two wells (nm)
y0pre = 0 #position of the minimum of the pre power stroke well
y0post = 6 #position of the minimum of the post power stroke well
v0 = (k0post/2)*(l0 - y0post)**2 - (k0pre/2)*(l0 - y0pre)**2 #energy difference between the two states at the position of the minimum of the post power stroke well


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


l = 5 #length of the interval in which the myosin head can detach
dx = 0.01 #length discretization
npos = int(l/dx) #number of possible positions
positions = [k*dx - l/2 for k in range(npos)]   #all possible positions


kBTemp = 4.14 #kB*temperature in zJ (T = 300K)
b = 1/kBTemp #inverse temperature
ny = 10.288 #inverse viscosity coefficients (fluidity) (ms-1.pN-1.nm)
nx = 10.288

T = 1500  #max time of the simulation, in ms
dt = 0.01    #time step (ms)
nsteps = int(T/dt)    #number of steps in the simulation

kmax = 1.21 
alphay = 8
alphas = 8
sl01 = 3.82
sr01 = 3.82

k = 1.34 #stiffness of the myosin

k10 = 1 #transition rates
k01 = 1

def heaviside(s) :
  if s < 0 :
    return 0
  else :
    return 1

K01 = lambda x, y, s : k01*heaviside(l0 - y)*heaviside(l/2 - abs(x - s)) #0.1 + kmax*(1 - np.tanh(alphay*(y - l0)))*(0.5*(1 - heaviside(s))*(1 + np.tanh(alphas*(s + sl01))) + 0.5*heaviside(s)*(1 - np.tanh(alphas*(s - sr01))))   #direct transition rates
K10 = lambda x, y, s : k10*heaviside(l/2 - abs(x - s))*(heaviside(y - l0) + 10*heaviside(3 - s))
#note that K10 is supposed to vanish outside of [-l/2, l/2]

E = 80 #energy shift in zJ
sbar0 = 1.2 #shift of the potential, in nm
sbar1 = 1.2

w0 = lambda x, y : E + ((k/2)*(x + y)**2 + double_well(y + sbar0, k0pre, k0post, l0, y0pre, y0post, v0))     #energy landscape and derivatives for detached head
dxw0 = lambda x, y : k*(x + y)
dyw0 = lambda x, y : k*(x + y) + d_double_well(y + sbar0, k0pre, k0post, l0, y0pre, y0post, v0)

w1 = lambda x, y : ((k/2)*(x + y)**2 + double_well(y + sbar1, k1pre, k1post, l1, y1pre, y1post, v1))     #energy landscape and derivatives for attached head
dxw1 = lambda x, y : k*(x + y)
dyw1 = lambda x, y : k*(x + y) + d_double_well(y + sbar1, k1pre, k1post, l1, y1pre, y1post, v1)

muT = 100   #shift due to ATP consumption, in zJ


plt.plot([dx*t - d/4 for t in range(4*npos)], [w0(0, dx*t - d/4) for t in range(4*npos)], label = 'detached')
plt.plot([dx*t - d/4 for t in range(4*npos)], [w1(0, dx*t - d/4) for t in range(4*npos)], label = 'attached')
plt.plot([dx*t - d/4 for t in range(4*npos)], [w0(0, dx*t - d/4) - muT for t in range(4*npos)], label = 'detached')

plt.show()




h = 11       #caracteristic length scale (nm) used to make the expression of reverse transition rates homogeneous
#Modifier à terme pour que l'intégrale fasse 1 je crois NONONONONON
#c'est un taux de transition, pas une probabilité

K01rev = lambda x, y, s : (1/h)*K01(x, y, s)*np.exp(b*(w1(s, y) - w0(x, y)))     #reverse transition rates
K10rev = lambda x, y, s : h*K10(x, y, s)*np.exp(b*((w0(x, y) - muT) - w1(s, y)))


alpha = np.array([[0 for t in range(nsteps)] for i in range(N)])      #actual description of the stochastic process
s = np.array([5 for t in range(nsteps)], dtype = float)
X = np.array([[s[0] + delta[i] for t in range(nsteps)] for i in range(N)], dtype = float)
Y = np.array([[0 for t in range(nsteps)] for i in range(N)], dtype = float)

for t in range(nsteps - 1):
  #Compute the rates K and K_rev
  K01t= K01(X[i, t], Y[i, t], s[t])
  K10revt= K10rev(X[i, t], Y[i, t], s[t] + delta[i])

  K10t= K10(X[i, t], Y[i, t], s[t])
  K01revt=K01rev(X[i, t], Y[i, t], s[t] + delta[i])

  #Clock
  c01 = [0.0 for i in range(N)]
  c10 = [0.0 for i in range(N)]
  e01 = -np.log(np.random.random(size=N))   # realisation of an exponential of parameter 1
  e10 = -np.log(np.random.random(size=N))

  #Clock
  c01rev = [0.0 for i in range(N)]
  c10rev = [0.0 for i in range(N)]
  e01rev = -np.log(np.random.random(size=N))   # realisation of an exponential of parameter 1
  e10rev = -np.log(np.random.random(size=N))

  #s dynamics
  s[t + 1] = s[t] + F*dt/v - (dt/v)*sum([alpha[i, t]*dxw1(s[t] + delta[i], Y[i, t]) for i in range(N)])
  #torus condition on s
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
      Y[i, t + 1] = Y[i, t] - dt*ny*dyw0(X[i, t], Y[i, t]) + np.sqrt(2*ny*dt/b)*By

      #clocks increases:
      c01[i]+= K01t*dt
      c10rev[i]+= K10revt*dt

      #alpha and X dynamics
      if c01[i] > e01[i] or c10rev[i] > e10rev[i]:
        alpha[i, t + 1] = 1
        X[i, t + 1] = s[t + 1] + delta[i]
        c01[i]=0.0
        e01[i]=-np.log(np.random.random())
        c10rev[i]=0.0
        e10rev[i]=-np.log(np.random.random())
      else :
        alpha[i, t + 1] = 0
        X[i, t + 1] = X[i, t] - dt*nx*dxw0(X[i, t], Y[i, t]) + np.sqrt(2*nx*dt/b)*Bx

    if alpha[i, t] == 1 :
      #Y dynamics
      #print(t, Y[i, t], X[i, t])
      Y[i, int(t + 1)] = Y[i, t] - dt*ny*dyw1(X[i, t], Y[i, t]) + np.sqrt(2*ny*dt/b)*By

      #clocks increases:
      c01rev[i]+= K01revt*dt
      c10[i]+= K10t*dt

      #alpha and X dynamics
      prob = [(K10(j*dx, Y[i, t], s[t] + delta[i]) + K01rev(j*dx, Y[i, t], s[t] + delta[i]))*dx for j in range(npos)] #space discretized probabilities of detachment
      detach_rate = sum(prob)  #calculate the overall detachment rate
      if c01rev[i] > e01rev[i] or c10[i] > e10[i]:   # guard: no detachment possible, skip the jump
        alpha[i, t + 1] = 0
        X[i, t + 1] = s[t] + delta[i] + np.random.choice(positions, p = (1/detach_rate)*np.array(prob))
        c01rev[i]=0.0
        e01rev[i]=-np.log(np.random.random())
        c10[i]=0.0
        e10[i]=-np.log(np.random.random())
      else :
        alpha[i, t + 1] = 1
        X[i, t + 1] = s[t + 1] + delta[i]


#Visualization of the results
fig, axs = plt.subplots(nrows=1, ncols=2)

axs[0].plot([dt*t for t in range(nsteps)], s)
#axs[0].plot([dt*t for t in range(nsteps)], X[0, :])
axs[1].plot([dt*t for t in range(nsteps)], Y[0, :])

plt.show()