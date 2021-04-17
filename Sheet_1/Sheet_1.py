import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (9, 5)

# Given
t_0 = 0
t_n = 10
h_1 = 10**(-3)
h_2 = 10**(-1)
m = 10**(-3)     # kg
k = 0.1          # N/m

time_1 = np.arange(t_0, t_n, h_1)
time_2 = np.arange(t_0, t_n, h_2)

# Initial conditions
x_0 = 0            # m
p_0 = 10**(-3)     # kg.m/s

def F(x):         
    return -k*x     # Force

def E(p, x):
    return (p**2/2*m) + (k*x**2/2)     # Total Energy = K.E + P.E

#----------- Explicit midpoint method -----------#

for h in [h_1, h_2]:
    if h == h_1:
        time = time_1
    else:
        time = time_2

    state_EM = np.zeros((time.shape[0], 2))     # To save position and momentum at every time step
    state_EM[0, :] = np.array([x_0, p_0])       # Column 0 = position, Column 1 = Momentum
    Energy_EM = np.zeros((time.shape[0],))
    Energy_EM[0] = E(p_0, x_0)

    # Heun's method
    # for i in range(1, time.shape[0]):
    #     pos_int = state[i-1, 0] + (h*(state[i-1, 1]/m))      # Intermdeiate position. Here, x'(t) = p(t)/m
    #     mom_int = state[i-1, 1] + (h*(-k*state[i-1, 0]))     # Interediate momentum. Here, p'(t) = -kx(t)

    #     # Position update
    #     pos_up = state[i-1, 0] + h/2*((state[i-1, 1]/m) + (mom_int/m))       
    #     state[i, 0] = pos_up

    #     # Momentum update
    #     mom_up = state[i-1, 1] + h/2*((-k*state[i-1, 0]) + (-k*pos_int))
    #     state[i, 1] = mom_up

    for i in range(1, time.shape[0]):
        state_EM[i, 0] = state_EM[i-1, 0] + (1/m*(state_EM[i-1, 1] + (F(state_EM[i-1, 0])*h/2))*h)     # Position update
        state_EM[i, 1] = state_EM[i-1, 1] + (F(state_EM[i-1, 0] + (state_EM[i-1, 1]/m*h/2))*h)
        Energy_EM[i] = E(state_EM[i, 1], state_EM[i, 0])

    np.save(f'State_EM_h={h}.npy', state_EM)
    np.save(f'Energy_EM_h={h}.npy', Energy_EM)
#-------------------------------------------------#

#----------- Velctiy-Verlet method -----------#
for h in [h_1, h_2]:
    if h == h_1:
        time = time_1
    else:
        time = time_2

    state_VV = np.zeros((time.shape[0], 2))     # To save position and momentum at every time step
    state_VV[0, :] = np.array([x_0, p_0])       # Column 0 = position, Column 1 = Momentum
    Energy_VV = np.zeros((time.shape[0],))
    Energy_VV[0] = E(p_0, x_0)

    for i in range(1, time.shape[0]):
        p = state_VV[i-1, 1] + (F(state_VV[i-1, 0])*h/2)
        state_VV[i ,0] = state_VV[i-1, 0] + (1/m*p*h)
        force = F(state_VV[i ,0])
        state_VV[i ,1] = p + (force*h/2)

    np.save(f'State_VV_h={h}.npy', state_VV)
    np.save(f'Energy_VV_h={h}.npy', Energy_VV)
#-------------------------------------------------#

# Plot position of (EM and VV) vs time and momentum of (EM and VV) vs time
state_EM = np.load('State_EM_h=0.001.npy')
state_VV = np.load('State_VV_h=0.001.npy')
plt.plot(time_1, state_EM[:, 0], label='Explicit Midpoint')
plt.plot(time_1, state_VV[:, 0], label='Velocity Verlet')
plt.xlabel('t in s')
plt.ylabel('x in m')
plt.legend()
plt.title('For h = 0.001')
plt.savefig('pos_vs_time_h=0.001.png')
plt.close()

plt.plot(time_1, state_EM[:, 1], label='Explicit Midpoint')
plt.plot(time_1, state_VV[:, 1], label='Velocity Verlet')
plt.xlabel('t in s')
plt.ylabel('p in kg.m/s')
plt.legend()
plt.title('For h = 0.001')
plt.savefig('mom_vs_time_h=0.001.png')
plt.close()
#--------------------------------------------------------------------------#
state_EM = np.load('State_EM_h=0.1.npy')
state_VV = np.load('State_VV_h=0.1.npy')
plt.plot(time_2, state_EM[:, 0], label='Explicit Midpoint')
plt.plot(time_2, state_VV[:, 0], label='Velocity Verlet')
plt.xlabel('t in s')
plt.ylabel('x in m')
plt.legend()
plt.title('For h = 0.1')
plt.savefig('pos_vs_time_h=0.1.png')
plt.close()

plt.plot(time_2, state_EM[:, 1], label='Explicit Midpoint')
plt.plot(time_2, state_VV[:, 1], label='Velocity Verlet')
plt.xlabel('t in s')
plt.ylabel('p in kg.m/s')
plt.legend()
plt.title('For h = 0.1')
plt.savefig('mom_vs_time_h=0.1.png')
plt.close()
#--------------------------------------------------------------------------#

# Analytical result plots

A = 10**(-1)
w = np.sqrt(k/m)

x_1 = A*np.sin(w*time_1)           # x = A.sin(wt)
x_2 = A*np.sin(w*time_2)

state_EM = np.load('State_EM_h=0.001.npy')
state_VV = np.load('State_VV_h=0.001.npy')
plt.plot(time_1, state_EM[:, 0], label='Explicit Midpoint')
plt.plot(time_1, state_VV[:, 0], label='Velocity Verlet')
plt.plot(time_1, x_1, label='Analytical')
plt.xlabel('t in s')
plt.ylabel('x in m')
plt.legend()
plt.title('For h = 0.001')
plt.savefig('pos(+ana)_vs_time_h=0.001.png')
plt.close()

state_EM = np.load('State_EM_h=0.1.npy')
state_VV = np.load('State_VV_h=0.1.npy')
plt.plot(time_2, state_EM[:, 0], label='Explicit Midpoint')
plt.plot(time_2, state_VV[:, 0], label='Velocity Verlet')
plt.plot(time_2, x_2, label='Analytical')
plt.xlabel('t in s')
plt.ylabel('x in m')
plt.legend()
plt.title('For h = 0.1')
plt.savefig('pos(+ana)_vs_time_h=0.1.png')
plt.close()
#--------------------------------------------------------------------------#
state_EM = np.load('State_EM_h=0.001.npy')
state_VV = np.load('State_VV_h=0.001.npy')
p_1 = A*w*m*np.cos(w*time_1)      # p = Awm.cos(wt)
p_2 = A*w*m*np.cos(w*time_2)

plt.plot(time_1, state_EM[:, 1], label='Explicit Midpoint')
plt.plot(time_1, state_VV[:, 1], label='Velocity Verlet')
plt.plot(time_1, p_1, label='Analytical')
plt.xlabel('t in s')
plt.ylabel('p in kg.m/s')
plt.legend()
plt.title('For h = 0.001')
plt.savefig('mom(+ana)_vs_time_h=0.001.png')
plt.close()

state_EM = np.load('State_EM_h=0.1.npy')
state_VV = np.load('State_VV_h=0.1.npy')
plt.plot(time_2, state_EM[:, 1], label='Explicit Midpoint')
plt.plot(time_2, state_VV[:, 1], label='Velocity Verlet')
plt.plot(time_2, p_2, label='Analytical')
plt.xlabel('t in s')
plt.ylabel('p in kg.m/s')
plt.legend()
plt.title('For h = 0.1')
plt.savefig('mom(+ana)_vs_time_h=0.1.png')
plt.close()
#--------------------------------------------------------------------------#

# Phase space plots
state_EM = np.load('State_EM_h=0.001.npy')
state_VV = np.load('State_VV_h=0.001.npy')
plt.plot(state_EM[:, 0], state_EM[:, 1], label='Explicit Midpoint')
plt.plot(state_VV[:, 0], state_VV[:, 1], label='Velocity Verlet')
plt.xlabel('x in m')
plt.ylabel('p in kg.m/s')
plt.legend()
plt.title('For h = 0.001')
plt.savefig('mom_vs_pos_h=0.001.png')
plt.close()
#--------------------------------------------------------------------------#
state_EM = np.load('State_EM_h=0.1.npy')
state_VV = np.load('State_VV_h=0.1.npy')
plt.plot(state_EM[:, 0], state_EM[:, 1], label='Explicit Midpoint')
plt.plot(state_VV[:, 0], state_VV[:, 1], label='Velocity Verlet')
plt.xlabel('x in m')
plt.ylabel('p in kg.m/s')
plt.legend()
plt.title('For h = 0.1')
plt.savefig('mom_vs_pos_h=0.1.png')
plt.close()

# Energy plots
Energy_EM = np.load('Energy_EM_h=0.001.npy')
Energy_VV = np.load('Energy_VV_h=0.001.npy')

Energy_EM_dev = Energy_EM - E(p_0, x_0)
Energy_VV_dev = Energy_VV - E(p_0, x_0)

plt.plot(time_1, Energy_EM_dev, label='Explicit Midpoint')
plt.plot(time_1, Energy_VV_dev, label='Velocity Verlet')
plt.xlabel('t in s')
plt.ylabel(r'$\Delta$E in kg$m^2/s^2$')
plt.legend()
plt.title('For h = 0.001')
plt.savefig('del_E_vs_time_h=0.001.png')
plt.close()

plt.plot(time_1, Energy_EM_dev/h_1, label='Explicit Midpoint')
plt.plot(time_1, Energy_VV_dev/h_1, label='Velocity Verlet')
plt.xlabel('t in s')
plt.ylabel(r'$\Delta$E/$\Delta$t in kg$m^2/s^3$')
plt.legend()
plt.title('For h = 0.001')
plt.savefig('del_E_del_t_vs_time_h=0.001.png')
plt.close()

plt.plot(time_1, Energy_EM_dev/h_1**2, label='Explicit Midpoint')
plt.plot(time_1, Energy_VV_dev/h_1**2, label='Velocity Verlet')
plt.xlabel('t in s')
plt.ylabel(r'$\Delta$E/$\Delta$$t^2$ in kg$m^2/s^4$')
plt.legend()
plt.title('For h = 0.001')
plt.savefig('del_E_del_t^2_vs_time_h=0.001.png')
plt.close()
#--------------------------------------------------------------------------#
Energy_EM = np.load('Energy_EM_h=0.1.npy')
Energy_VV = np.load('Energy_VV_h=0.1.npy')

Energy_EM_dev = Energy_EM - E(p_0, x_0)
Energy_VV_dev = Energy_VV - E(p_0, x_0)

plt.plot(time_2, Energy_EM_dev, label='Explicit Midpoint')
plt.plot(time_2, Energy_VV_dev, label='Velocity Verlet')
plt.xlabel('t in s')
plt.ylabel(r'$\Delta$E in kg$m^2/s^2$')
plt.legend()
plt.title('For h = 0.1')
plt.savefig('del_E_vs_time_h=0.1.png')
plt.close()

plt.plot(time_2, Energy_EM_dev/h_2, label='Explicit Midpoint')
plt.plot(time_2, Energy_VV_dev/h_2, label='Velocity Verlet')
plt.xlabel('t in s')
plt.ylabel(r'$\Delta$E/$\Delta$t in kg$m^2/s^3$')
plt.legend()
plt.title('For h = 0.1')
plt.savefig('del_E_del_t_vs_time_h=0.1.png')
plt.close()

plt.plot(time_2, Energy_EM_dev/h_2**2, label='Explicit Midpoint')
plt.plot(time_2, Energy_VV_dev/h_2**2, label='Velocity Verlet')
plt.xlabel('t in s')
plt.ylabel(r'$\Delta$E/$\Delta$$t^2$ in kg$m^2/s^4$')
plt.legend()
plt.title('For h = 0.1')
plt.savefig('del_E_del_t^2_vs_time_h=0.1.png')
plt.close()

