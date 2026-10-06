import numpy as np
import matplotlib.pyplot as plt

def simulate_myosin_trajectory(T, s0, v, f_func, g_func, f_max, g_max):
    """
    Simulates the alternating inhomogeneous Poisson process for a myosin head.
    """
    get_s = lambda t: s0 + v * t #constant speed
    t = 0.0
    alpha = 0  # 0: Detached, 1: Attached
    
    history_time = [0.0]
    history_state = [alpha]
    
    while t < T:
        if alpha == 0:
            # Propose Attachment Event (0 -> 1)
            u = np.random.exponential(1.0 / f_max)
            t += u
            if t > T: break
            
            current_s = get_s(t)
            acceptance_prob = f_func(current_s) / f_max #acceptance probability for each candidate that arrives at time \[r\]
            if np.random.uniform(0, 1) <= acceptance_prob:
                alpha = 1
                history_time.append(t)
                history_state.append(alpha)
        else:
            # Propose Detachment Event (1 -> 0)
            u = np.random.exponential(1.0 / g_max)
            t += u
            if t > T: break
            
            current_s = get_s(t)
            acceptance_prob = g_func(current_s) / g_max #same for the detachment case 
            if np.random.uniform(0, 1) <= acceptance_prob:
                alpha = 0
                history_time.append(t)
                history_state.append(alpha)
                
    history_time.append(T)
    history_state.append(history_state[-1])
    return np.array(history_time), np.array(history_state)

# --- Example Profiles and Plotting Routine ---
if __name__ == "__main__":
    # Physical profile parameters
    f_rate_s = lambda s: 15.0 * np.exp(- (s**2) / 0.5)
    g_rate_s = lambda s: 2.0 + 4.0 * (s**2)
    MAX_F, MAX_G = 15.0, 25.0
    
    T_horizon = 10.0
    s0_pos, velocity = -2.0, 0.4
    
    # Run simulation
    times, states = simulate_myosin_trajectory(T_horizon, s0_pos, velocity, f_rate_s, g_rate_s, MAX_F, MAX_G)
    
    # --- Generate Plot Panels ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Panel 1: State Over Time alpha(t)
    ax1.step(times, states, where='post', color='darkmagenta', lw=2.5, label=r'Myosin State $\alpha_t$')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Detached (0)', 'Attached (1)'])
    ax1.set_title('Alternating Inhomogeneous Poisson Process Trajectory', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Panel 2: Continuous Intensity Measures and Overlying Position
    t_continuous = np.linspace(0, T_horizon, 500)
    s_continuous = s0_pos + velocity * t_continuous
    f_vals = f_rate_s(s_continuous)
    g_vals = g_rate_s(s_continuous)
    
    ax2.plot(t_continuous, f_vals, color='forestgreen', lw=2, label=r'Attachment Rate $f(s(t))$')
    ax2.plot(t_continuous, g_vals, color='crimson', lw=2, label=r'Detachment Rate $g(s(t))$')
    ax2.set_xlabel('Time ($t$)')
    ax2.set_ylabel('Intensity Rate ($\lambda$)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Overlay relative position on a twin axis to track strain alignment
    ax3 = ax2.twinx()
    ax3.plot(t_continuous, s_continuous, color='gray', linestyle='--', alpha=0.7, label='Position $s(t)$')
    ax3.set_ylabel('Site Strain/Position ($s$)')
    ax3.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()