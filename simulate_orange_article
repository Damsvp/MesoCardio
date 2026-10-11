import numpy as np
import matplotlib.pyplot as plt

def simulate_article_algorithm(t_max, dt, x_init, k_func, K_sampler):
    """
    Implementation of Algorithm 1 from the article (Numerical scheme with time step dt).
    
    Parameters:
    -----------
    t_max     : float, final simulation time (n_max * dt)
    dt        : float, discrete time step (\delta t)
    x_init    : float, initial value of the state variable X
    k_func    : function, global rate k(y(t)) depending on the environment/time
    K_sampler : function, samples the jump amplitude \Delta X_n given x and y(t)
    """
    # Maximum number of time steps
    n_max = int(t_max / dt)
    
    # --- Initial state ---
    n = 0
    c_J = 0.0
    e_J = np.random.exponential(1.0)  # e_J ~ E(1)
    
    # Initialize the trajectory array for X_n
    X = np.zeros(n_max + 1)
    X[0] = x_init
    
    # Define the discrete time grid
    time_grid = np.linspace(0, t_max, n_max + 1)
    
    # --- while n < n_max do ---
    while n < n_max:
        
        # --- while c_J < e_J do ---
        # This loop accumulates intensity until a jump threshold is crossed.
        # Added safety condition: ensure n does not exceed n_max.
        while (c_J < e_J) and (n < n_max):
            t_n = n * dt
            
            # Increment the cumulative intensity counter
            c_J += k_func(t_n) * dt
            
            # X remains constant while waiting for a jump event
            X[n+1] = X[n]
            
            n += 1
            
        # Break out of the loop if n reaches n_max without a jump
        if n >= n_max:
            break
            
        # --- Jump Event Occurs (c_J >= e_J) ---
        t_n = n * dt
        
        # Sample the jump amplitude \Delta X_n
        # dX ~ k^-1(y) * K(x, y) dx
        delta_X = K_sampler(X[n], t_n)
        
        # Apply the jump to the state variable
        X[n+1] = X[n] + delta_X
        
        # Reset the intensity counter and draw a new threshold
        c_J = 0.0
        e_J = np.random.exponential(1.0)  # e_J ~ E(1)
        
        # Final loop increment
        n += 1
        
    return time_grid[:n+1], X[:n+1]

# --- Example Profile and Plotting Routine ---
if __name__ == "__main__":
    # 1. Define rate function and jump sampler for the example
    # Example: A sinusoidal time-varying global rate k(t)
    global_rate = lambda t: 5.0 * (1.0 + np.sin(t)) 
    
    # Spatial jump sampler (e.g., normal distribution centered around a step of 1.5)
    spatial_sampler = lambda x, t: np.random.normal(1.5, 0.3)
    
    # 2. Simulation parameters
    T_horizon = 15.0
    time_step = 0.01  # \delta t
    initial_value = 0.0
    
    # 3. Run the simulation
    times, X_trajectory = simulate_article_algorithm(
        t_max=T_horizon, 
        dt=time_step, 
        x_init=initial_value, 
        k_func=global_rate, 
        K_sampler=spatial_sampler
    )
    
    # 4. Generate Plot Panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Panel 1: State Trajectory X_n Over Time
    ax1.step(times, X_trajectory, where='post', color='darkblue', lw=2, label='Jump Process $X_n$')
    ax1.set_ylabel('$X_n$')
    ax1.set_title('Inhomogeneous Process Simulation (Article Time-Rescaling Method)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel 2: Continuous Background Intensity k(t)
    t_continuous = np.linspace(0, T_horizon, 500)
    ax2.plot(t_continuous, global_rate(t_continuous), color='orangered', lw=2, label='Global Rate $k(y(t))$')
    ax2.set_xlabel('Time ($t$)')
    ax2.set_ylabel('Intensity Rate ($k$)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()