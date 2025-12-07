from SRtools import SR_hetro as srh
import numpy as np
from numba import jit
from joblib import Parallel, delayed
from SRtools import deathTimesDataSet as dtds
import os
from SRtools import sr_mcmc as srmc


jit_nopython = True

"""
After implementing your class, change sr_mcmc.model so it calls your class instead of the default one and uses your metric function.
"""


class SR_Noises(srh.SR_Hetro):
    def __init__(self, eta, beta, kappa, epsilon, xc, npeople, nsteps, t_end,
                 eta_var = 0, beta_var = 0, kappa_var =0, epsilon_var =0, xc_var =0,
                   t_start=0, tscale='years', external_hazard=np.inf, time_step_multiplier=1, parallel=False, bandwidth=3, heun=False, noise_type='normal'):
        """
        If you want to add parameters to the __init__ method, you can do so here before the call to super().__init__. if you add beta2 as aparameter for example then
        add self.beta2=beta2 here.
        """
        self.noise_type = noise_type

        #this is the call to my class, do not modify it. also, do not earase any of the parameters I cal here unless you give them a default value or somehting
        super().__init__(eta, beta, kappa, epsilon, xc, npeople, nsteps, t_end, eta_var, beta_var, kappa_var, epsilon_var, xc_var, t_start, tscale, external_hazard, time_step_multiplier, parallel, bandwidth, heun)




    def calc_death_times(self):
        s = len(self.t)
        dt = self.t[1]-self.t[0]
        sdt = np.sqrt(dt)
        t = self.t
        # Convert noise_type string to integer code for numba compatibility
        noise_type_map = {'normal': 0, 'log_normal': 1, 'poisson': 2, 'multiplicative_normal_beta': 3}
        noise_type_code = noise_type_map.get(self.noise_type, 0)  # Default to normal if unknown
        if self.parallel:
            death_times, events = death_times_euler_brownian_bridge_parallel(s,dt,t,self.eta,self.eta_var,self.beta,self.beta_var,self.kappa,self.kappa_var,self.epsilon, self.epsilon_var,self.xc,self.xc_var,sdt,self.npeople,self.external_hazard,self.time_step_multiplier,noise_type_code)
        else:
            death_times, events = death_times_euler_brownian_bridge(s,dt,t,self.eta,self.eta_var,self.beta,self.beta_var,self.kappa,self.kappa_var,self.epsilon, self.epsilon_var,self.xc,self.xc_var,sdt,self.npeople,self.external_hazard,self.time_step_multiplier,noise_type_code)

        return np.array(death_times), np.array(events)
    

def getSr_noises(theta, n=25000,nsteps=6000,t_end=110, external_hazard = np.inf,time_step_multiplier =1, npeople =None, parallel = False,eta_var = 0, beta_var = 0, epsilon_var = 0, xc_var = 0.2, kappa_var = 0, hetro = True, bandwidth = 3, noise_type = 'normal'):
    """
    Get a SR model with noises.
    
    Get a SR model with noises.
    Args:
        theta: The parameters of the model.
        n: The number of people to simulate.
        nsteps: The number of time steps to simulate.
        t_end: The end time of the simulation.
        external_hazard: The external hazard.
        time_step_multiplier: The time step multiplier.
        npeople: The number of people to simulate.
        parallel: Whether to run the simulation in parallel.
        eta_var: The variance of the eta parameter.
        beta_var: The variance of the beta parameter.
        epsilon_var: The variance of the epsilon parameter.
        xc_var: The variance of the xc parameter.
        kappa_var: The variance of the kappa parameter.
        hetro: Whether to use hetro.
        bandwidth: The bandwidth of the kernel.
        noise_type: The type of noise to use.
    """
    if npeople is not None:
        n = npeople
    eta = theta[0]
    beta = theta[1]
    epsilon = theta[2]
    xc = theta[3]
    if not hetro:
        eta_var =0
        beta_var =0
        epsilon_var =0
        xc_var =0
        kappa_var =0

    if external_hazard is None or external_hazard == 'None':
        external_hazard = np.inf
    
    sim = SR_Noises(eta=eta,beta=beta,epsilon=epsilon,xc=xc,
                   eta_var=eta_var,beta_var=beta_var,kappa_var=kappa_var,epsilon_var=epsilon_var,xc_var=xc_var,
                   kappa=0.5,npeople=n,nsteps=nsteps,t_end=t_end,external_hazard=external_hazard, time_step_multiplier=time_step_multiplier,
                     parallel=parallel, bandwidth=bandwidth, noise_type=noise_type)
    
    return sim









# Euler with Brownian Bridge method
@jit(nopython=jit_nopython)
def death_times_euler_brownian_bridge(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                                     epsilon0, epsilon_var, xc0, xc_var, sdt, npeople,
                                     external_hazard=np.inf, time_step_multiplier=1, noise_type=0):
    """
    Euler method with Brownian bridge crossing detection.
    This method uses the standard Euler scheme but adds Brownian bridge
    crossing probability tests to detect barrier crossings between time steps.
    
    noise_type codes:
    - 0: normal (additive normal noise, default)
    - 1: log_normal (log-normal noise)
    - 2: poisson (Poisson noise)
    - 3: multiplicative_normal_beta (multiplicative normal noise on beta term)
    """
    death_times = []
    events = []
    ndt = dt / time_step_multiplier
    nsdt = sdt / np.sqrt(time_step_multiplier)
    constant_hazard = np.isfinite(external_hazard)
    if constant_hazard:
        chance_to_die_externally = np.exp(-external_hazard) * ndt
    
    for person in range(npeople):
        x = 0.0
        j = 0
        eta = eta0 * np.random.normal(1.0, eta_var)
        beta = beta0 * np.random.normal(1.0, beta_var)
        kappa = kappa0 * np.random.normal(1.0, kappa_var)
        epsilon = epsilon0 * np.random.normal(1.0, epsilon_var)
        xc = xc0 * np.random.normal(1.0, xc_var)
        sqrt_2epsilon = np.sqrt(2 * epsilon)
        crossed = False
        
        while j < s - 1 and not crossed:
            for _ in range(time_step_multiplier):
                # Standard Euler step
                drift = eta * t[j] - beta * x / (x + kappa)
                
                # Apply noise based on noise_type code
                if noise_type == 0:
                    # Additive normal noise (default)
                    noise = sqrt_2epsilon * np.random.normal()
                    x_new = x + ndt * drift + noise * nsdt
                elif noise_type == 1:
                    # Log-normal noise: exp(N(0, sqrt(2*epsilon)))
                    # Mean of log-normal should be 1, variance should match epsilon
                    # For log-normal: if Y = exp(X) where X ~ N(mu, sigma^2), then E[Y] = exp(mu + sigma^2/2)
                    # To have E[Y] = 1, we need mu = -sigma^2/2
                    # Variance of Y = exp(2*mu + sigma^2) * (exp(sigma^2) - 1)
                    # For small variance approximation: use mu = -epsilon*ndt, sigma = sqrt(2*epsilon*ndt)
                    log_noise_mean = -epsilon * ndt
                    log_noise_std = sqrt_2epsilon * nsdt
                    noise_multiplier = np.exp(np.random.normal(log_noise_mean, log_noise_std))
                    x_new = x + ndt * drift + (noise_multiplier - 1.0) * x
                elif noise_type == 2:
                    # Poisson noise: scale Poisson random variable
                    # Use lambda = epsilon * ndt to match variance
                    poisson_lambda = 2*epsilon * ndt
                    if poisson_lambda > 0:
                        poisson_noise = np.random.poisson(poisson_lambda)
                        # Scale to match the variance of normal noise
                        noise = (poisson_noise) #- poisson_lambda)
                    else:
                        noise = 0.0
                    x_new = x + ndt * drift + noise 
                elif noise_type == 3:
                    # Multiplicative normal noise on beta term, this results in dx/dt = eta * t - beta(1-noise) * x / (x + kappa)
                    # This is equivalent to dx/dt = eta * t - beta * x / (x + kappa) + beta * noise * x / (x + kappa) so we assume the noise is sqrt(2*epsilon)/beta
                    noise = sqrt_2epsilon * (x / (x + kappa))* np.random.normal()
                    x_new = x + ndt * drift + noise * nsdt
                else:
                    # Default to normal if unknown noise_type
                    noise = sqrt_2epsilon * np.random.normal()
                    x_new = x + ndt * drift + noise * nsdt
                
                x_new = max(x_new, 0.0)
                
                # Check external hazard
                if constant_hazard and np.random.rand() < chance_to_die_externally:
                    x = xc
                    crossed = True
                    break
                
                # Direct crossing check
                if x_new >= xc:
                    x = x_new
                    crossed = True
                    break
                
                # Brownian bridge crossing test if not crossed directly
                if (x < xc) and (x_new < xc) and (x > 0*kappa):
                    dx1 = xc - x
                    dx2 = xc - x_new
                    if dx1 > 0.0 and dx2 > 0.0:
                        # Brownian bridge crossing probability
                        # P = exp(-2 * (xc - x) * (xc - x_new) / (2 * epsilon * ndt))
                        var = 2.0 * epsilon * ndt
                        if var > 0.0:
                            p_cross = np.exp(-2.0 * dx1 * dx2 / var)
                            if np.random.rand() < p_cross:
                                x = xc
                                crossed = True
                                break
                
                x = x_new
            j += 1
        
        death_times.append(j * dt)
        if crossed or x >= xc:
            events.append(1)
        else:
            events.append(0)
    
    return np.array(death_times), np.array(events)

# Parallel version of Euler with Brownian Bridge method
def death_times_euler_brownian_bridge_parallel(s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
                                              epsilon0, epsilon_var, xc0, xc_var, sdt, npeople,
                                              external_hazard=np.inf, time_step_multiplier=1, noise_type=0, n_jobs=-1, chunk_size=1000):
    """
    Parallel version of death_times_euler_brownian_bridge.
    Splits npeople into chunks and runs death_times_euler_brownian_bridge on each chunk in parallel.
    
    noise_type codes:
    - 0: normal (additive normal noise, default)
    - 1: log_normal (log-normal noise)
    - 2: poisson (Poisson noise)
    - 3: multiplicative_normal_beta (multiplicative normal noise on beta term)
    """
    from joblib import Parallel, delayed
    import numpy as np

    def worker(npeople_chunk, s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
               epsilon0, epsilon_var, xc0, xc_var, sdt, external_hazard, time_step_multiplier, noise_type):
        # Call the numba-jitted function for this chunk
        return death_times_euler_brownian_bridge(
            s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
            epsilon0, epsilon_var, xc0, xc_var, sdt, npeople_chunk,
            external_hazard, time_step_multiplier, noise_type
        )

    # Split npeople into chunks
    n_chunks = npeople // chunk_size
    remainder = npeople % chunk_size
    chunk_sizes = [chunk_size] * n_chunks
    if remainder > 0:
        chunk_sizes.append(remainder)

    results = Parallel(n_jobs=n_jobs)(
        delayed(worker)(
            n_chunk, s, dt, t, eta0, eta_var, beta0, beta_var, kappa0, kappa_var,
            epsilon0, epsilon_var, xc0, xc_var, sdt, external_hazard, time_step_multiplier, noise_type
        ) for n_chunk in chunk_sizes if n_chunk > 0
    )

    # Concatenate results
    death_times = np.concatenate([res[0] for res in results])
    events = np.concatenate([res[1] for res in results])
    return death_times, events



