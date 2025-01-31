import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import matplotlib
#increase the font size
matplotlib.rcParams.update({'font.size': 18})

jax.config.update("jax_enable_x64", True)

# --------------------
# Target Distribution: 10D Gaussian
# --------------------
D = 10  # Dimension
true_mean = jnp.zeros(D)
cov = jnp.eye(D)  # Identity covariance

def log_prob(params):
    return -0.5 * jnp.dot(params, params)  # -0.5 * x^T x

# --------------------
# Improved Metropolis-Hastings
# --------------------
def mcmc_sampler(key, n_steps, step_size):
    samples = jnp.zeros((n_steps, D))
    current = 5.0 * jax.random.normal(key, (D,))  # Start far from mean (5σ)
    accepted = 0
    
    for i in tqdm(range(n_steps), desc="MCMC"):
        key, subkey = jax.random.split(key)
        proposal = current + jax.random.normal(subkey, (D,)) * step_size
        
        log_ratio = log_prob(proposal) - log_prob(current)
        if jnp.log(jax.random.uniform(subkey)) < log_ratio:
            current = proposal
            accepted += 1
            
        samples = samples.at[i].set(current)
    
    acceptance_rate = accepted / n_steps
    return samples, acceptance_rate

# --------------------
# Corrected HMC Sampler
# --------------------
def hmc_sampler(key, n_steps, step_size, n_leapfrog=20):
    samples = jnp.zeros((n_steps, D))
    current = 5.0 * jax.random.normal(key, (D,))  # Start far from mean (5σ)
    grad_log_prob = jax.grad(log_prob)
    
    for i in tqdm(range(n_steps), desc="HMC"):
        key, subkey = jax.random.split(key)
        q = current.copy()
        p = jax.random.normal(subkey, (D,))  # Resample momentum
        
        # Leapfrog integration
        p += 0.5 * step_size * grad_log_prob(q)
        for _ in range(n_leapfrog - 1):
            q += step_size * p
            p += step_size * grad_log_prob(q)
        q += step_size * p
        p += 0.5 * step_size * grad_log_prob(q)
        
        # Metropolis acceptance
        current_energy = log_prob(current) - 0.5 * jnp.dot(p, p)
        proposed_energy = log_prob(q) - 0.5 * jnp.dot(p, p)
        
        if jnp.log(jax.random.uniform(subkey)) < (proposed_energy - current_energy):
            current = q
            
        samples = samples.at[i].set(current)
    
    return samples

# --------------------
# Robust Convergence Check
# --------------------
def analyze_convergence(samples, true_mean, threshold=0.8, window=100):
    """Check convergence over a rolling window"""
    mse = jnp.mean((samples - true_mean)**2, axis=1)
    # Require threshold met for 'window' consecutive steps
    converged = (jnp.convolve(mse < threshold, jnp.ones(window), mode='valid') >= window)
    return jnp.argmax(converged) + window if jnp.any(converged) else -1, mse

# --------------------
# Experiment Setup
# --------------------
def main():
    # Hyperparameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--step_size_mcmc", type=float, default=0.1)
    parser.add_argument("--run_mcmc", default=False, action="store_true")

    args = parser.parse_args()
    n_steps = args.n_steps
    step_size_mcmc = args.step_size_mcmc  # Tuned for ~25% acceptance
    step_size_hmc = 0.3   # Larger step size for HMC
    n_leapfrog = 20       # Longer trajectories

    if args.run_mcmc:

        # Run samplers
        key = jax.random.PRNGKey(42)
        
        # MCMC
        key, subkey = jax.random.split(key)
        mcmc_samples, mcmc_accept = mcmc_sampler(subkey, n_steps, step_size_mcmc)
        
        # HMC
        key, subkey = jax.random.split(key)
        hmc_samples = hmc_sampler(subkey, n_steps, step_size_hmc, n_leapfrog)
        
        # Analyze convergence
        mcmc_step, mcmc_mse = analyze_convergence(mcmc_samples, true_mean)
        hmc_step, hmc_mse = analyze_convergence(hmc_samples, true_mean)
        
        print(f"MCMC converged in {mcmc_step} steps (Acceptance: {mcmc_accept:.2%})")
        print(f"HMC converged in {hmc_step} steps")

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(mcmc_mse, label=f'MCMC (step size={step_size_mcmc})')
        plt.plot(hmc_mse, label=f'HMC (step size={step_size_hmc}, L={n_leapfrog})')
        plt.axhline(0.8, color='k', linestyle='--', label='Convergence threshold')
        plt.yscale('log')
        plt.xlabel('Step')
        plt.ylabel('MSE')
        plt.title('10D Gaussian Convergence Comparison (Fixed)')
        plt.legend()
        # plt.show()
        plt.savefig('convergence.png')

        #save the samples
        np.save("mcmc_samples.npy", mcmc_samples)
        np.save("hmc_samples.npy", hmc_samples)
    print('loading samplies')
    mcmc_samples = np.load("mcmc_samples.npy")
    hmc_samples = np.load("hmc_samples.npy")

    def animate_convergence_hmc_mcmc(hmc_samples, mcmc_samples, true_mean, save_path="hmc_mcmc_convergence.gif"):
        """
        Create an animation of HMC and MCMC convergence with new arrows generated at each frame.
        Includes an MSE vs step plot with a scatter point for the current frame.
        """
        # Project samples and true mean onto the first two dimensions
        projected_hmc_samples = hmc_samples[:, :2]
        projected_mcmc_samples = mcmc_samples[:, :2]
        projected_mean = true_mean[:2]

        # Calculate MSE

        mcmc_step, mcmc_mse = analyze_convergence(mcmc_samples, true_mean)
        hmc_step, hmc_mse = analyze_convergence(hmc_samples, true_mean)

        # Define grid for contour plot
        x = jnp.linspace(-6, 6, 100)
        y = jnp.linspace(-6, 6, 100)
        X, Y = jnp.meshgrid(x, y)
        Z = jnp.exp(-0.5 * (X**2 + Y**2))  # Gaussian PDF in 2D

        # Create figure and subplots
        fig, (ax_mse, ax_contour) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1]})
        x_axis = jnp.arange(0, len(mcmc_mse))
        # Left plot: MSE vs step
        ax_mse.plot(x_axis,mcmc_mse, label="MCMC MSE", color="red")
        ax_mse.plot(x_axis,hmc_mse, label="HMC MSE", color="blue")
        point_mse_mcmc = ax_mse.scatter([], [], color="white", label="MCMC Current Step",zorder=2)
        point_mse_hmc = ax_mse.scatter([], [], color="white", label="HMC Current Step",zorder=2)
        ax_mse.set_xlabel("Step")
        ax_mse.set_ylabel("MSE")
        ax_mse.set_yscale("log")
        ax_mse.set_title("MSE vs Step")
        ax_mse.legend()
        ax_mse.set_xscale("log")
        ax_mse.axhline(0.8, color='white', linestyle='--', label='Convergence threshold')
        #make the background black
        fig.patch.set_facecolor('black')
        #set axis color to white
        ax_mse.spines['bottom'].set_color('white')
        ax_mse.spines['top'].set_color('white')
        ax_mse.spines['right'].set_color('white')
        ax_mse.spines['left'].set_color('white')
        ax_mse.xaxis.label.set_color('white')
        ax_mse.yaxis.label.set_color('white')
        ax_mse.tick_params(axis='x', colors='white')
        ax_mse.tick_params(axis='y', colors='white')
        ax_mse.title.set_color('white')

        ax_contour.spines['bottom'].set_color('white')
        ax_contour.spines['top'].set_color('white')
        ax_contour.spines['right'].set_color('white')
        ax_contour.spines['left'].set_color('white')
        ax_contour.xaxis.label.set_color('white')
        ax_contour.yaxis.label.set_color('white')
        ax_contour.tick_params(axis='x', colors='white')
        ax_contour.tick_params(axis='y', colors='white')
        ax_contour.title.set_color('white')
        #reduce the space between the plots
        fig.subplots_adjust(wspace=0.2)


        # Right plot: Contour and samples
        ax_contour.contour(X, Y, Z, levels=10, cmap="viridis", alpha=0.5)
        ax_contour.scatter(*projected_mean, color="red", label="True Mean", zorder=3)
        scat_hmc = ax_contour.scatter([], [], color="blue", alpha=0.7, label="HMC Samples", zorder=2)
        scat_mcmc = ax_contour.scatter([], [], color="red", alpha=0.7, label="MCMC Samples", zorder=2)
        ax_contour.legend()
        ax_contour.set_xlim(-6, 6)
        ax_contour.set_ylim(-6, 6)
        ax_contour.set_title("Convergence (2D Projection)")
        skip = 5

        def update(frame):
            ax_mse.patch.set_facecolor('black')
            ax_contour.patch.set_facecolor('black')
            frame *= skip
            """Update function for animation."""
            # Update MSE scatter points
            # if frame < len(mse_mcmc) and frame < len(mse_hmc):
            # point_mse_mcmc.set_data(frame, mcmc_mse[frame])
            # point_mse_hmc.set_data(frame, hmc_mse[frame])
            point_mse_mcmc.set_offsets((frame, np.array(mcmc_mse)[frame]))
            point_mse_hmc.set_offsets((frame, np.array(hmc_mse)[frame]))
            scat_hmc.set_offsets(projected_hmc_samples[:frame])
            scat_mcmc.set_offsets(projected_mcmc_samples[:frame])
            print(frame)

            # Update samples
            if frame > 0:
                # Add arrows for HMC
                i = frame - skip
                dx_hmc = projected_hmc_samples[i + skip, 0] - projected_hmc_samples[i, 0]
                dy_hmc = projected_hmc_samples[i + skip, 1] - projected_hmc_samples[i, 1]
                ax_contour.arrow(
                    projected_hmc_samples[i, 0],
                    projected_hmc_samples[i, 1],
                    dx_hmc,
                    dy_hmc,
                    color="blue",
                    alpha=0.5,
                    head_width=0.2,
                    head_length=0.2,
                    length_includes_head=True,
                )

                # Add arrows for MCMC
                dx_mcmc = projected_mcmc_samples[i + skip, 0] - projected_mcmc_samples[i, 0]
                dy_mcmc = projected_mcmc_samples[i + skip, 1] - projected_mcmc_samples[i, 1]
                ax_contour.arrow(
                    projected_mcmc_samples[i, 0],
                    projected_mcmc_samples[i, 1],
                    dx_mcmc,
                    dy_mcmc,
                    color="green",
                    alpha=0.5,
                    head_width=0.2,
                    head_length=0.2,
                    length_includes_head=True,
                )

            return point_mse_mcmc, point_mse_hmc, scat_hmc, scat_mcmc

        ani = FuncAnimation(
            fig, update, frames=300, interval=200, blit=False
        )

        ani.save(save_path, fps=30, writer="pillow")
        print(f"Animation saved to {save_path}")
        plt.close(fig)

    # Call the function
    animate_convergence_hmc_mcmc(hmc_samples, mcmc_samples, true_mean, save_path="hmc_mcmc_convergence_with_mse.gif")


if __name__ == "__main__":
    main()