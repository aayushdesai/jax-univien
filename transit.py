import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

@partial(jax.jit, static_argnames=('R_star', 'M_star'))
def transit_model(params, t, R_star=1.0, M_star=1.0):
    """
    Parameters:
    params = [P (days), RpRs, inc (deg), u1, u2, t0]
    t: observation times (days)
    Returns: Normalized flux
    """
    # Unpack parameters
    P, RpRs, inc_deg, u1, u2, t0 = params
    inc = jnp.deg2rad(inc_deg)
    
    # ----- Orbital Mechanics -----
    G = 6.67430e-11  # m^3 kg^-1 s^-2
    P_sec = P * 86400.0      # Convert days to seconds
    # Semi-major axis in meters, from Kepler's third law (assuming M_star in solar masses)
    a = ((G * M_star * 1.9885e30 * P_sec**2) / (4.0 * jnp.pi**2))**(1/3)
    # Normalize by stellar radius
    a_Rs = a / (R_star * 6.957e8)
    
    # Orbital phase
    phase = 2.0 * jnp.pi * (t - t0) / P
    x = a_Rs * jnp.sin(phase)
    z = a_Rs * jnp.cos(phase) * jnp.cos(inc)
    r_proj = jnp.sqrt(x**2 + z**2)  # planet center, in stellar radii
    
    # ----- Limb Darkening (simple approximation) -----
    # mu = sqrt(1 - r^2) if r < 1, else 0
    mu = jnp.sqrt(jnp.clip(1.0 - r_proj**2, 0.0, 1.0))
    limb_dark = 1.0 - u1*(1.0 - mu) - u2*(1.0 - mu)**2
    
    # ----- Smooth Transit Boundary -----
    # Instead of a step function, use a logistic function around r_proj = 1 + RpRs
    # This ensures we have a continuous derivative near the boundary.
    # Larger 'k' -> sharper transition (but still continuous)
    k = 50.0
    boundary = 1.0 + RpRs
    # logistic: 0 if r_proj >> boundary, and ~1 if r_proj << boundary
    transit_factor = 0.5 * (1.0 - jnp.tanh(k*(r_proj - boundary)))
    # Then scale by area ratio
    area_ratio = transit_factor * (RpRs**2)
    
    # Final flux
    flux = 1.0 - area_ratio * limb_dark
    
    return flux

def test_gradients():
    # True parameters [P, Rp/Rs, inc, u1, u2, t0]
    true_params = jnp.array([5.0, 0.1, 87.0, 0.4, 0.2, 0.0])
    
    # Generate synthetic data around transit
    t = jnp.linspace(-0.3, 0.3, 1000)
    flux_true = transit_model(true_params, t)
    noise = jax.random.normal(jax.random.PRNGKey(42), t.shape) * 1e-4
    flux_obs = flux_true + noise

    # Mean-squared-error loss
    def loss(params):
        flux_pred = transit_model(params, t)
        return jnp.mean((flux_pred - flux_obs)**2)

    # Autodiff gradient
    grad_analytic = jax.grad(loss)(true_params)

    # Finite-difference gradient
    eps = 1e-8
    grad_fd = jnp.zeros_like(true_params)
    for i in range(len(true_params)):
        params_high = true_params.at[i].add(eps)
        params_low  = true_params.at[i].add(-eps)
        loss_high = loss(params_high)
        loss_low  = loss(params_low)
        grad_fd = grad_fd.at[i].set((loss_high - loss_low) / (2.0 * eps))

    # Calculate relative error
    rel_error = jnp.abs(grad_analytic - grad_fd) / (jnp.abs(grad_fd) + 1e-12)
    
    print("Param\t Autodiff\t FiniteDiff\t Rel.Error")
    for i in range(len(true_params)):
        print(f"{i}\t {grad_analytic[i]:.3e}\t {grad_fd[i]:.3e}\t {rel_error[i]:.3e}")

def plot_transit():
    """
    Compare flux curves for two sets of parameters:
    'good' (close to true) and 'bad' (slightly off inc).
    """
    params_good = jnp.array([5.0, 0.1, 87.0, 0.4, 0.2, 0.0])
    params_bad  = params_good.at[2].set(85.0)  # reduce inclination
    
    t = jnp.linspace(-0.3, 0.3, 1000)
    flux_good = transit_model(params_good, t)
    flux_bad  = transit_model(params_bad, t)
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, flux_good, label='Inclination = 87°')
    plt.plot(t, flux_bad, '--', label='Inclination = 85°')
    plt.xlabel('Time from transit center [days]')
    plt.ylabel('Normalized flux')
    plt.title('Transit Light Curve Sensitivity to Inclination')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_gradients()
    plot_transit()