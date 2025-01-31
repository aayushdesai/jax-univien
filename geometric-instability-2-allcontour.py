import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import matplotlib 
matplotlib.rcParams.update({'font.size': 16})

jax.config.update("jax_enable_x64", True)

def kepler_eqn(M, e, tol=1e-10):
    """
    Solve Kepler's equation: M = E - e*sin(E) for E, given mean anomaly M.
    We'll do a simple Newton's method in JAX.
    """
    def newton_step(E, _):
        f = E - e*jnp.sin(E) - M
        fp = 1.0 - e*jnp.cos(E)
        return E - f/fp, 0

    E0 = M
    E, _ = jax.lax.scan(newton_step, E0, xs=None, length=100)
    return E

def true_anomaly(M, e):
    E = kepler_eqn(M, e)
    nu = 2.0 * jnp.arctan(
        jnp.sqrt((1.0 + e)/(1.0 - e)) * jnp.tan(0.5 * E)
    )
    return nu

def rv_model(t, params):
    """
    Single-planet RV model, ignoring small relativistic effects.
    params = (P, K, e, omega, gamma)
    """
    P, K, e, omega, gamma = params
    n = 2.0 * jnp.pi / P
    M_t = n * t
    nu_t = true_anomaly(M_t, e)
    rv = gamma + K*(jnp.cos(nu_t + omega) + e*jnp.cos(omega))
    return rv

def generate_synthetic_data(params_true, t_array, noise_std=2.0):
    key = jax.random.PRNGKey(42)
    rv_clean = rv_model(t_array, params_true)
    noise = noise_std * jax.random.normal(key, shape=t_array.shape)
    return rv_clean + noise

def rv_loss(params, t_array, rv_obs):
    rv_pred = rv_model(t_array, params)
    return jnp.mean((rv_pred - rv_obs)**2)

grad_loss = jax.grad(rv_loss, argnums=0)

def set_bkgs_black(fig, ax):
    """Set figure/axes background to black with white labeling."""
    fig.patch.set_facecolor('black')
    ax.patch.set_facecolor('black')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.title.set_color('white')
    return fig, ax

def build_param_ranges(final_params, idx, n_points=50,param_history=None):
    """
    Return a jnp.linspace for parameter at final_params[idx].
    We define a +/- 30% range around the final param. 
    Special handling for eccentricity idx=2 (clamp to [0,1)).
    If final_params[idx] ~ 0, we expand a bit.
    """
    val = final_params[idx]
    # special case for eccentricity
    if idx == 2:
        e_min = 0.0
        e_max = min(1.0, float(val + 0.3*abs(val) + 0.2))  # push a bit if val is small
        if e_max < 0.01:
            e_max = 0.3  # fallback in case final e ~ 0
        return jnp.linspace(e_min, e_max, n_points)
    else:
        # for other params: define +-30%
        low = float(val - 0.3*abs(val)) 
        high = float(val + 0.3*abs(val)) 
        # if val ~ 0, expand range a bit
        spread = abs(high - low)
        if spread < 1e-6:  
            low = float(val - 0.5)
            high = float(val + 0.5)
        # ensure low < high
        if low == high:
            low -= 0.5
            high += 0.5
        if low > min(param_history[:,idx]):
            low = min(param_history[:,idx]) - 0.5
        if high < max(param_history[:,idx]):
            high = max(param_history[:,idx]) + 0.5

        return jnp.linspace(low, high, n_points)

def demo_all_2D_contours():
    """
    1) Generate partial coverage data.
    2) Fit all parameters by gradient descent.
    3) For each pair of parameters (i,j), vary them in a 2D grid while 
       fixing the other 3 at the final best-fit values. Plot the filled contour.
    """
    # --- 1. Partial coverage data
    params_true = jnp.array([30.0, 10.0, 0.3, 1.0, 2.0])  # (P, K, e, omega, gamma)
    max_obs = 50
    t_data = jnp.linspace(0, max_obs, 25)
    rv_obs = generate_synthetic_data(params_true, t_data, noise_std=2.0)

    # --- 2. Gradient descent fit
    params_init = jnp.array([25.0, 5.0, 0.1, 0.0, 0.0])
    # lr = 1e-4
    plot_str = 'more_obs'
    lr = 1e-2
    steps = 3000
    params = params_init
    param_history = [params_init]
    for _ in tqdm.tqdm(range(steps)):
        g = grad_loss(params, t_data, rv_obs)
        params = params - lr*g
        param_history.append(params)
    param_history = jnp.array(param_history)
    print("Final best-fit params:", params)

    # We'll define param_names to label axes in plots
    param_names = ["Period (P)", "K [m/s]", "e", f"$\\omega$ [rad]", f"$\\gamma$ [m/s]"]

    # Indices of the 5 parameters
    param_indices = [0, 1, 2, 3, 4]
    pairs = []
    for i in range(5):
        for j in range(i+1, 5):
            pairs.append((i,j))

    # We'll produce a separate figure for each pair
    for (i, j) in pairs:
        # 1D arrays for param i and j
        range_i = build_param_ranges(params, i, n_points=60,param_history=param_history)
        range_j = build_param_ranges(params, j, n_points=60,param_history=param_history)

        # We'll vmap over these 1D arrays to produce a 2D "loss surface"
        def partial_loss(pi, pj):
            # Rebuild all 5, substituting pi, pj for the i-th, j-th param
            p_array = []
            for idxP in range(5):
                if idxP == i:
                    p_array.append(pi)
                elif idxP == j:
                    p_array.append(pj)
                else:
                    p_array.append(params[idxP])
            p_array = jnp.array(p_array)
            return rv_loss(p_array, t_data, rv_obs)

        # Evaluate on a mesh
        Pi, Pj = jnp.meshgrid(range_i, range_j, indexing='xy')
        Z = jax.vmap(
                lambda x: jax.vmap(
                    lambda y: partial_loss(x,y)
                )(range_j)
             )(range_i)
        # Z has shape [len(range_i), len(range_j)]

        # 2) Plot: We'll do a black background, filled contour
        fig, ax = plt.subplots(figsize=(7,5))
        fig, ax = set_bkgs_black(fig, ax)

        # Use .T if you want the second param on the Y-axis
        # but let's keep "i -> x, j -> y" standard:
        c_levels = 50
        cf = ax.contourf(range_i, range_j, Z.T, levels=c_levels, cmap='viridis')
        cbar = plt.colorbar(cf, ax=ax)
        # style colorbar for black bg
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        cbar.set_label("Loss", color='white')

        ax.set_xlabel(f"{param_names[i]}")
        ax.set_ylabel(f"{param_names[j]}")
        ax.set_title(f"2D contour: {param_names[i]} vs {param_names[j]}")
        ax.plot(params[i], params[j], color='red', marker='x', label='Best Fit',markersize=20)
        ax.plot(param_history[:,i][::5], param_history[:,j][::5], 'C3', ms=5, alpha=0.7, 
            label='Grad Descent Path')
        plt.tight_layout()

        # Save figure e.g. "contour_pair_0_1.png"
        fig_name = f"contour_pair_{i}_{j}_{plot_str}.png"
        plt.savefig(fig_name, dpi=120)
        plt.close(fig)

    print("Generated all 10 pairwise 2D contour plots.")
    print("Check files named contour_pair_i_j.png for each pair.")

if __name__ == "__main__":
    demo_all_2D_contours()