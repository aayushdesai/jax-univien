import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

def rv_model(params, t):
    """
    Simple circular orbit radial velocity model:
      v(t) = K * sin(2π t / P + phi)
    Here, phi is fixed for demonstration (phi=0).
    params = [K, P]
    """
    K, P = params
    phi = 0.0  # fixed for simplicity
    return K * jnp.sin((2.0 * jnp.pi / P) * t + phi)

def generate_synthetic_data(true_params, t, noise_std=1.0):
    """
    Generate synthetic radial velocity observations.
    """
    key = jax.random.PRNGKey(42)
    v_true = rv_model(true_params, t)
    noise = noise_std * jax.random.normal(key, shape=t.shape)
    return v_true + noise

def loss_function(params, t, v_obs):
    """
    Mean squared error between model and observations.
    """
    v_pred = rv_model(params, t)
    return jnp.mean((v_pred - v_obs)**2)

# We'll compute gradient and Hessian using JAX:
grad_loss = jax.grad(loss_function, argnums=0)
hess_loss = jax.hessian(loss_function, argnums=0)

def demo_geometric_issue():
    """
    Demonstrate how gradient alone can be misleading when
    there's a narrow valley or correlated parameters.
    """
    # 1. Generate some synthetic data
    true_params = jnp.array([10.0, 10.0])   # K=10 m/s, P=10 days
    t = jnp.linspace(0, 30, 20)            # 20 observations over 30 days
    v_obs = generate_synthetic_data(true_params, t, noise_std=2.0)
    
    # 2. Let's do a naive gradient descent to illustrate
    init_params = jnp.array([5.0, 8.0])    # starting guess
    learning_rate = 0.001
    n_steps = 2000
    
    params = init_params
    for step in range(n_steps):
        g = grad_loss(params, t, v_obs)
        params = params - learning_rate * g
    
    # 3. Compute gradient & Hessian at the final parameters
    grad_at_min = grad_loss(params, t, v_obs)
    hess_at_min = hess_loss(params, t, v_obs)
    
    # 4. Print results & Hessian info
    print("True params:        ", true_params)
    print("Recovered params:   ", params)
    print("Gradient at minimum:", grad_at_min)
    print("Hessian at minimum:\n", hess_at_min)
    
    # Eigen-decomposition of Hessian: large condition number => big difference in curvature
    eigvals, eigvecs = jnp.linalg.eig(hess_at_min)
    print("Eigenvalues of Hessian:", eigvals)
    print("Condition number:", jnp.abs(eigvals.max() / eigvals.min()))
    
    # 5. Plot the data and model
    t_dense = jnp.linspace(0, 30, 200)
    v_fit = rv_model(params, t_dense)
    
    plt.figure(figsize=(7,5))
    plt.scatter(t, v_obs, color='k', label='Noisy data')
    plt.plot(t_dense, v_fit, 'r-', label='Fitted model')
    plt.xlabel('Time [days]')
    plt.ylabel('RV [m/s]')
    plt.title('Toy RV Fit: Geometric Curvature Issue')
    plt.legend()
    plt.show()
    
    # 6. Visualize contour near final params to see "banana" shape
    # We'll do a small grid around the recovered parameters
    K_grid = jnp.linspace(params[0]-5, params[0]+5, 50)
    P_grid = jnp.linspace(params[1]-5, params[1]+5, 50)
    KK, PP = jnp.meshgrid(K_grid, P_grid)
    
    def batched_loss(k, p):
        return loss_function(jnp.array([k, p]), t, v_obs)
    Z = jax.vmap(lambda k: jax.vmap(lambda p: batched_loss(k,p))(P_grid))(K_grid)

    plt.figure(figsize=(7,5))
    cs = plt.contour(K_grid, P_grid, Z, levels=20, cmap='viridis')
    plt.clabel(cs, inline=1, fontsize=8)
    plt.plot(params[0], params[1], 'rx', label='Fitted Min')
    plt.title('Loss Surface Near Minimum')
    plt.xlabel('K [m/s]')
    plt.ylabel('P [days]')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    demo_geometric_issue()