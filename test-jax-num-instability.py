import jax.numpy as jnp
from jax import grad
import jax.scipy as jsp

from scipy.special import roots_laguerre

def F(c1, c2, tmax, N):
    """
    Inputs:
    c1, c2  are the coefficients in the integral
    tmax is the upper limit of the integral
    N is the number of points. Can be low with a higher-order integration scheme
    Output: scalar integral value
    """
    t0 = jnp.linspace(c2, tmax, N)
    return jsp.integrate.trapezoid(c1/(1+t0**2), t0)

c1 = -jnp.pi*jnp.exp(1)
c2 = 2*jnp.pi
tmax = jnp.pi*1e7
N = 10000000

gradF = grad(F, (0,1))
dFc1, dFc2   = gradF(c1, c2, tmax, N)


integral_value_jax = F(c1, c2, tmax, N)
intergral_value_analytic = ( jnp.arctan(tmax)-jnp.arctan(c2) ) * c1
print('integral = ', integral_value_jax)
print('analytic = ', intergral_value_analytic)
print(f'error in integral = {(jnp.abs(integral_value_jax - intergral_value_analytic)/jnp.abs(intergral_value_analytic)):.3e} ')
print('grad1 = ', dFc1,'analytic',jnp.arctan(tmax)-jnp.arctan(c2))
print('grad2 = ', dFc2,'analytic', c1/(1+c2**2))

print(f'error in grad1 = {(dFc1.item() - (jnp.arctan(tmax)-jnp.arctan(c2)))/(jnp.arctan(tmax)-jnp.arctan(c2)):.3e} ')
print(f'error in grad2 = {(dFc2.item() - -c1/(1+c2**2))/(c1/(1+c2**2)):.3e} ')


import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

fig, ax = plt.subplots()
x = jnp.logspace(0,9,1000)
tmax = jnp.pi*x[0]
integral_value_jax = F(c1, c2, tmax, N)
integral_value_analytic = ( jnp.arctan(tmax)-jnp.arctan(c2) ) * c1

integral_error = jnp.abs(integral_value_jax - integral_value_analytic)/jnp.abs(integral_value_analytic)

grad1,grad2 = gradF(c1, c2, tmax, N)
analytic_grad1 = jnp.arctan(tmax)-jnp.arctan(c2)
analytic_grad2 = -c1/(1+c2**2)
scat1 = ax.scatter(x[0], jnp.abs((integral_value_jax - integral_value_analytic)/(integral_value_analytic)),label='Integral',color='C0',alpha=0.5)
scat2 = ax.scatter(x[0], jnp.abs((grad1 - analytic_grad1)/(analytic_grad1)),label=f'$\\nabla_A f$',color='C1',alpha=0.5)
scat3 = ax.scatter(x[0], jnp.abs((grad2 - analytic_grad2)/(analytic_grad2)),label='$\\nabla_B f$',color='C2',alpha=0.5)

# scat22 = ax.scatter(x[0],analytic_grad1,label='Analytic',color='C1',alpha=0.5,marker='x')
# scat33 = ax.scatter(x[0],analytic_grad2,label='Analytic',color='C2',alpha=0.5,marker='x')
# scat11 = ax.scatter(x[0],integral_value_analytic,label='Analytic',color='C0',alpha=0.5,marker='x')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e-8,1e-5)
ax.set_xlim(1,1e1)

#animate over all values of x
x_vals = []
y_vals1 = []
y_vals2 = []
y_vals3 = []
y_vals4 = []
y_vals5 = []
y_vals6 = []
fig.patch.set_facecolor('black')
ax.patch.set_facecolor('black')
        #set axis color to white
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['right'].set_color('white')
ax.spines['left'].set_color('white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.title.set_color('white')
skip = 10
ax.set_xlabel('Upper Limit of Integration')
ax.set_ylabel('Relative Error')
def animate(i):
    i = i*skip
    tmax = jnp.pi*x[i]
    integral_value_jax = F(c1, c2, tmax, N)
    integral_value_analytic = ( jnp.arctan(tmax)-jnp.arctan(c2) ) * c1
    grad1,grad2 = gradF(c1, c2, tmax, N)
    analytic_grad1 = jnp.arctan(tmax)-jnp.arctan(c2)
    analytic_grad2 = -c1/(1+c2**2)
    # scat1.set_offsets((x[i], jnp.abs((integral_value_jax - integral_value_analytic)/(integral_value_analytic))))
    # scat2.set_offsets((x[i], jnp.abs((grad1 - analytic_grad1)/(analytic_grad1))))
    # scat3.set_offsets((x[i], jnp.abs((grad2 - analytic_grad2)/(analytic_grad2))))
    x_vals.append(x[i])
    y_vals1.append(jnp.abs((integral_value_jax - integral_value_analytic)/(integral_value_analytic)))
    y_vals2.append(jnp.abs((grad1 - analytic_grad1)/(analytic_grad1)))
    y_vals3.append(jnp.abs((grad2 - analytic_grad2)/(analytic_grad2)))
    y_vals4.append(analytic_grad1)
    y_vals5.append(analytic_grad2)
    y_vals6.append(integral_value_analytic)
    scat1.set_offsets(np.c_[x_vals, y_vals1])
    scat2.set_offsets(np.c_[x_vals, y_vals2])
    scat3.set_offsets(np.c_[x_vals, y_vals3])
    # scat11.set_offsets(np.c_[x_vals, y_vals6])
    # scat22.set_offsets(np.c_[x_vals, y_vals4])
    # scat33.set_offsets(np.c_[x_vals, y_vals5])

    if jnp.abs((grad1 - analytic_grad1)/(analytic_grad1)) > ax.get_ylim()[1]:
        ax.set_ylim(1e-8,jnp.abs((grad1 - analytic_grad1)/(analytic_grad1))*100)
    if x[i]>ax.get_xlim()[1]:
        ax.set_xlim(1,x[i]*10)
    ax.set_title(f'U = {np.pi*x[i]:.2e}')

ax.legend(title='Error Metrics')
ani = animation.FuncAnimation(
            fig, animate, frames=int(1000/skip), interval=200, blit=False
        )

ani.save('num_instability.gif', fps=10, writer="pillow")
