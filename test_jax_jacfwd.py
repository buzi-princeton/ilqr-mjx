import jax
import jax.numpy as jnp


def f(x):
    return jnp.asarray(
        [x[0], 5 * x[2], 4 * x[1]**2 - 2 * x[2], x[2] * jnp.sin(x[0])])


print(jax.jacfwd(f)(jnp.array([1., 2., 3.])))
