import jax.numpy as jnp

def project_duration(duration: jnp.ndarray, timesteps: int):
    """
    Take an array representing policy durations and project them forward by timesteps.
    Example: project_duration(duration = [0,5,2], timesteps = 2) == [[0,5,2],[1,6,3]].

    This function is a simple trick, make your own implementation if you need more sophisticed behavior.
    """
    # broadcasting
    return duration[None, :] + jnp.arange(timesteps)[:, None]