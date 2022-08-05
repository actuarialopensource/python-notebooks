import jax.numpy as jnp

def get_q_from_select_ultimate(
    mortality_table_index: jnp.ndarray,
    issue_age: jnp.ndarray,
    duration: jnp.ndarray,
    select: jnp.ndarray,
    ultimate: jnp.ndarray,
    min_age_select=18,
    min_age_ultimate=18,
):
    """
    Get the mortality rates from select/ultimate mortality table. 

    When duration is out of bounds it pulls rates from the end of the table.
    https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#out-of-bounds-indexing
    """
    return jnp.where(
        duration < select.shape[-1],
        select[mortality_table_index, issue_age - min_age_select, duration],
        ultimate[mortality_table_index, (issue_age - min_age_ultimate) + duration],
    )

def get_npx_from_q(q: jnp.ndarray):
    """
    Take the cumulative product along axis=0 of 1-q, append ones at beginning, and remove last element.
    """
    # initially exposures are 1, have same shape as elements along dimension 0
    initial = jnp.ones((1, *q.shape[1:]))
    # clip end to force shape to be same as q parameter
    decremented = jnp.cumprod(1-q, axis=0)[:-1]
    return jnp.concatenate([initial, decremented])