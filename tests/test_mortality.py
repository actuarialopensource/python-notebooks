from jaxtuary.mortality import get_q_from_select_ultimate
import jax.numpy as jnp
import numpy as onp


def test_get_q_from_select_ultimate():
    # test 1:
    # issue_age = 18, duration = [0,5,2]
    select = jnp.array(
        [
            [[0.1, 0.2], [0.4, 0.5], [0.7, 0.8]],
            [[0.15, 0.25], [0.45, 0.55], [0.75, 0.85]],
        ]
    )
    ultimate = jnp.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])

    q = get_q_from_select_ultimate(
        mortality_table_index=jnp.array([0, 1]),
        issue_age=jnp.array([18, 19]),
        duration=jnp.array([[0, 1], [1, 2]]),
        select=select,
        ultimate=ultimate,
        min_age_select=18,
        min_age_ultimate=18,
    )

    assert onp.all(q == jnp.array([[0.1, 0.55], [0.2, 0.8]]))
