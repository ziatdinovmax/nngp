from typing import Callable, Dict, List
import jax.numpy as jnp


def get_mlp(architecture: List[int]) -> Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]:
    """Returns a function that represents an MLP for a given architecture."""
    def mlp(X: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """MLP for a single MCMC sample of weights and biases, handling arbitrary number of layers."""
        h = X
        for i in range(len(architecture)):
            h = jnp.tanh(jnp.matmul(h, params[f"w{i}"]) + params[f"b{i}"])
        # No non-linearity after the last layer
        z = jnp.matmul(h, params[f"w{len(architecture)}"]) + params[f"b{len(architecture)}"]
        return z
    return mlp
