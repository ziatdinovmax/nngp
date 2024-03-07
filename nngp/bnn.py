from typing import Dict, Tuple, Optional, Union, List
import jax.random as jra
import jax.numpy as jnp
from jax import vmap
import numpyro
import numpyro.distributions as dist

from numpyro.infer import MCMC, NUTS, Predictive, init_to_median

from .nn import get_mlp
from .priors import get_mlp_prior


class BNN:

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: List[int] = None,
                 activation: str = 'tanh',
                 noise_prior: Optional[dist.Distribution] = None
                 ) -> None:
        if noise_prior is None:
            noise_prior = dist.LogNormal(0.0, 1.0)
        hdim = hidden_dim if hidden_dim is not None else [32, 16, 8]
        self.nn = get_mlp(hdim, activation)
        self.nn_prior = get_mlp_prior(input_dim, output_dim, hdim)
        self.noise_prior = noise_prior

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, **kwargs) -> None:
        """BNN probabilistic model"""

        # Sample NN parameters
        nn_params = self.nn_prior()
        # Pass inputs through a NN with the sampled parameters
        mu = self.nn(X, nn_params)

        # Sample noise
        sig = self.sample_noise()

        # Score against the observed data points
        numpyro.sample("y", dist.Normal(mu, sig), obs=y)

    def fit(self, X: jnp.ndarray, y: jnp.ndarray,
            num_warmup: int = 2000, num_samples: int = 2000,
            num_chains: int = 1, chain_method: str = 'sequential',
            progress_bar: bool = True, rng_key: Optional[jnp.array] = None,
            ) -> None:
        """
        Run HMC to infer parameters of the BNN

        Args:
            X: 2D feature vector
            y: 1D target vector
            num_warmup: number of HMC warmup states
            num_samples: number of HMC samples
            num_chains: number of HMC chains
            chain_method: 'sequential', 'parallel' or 'vectorized'
            progress_bar: show progress bar
            rng_key: random number generator key
        """
        key = rng_key if rng_key is not None else jra.PRNGKey(0)
        X, y = self.set_data(X, y)
        init_strategy = init_to_median(num_samples=10)
        kernel = NUTS(self.model, init_strategy=init_strategy)
        self.mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=progress_bar,
            jit_model_args=False
        )
        self.mcmc.run(key, X, y)

    def get_samples(self, chain_dim: bool = False) -> Dict[str, jnp.ndarray]:
        """Get posterior samples (after running the MCMC chains)"""
        return self.mcmc.get_samples(group_by_chain=chain_dim)

    def sample_noise(self) -> jnp.ndarray:
        """
        Sample observational noise variance
        """
        return numpyro.sample("sig", self.noise_prior)

    def sample_single_posterior_predictive(self, rng_key, X_new, params, n_draws):
        sigma = params["sig"]
        loc = self.nn(X_new, params)
        sample = dist.Normal(loc, sigma).sample(rng_key, (n_draws,)).mean(0)
        return loc, sample

    def predict(self,
                X_new: jnp.ndarray,
                n_draws: int = 1,
                rng_key: jnp.ndarray = None,
                ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict the mean and variance of the target values for new inputs.

        Parameters:
        - X_new: New input data for predictions.
        - n_draws: Number of draws to sample from the posterior predictive distribution.
        - rng_key: Random number generator key for JAX operations.

        Returns:
        Tuple containing the means and samples from the posterior predictive distribution.
        """
        if rng_key is None:
            rng_key = jra.PRNGKey(0)
        return self._vmap_predict(X_new, n_draws, rng_key)

    def sample_from_posterior(self,
                              X_new: jnp.ndarray,
                              n_draws: int = 1,
                              rng_key: Optional[jnp.ndarray] = None
                              ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict the mean and variance of the target values for new inputs
        """
        if rng_key is None:
            rng_key = jra.PRNGKey(0)
        _, f_samples = self._vmap_predict(X_new, n_draws, rng_key)
        return f_samples

    def _vmap_predict(self, X_new: jnp.ndarray, n_draws: int, rng_key: jnp.ndarray
                      ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Helper method to vectorize predictions over posterior samples
        """
        samples = self.get_samples(chain_dim=False)
        num_samples = len(next(iter(samples.values())))
        vmap_args = (jra.split(rng_key, num_samples), samples)

        predictive = lambda p1, p2: self.sample_single_posterior_predictive(p1, X_new, p2, n_draws)
        mu, f_samples = vmap(predictive)(*vmap_args)

        return mu, f_samples

    def set_data(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None
                 ) -> Union[Tuple[jnp.ndarray], jnp.ndarray]:
        X = X if X.ndim > 1 else X[:, None]
        if y is not None:
            y = y[:, None] if y.ndim < 2 else y
            return X, y
        return X
