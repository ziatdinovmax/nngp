from typing import Dict, Tuple, Callable, Optional
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median

kernel_fn_type = Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray],  jnp.ndarray]


class GP:
    """
    Fully Bayesian exact Gaussian process
    """

    def __init__(self,
                 input_dim: int, kernel: kernel_fn_type,
                 lengthscale_prior: Optional[dist.Distribution] = None,
                 noise_prior: Optional[dist.Distribution] = None
                 ) -> None:
        self.kernel = kernel
        self.kernel_dim = input_dim
        self.lengthscale_prior = lengthscale_prior
        self.noise_prior = noise_prior
        self.X_train = None
        self.y_train = None

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, **kwargs: float) -> None:
        """GP probabilistic model with inputs X and targets y"""
        # Initialize mean function at zeros
        f_loc = jnp.zeros(X.shape[0])
        # Sample kernel parameters
        kernel_params = self._sample_kernel_params()
        # Sample noise
        noise = self._sample_noise()
        # Compute kernel
        k = self.kernel(X, X, kernel_params, noise, **kwargs)
        # Sample y according to the standard Gaussian process formula
        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=f_loc, covariance_matrix=k),
            obs=y,
        )

    def fit(
        self,
        rng_key: jnp.array,
        X: jnp.ndarray,
        y: jnp.ndarray,
        num_warmup: int = 2000,
        num_samples: int = 2000,
        num_chains: int = 1,
        chain_method: str = "sequential",
        progress_bar: bool = True,
        print_summary: bool = True,
        **kwargs: float
    ) -> None:
        """
        Run Hamiltonian Monter Carlo to infer the GP parameters

        Args:
            rng_key: random number generator key
            X: 2D feature vector
            y: 1D target vector
            num_warmup: number of HMC warmup states
            num_samples: number of HMC samples
            num_chains: number of HMC chains
            chain_method: 'sequential', 'parallel' or 'vectorized'
            progress_bar: show progress bar
            print_summary: print summary at the end of sampling
            device:
                optionally specify a cpu or gpu device on which to run the inference;
                e.g., ``device=jax.devices("cpu")[0]``
            **jitter:
                Small positive term added to the diagonal part of a covariance
                matrix for numerical stability (Default: 1e-6)
        """
        X, y = self._set_data(X, y)
        self.X_train = X
        self.y_train = y

        init_strategy = init_to_median(num_samples=10)
        kernel = NUTS(self.model, init_strategy=init_strategy)
        self.mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=progress_bar,
            jit_model_args=False,
        )
        self.mcmc.run(rng_key, X, y, **kwargs)

        if print_summary:
            self._print_summary()

    def _sample_noise(self) -> jnp.ndarray:
        """
        Sample model's noise variance with either default
        weakly-informative log-normal priors or with a custom prior
        (must be provided at the initialization stage)
        """
        if self.noise_prior is not None:
            noise_prior = self.noise_prior
        else:
            noise_prior = dist.LogNormal(0, 1)
        return numpyro.sample("noise", noise_prior)

    def _sample_kernel_params(self, output_scale=True) -> Dict[str, jnp.ndarray]:
        """
        Sample kernel parameters with either default
        weakly-informative log-normal priors or with a custom prior
        (must be provided at the initialization stage)
        """
        if self.lengthscale_prior is not None:
            lscale_prior = self.lengthscale_prior
        else:
            lscale_prior = dist.LogNormal(0.0, 1.0)
        with numpyro.plate("ard", self.kernel_dim):  # ARD kernel
            length = numpyro.sample("k_length", lscale_prior)
        if output_scale:
            scale = numpyro.sample("k_scale", dist.LogNormal(0.0, 1.0))
        else:
            scale = numpyro.deterministic("k_scale", jnp.array(1.0))
        kernel_params = {"k_length": length, "k_scale": scale}
        return kernel_params

    def compute_posterior_mean_and_cov(self, X_new: jnp.ndarray,
                                       params: Dict[str, jnp.ndarray],
                                       noiseless: bool = False,
                                       **kwargs: float
                                       ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns parameters (mean and cov) of multivariate normal posterior
        for a single sample of trained GP parameters
        """
        noise = params["noise"]
        noise_p = noise * (1 - jnp.array(noiseless, int))
        # compute kernel matrices for train and new/test data
        k_XX = self.kernel(self.X_train, self.X_train, params, noise, **kwargs)
        k_pp = self.kernel(X_new, X_new, params, noise_p, **kwargs)
        k_pX = self.kernel(X_new, self.X_train, params)
        # compute predictive mean covariance
        K_xx_inv = jnp.linalg.inv(k_XX)
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, self.y_train))
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
        return mean, cov

    def _set_data(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> Union[Tuple[jnp.ndarray], jnp.ndarray]:
        X = X if X.ndim > 1 else X[:, None]
        if y is not None:
            return X, y.squeeze()
        return X

    def _print_summary(self):
        samples = self.get_samples(1)
        numpyro.diagnostics.print_summary(samples)

