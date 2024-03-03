from dataclasses import dataclass
import numpyro.distributions as dist


@dataclass
class GPPriors:
    lengthscale_prior: dist.Distribution = dist.LogNormal(0.0, 1.0)
    noise_prior: dist.Distribution = dist.LogNormal(0.0, 1.0)
    output_scale_prior: dist.Distribution = dist.LogNormal(0.0, 1.0)
