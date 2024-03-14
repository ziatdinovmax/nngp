# Deep Kernel Learning "in reverse"

Most literature and applications of Deep Kernel Learning (DKL) focus on leveraging deep learning to efficiently handle high-dimensional data with Gaussian processes (GPs), improving the scalability and applicability of GP to large datasets. Here, we demonstrate an application of DKL 'in reverse' for the low-dimensional problems that are traditionally ill-suited for GP. Specifically, we use a neural network part of the DKL to learn the transformation of the original low-dimensional data into a new, higher-dimensional coordinate space where the assumptions of GP about data smoothness and continuity are more likely to hold. This transformation makes the data more amenable to GP modeling, even when the original data exhibits non-stationary behavior, discontinuities, or other characteristics that would make direct GP modeling challenging.


Comparison between (fully Bayesian) Gaussian process and (fully Bayesian) Deep Kernel Learning for non-stationary and functions:


<img src="https://github.com/ziatdinovmax/nngp/assets/34245227/18380b97-d4dc-4bd2-9698-8f4cd19520fe" width="1000"><br>


Comparison between (fully Bayesian) Gaussian process and (fully Bayesian) Deep Kernel Learning for discontinuous functions:

<img src="https://github.com/ziatdinovmax/nngp/assets/34245227/4392435d-73de-48ed-8fa7-192878730fc1" width="1000">
