# Machine Learning for Inverse Problems and Data Assimilation Demos

This repository contains textbook-ready demo notebooks for *Machine Learning for Inverse Problems and Data Assimilation*.

The textbook arXiv page is <https://arxiv.org/abs/2410.10523>. Chapter and section references in this demo codebase are based on `[v2] Mon, 6 Oct 2025`.

Some materials in this repository are based on the ML for IP and DA winter school in Amsterdam last year, available at <https://github.com/baptistar/MLforIPDA>.

These notebooks may contain small mistakes. If you find an issue, please contact bhchen@caltech.edu.

## Colab Runtime Recommendation

When opening these demos in Google Colab, we recommend using a high-RAM runtime
with a GPU accelerator. A standard T4 or L4 GPU is sufficient for the intended
interactive experiments.

## Demo Index

### 1. Basic Machine Learning Pipeline

- Notebook: `Demos/Basic_Machine_Learning_Pipeline.ipynb`
- Textbook reference: Chapter 12, Section 12.1: Notational Conventions; Chapter 12, Section 12.2: Neural Networks; Chapter 15, Section 15.1: Gradient Descent
- Introduction: This notebook introduces a complete PyTorch supervised-learning
  pipeline: data generation and loading, MLP model definition, weighted loss
  functions, optimizer setup, training loops, checkpoint reuse, and diagnostics.
  It uses a chirp regression problem to compare learning settings and an MNIST
  CNN with UMAP visualizations to diagnose learned high-dimensional features.
- Colab: [Open in Colab](https://colab.research.google.com/github/wispcarey/ML_for_IP_and_DA/blob/main/Demos/Basic_Machine_Learning_Pipeline.ipynb)

### 2. Introduction to Bayesian Inverse Problems

- Notebook: `Demos/Introduction_To_Bayesian_Inverse_Problems.ipynb`
- Textbook reference: Chapter 1, Section 1.1: Bayesian Inversion; Chapter 3, Section 3.4.2: Empirical Approximation Of The Prior
- Introduction: This notebook introduces the general Bayesian inverse problem setting
  $y = G(u) + \eta$, where the goal is to infer an unknown quantity $u$ from noisy
  observations $y$. It explains the roles of the prior, likelihood, and posterior,
  then illustrates typical subproblems and methods: scalar linear and nonlinear
  inverse problems, multimodal posteriors, empirical-prior MNIST inpainting, and
  total-variation MAP reconstruction for image denoising and deblurring.
- Colab: [Open in Colab](https://colab.research.google.com/github/wispcarey/ML_for_IP_and_DA/blob/main/Demos/Introduction_To_Bayesian_Inverse_Problems.ipynb)

### 3. Classical Posterior Sampling

- Notebook: `Demos/Classical_Posterior_Sampling.ipynb`
- Textbook reference: Chapter 1, Section 1.2.2: Posterior Expectations; Chapter 15, Section 15.3: Ensemble Kalman Inversion; Chapter 15, Section 15.5: Markov Chain Monte Carlo
- Introduction: This notebook compares classical non-ML approaches for a
  two-dimensional Bayesian inverse problem with a bimodal prior. It shows
  importance sampling with effective sample size diagnostics, random walk
  Metropolis with burn-in and proposal-scale diagnostics, and ensemble Kalman
  inversion as a derivative-free ensemble method that can collapse toward a
  data-fitting consensus.
- Colab: [Open in Colab](https://colab.research.google.com/github/wispcarey/ML_for_IP_and_DA/blob/main/Demos/Classical_Posterior_Sampling.ipynb)

### 4. Variational Posterior Approximation

- Notebook: `Demos/Variational_Posterior_Approximation.ipynb`
- Textbook reference: Chapter 1, Section 1.2.1: Maximum A Posteriori Estimator; Chapter 2: Variational Inference; Chapter 4: Transport To The Posterior; Chapter 13, Section 13.3: Normalizing Flows
- Introduction: This notebook studies posterior approximation for a
  two-dimensional biochemical oxygen demand inverse problem. It uses MAP
  estimation as a point-estimate baseline, then compares mean-field and
  full-covariance Gaussian variational inference with a RealNVP transport-map
  variational approximation for a fixed observation.
- Colab: [Open in Colab](https://colab.research.google.com/github/wispcarey/ML_for_IP_and_DA/blob/main/Demos/Variational_Posterior_Approximation.ipynb)

### 5. Amortized Posterior Sampling

- Notebook: `Demos/Amortized_Posterior_Sampling.ipynb`
- Textbook reference: Chapter 4: Transport To The Posterior; Chapter 5, Section 5.4: Likelihood-Based Inference; Chapter 5, Section 5.5: Likelihood-Free Inference; Chapter 11, Section 11.1.5: Maximum Mean Discrepancy And Energy Distance; Chapter 13, Section 13.3: Normalizing Flows
- Introduction: This notebook trains two amortized posterior samplers on simulated
  joint pairs from the biochemical oxygen demand model: a conditional RealNVP
  flow with likelihood-based training, and an energy-distance MLP transport map
  based on *Amortized Energy-Based Bayesian Inference*. Both methods reuse one
  trained conditional map to generate approximate posterior samples for
  low-signal, standard, and high-signal observations without retraining.
- Colab: [Open in Colab](https://colab.research.google.com/github/wispcarey/ML_for_IP_and_DA/blob/main/Demos/Amortized_Posterior_Sampling.ipynb)

### 6. Introduction to Data Assimilation and the Kalman Filter

- Notebook: `Demos/Introduction_To_DA_And_Kalman_Filter.ipynb`
- Textbook reference: Chapter 6, Section 6.2: Formulation Of Data Assimilation; Chapter 6, Section 6.3.1: Kalman Filter
- Introduction: This notebook introduces the forecast-analysis formulation of
  data assimilation through the linear Gaussian Kalman filter. It includes the
  Kalman update from predictive moments, the affine analysis-map viewpoint with
  a reference to <https://arxiv.org/abs/2209.11371>, and the original Kalman
  tracking experiment with full and partial observations.
- Colab: [Open in Colab](https://colab.research.google.com/github/wispcarey/ML_for_IP_and_DA/blob/main/Demos/Introduction_To_DA_And_Kalman_Filter.ipynb)

### 7. Particle Filters for Data Assimilation

- Notebook: `Demos/Particle_Filters_For_Data_Assimilation.ipynb`
- Textbook reference: Chapter 6, Section 6.3.6: Bootstrap Particle Filter; Chapter 6, Section 6.3.7: Optimal Particle Filter; Chapter 6, Section 6.3.8: Evaluating Probabilistic Estimation
- Introduction: This notebook compares bootstrap, optimal, and auxiliary particle
  filters on the shared linear Gaussian and Lorenz-63 filtering testbeds. It
  keeps the sequential importance sampling formulas explicit and uses RMSE
  diagnostics to show how proposal choice, particle count, observation noise,
  and initialization affect filtering performance.
- Colab: [Open in Colab](https://colab.research.google.com/github/wispcarey/ML_for_IP_and_DA/blob/main/Demos/Particle_Filters_For_Data_Assimilation.ipynb)

### 8. Ensemble Kalman Filters for Data Assimilation

- Notebook: `Demos/Ensemble_Kalman_Filters.ipynb`
- Textbook reference: Chapter 6, Section 6.3.5: Ensemble Kalman Filter
- Introduction: This notebook compares stochastic EnKF and deterministic EnSRF
  updates on the shared linear Gaussian and Lorenz-63 filtering testbeds. It
  emphasizes ensemble covariance estimates, perturbed-observation versus
  square-root analysis updates, post-analysis inflation, and RMSE diagnostics.
- Colab: [Open in Colab](https://colab.research.google.com/github/wispcarey/ML_for_IP_and_DA/blob/main/Demos/Ensemble_Kalman_Filters.ipynb)

### 9. Lorenz-63 Parameter Estimation

- Notebook: `Demos/Lorenz63_Parameter_Estimation.ipynb`
- Textbook reference: Chapter 1, Section 1.1: Bayesian Inversion; Chapter 1,
  Section 1.2.1: Maximum A Posteriori Estimator; Chapter 6, Section 6.3.5:
  Ensemble Kalman Filter; Chapter 15, Section 15.5: Markov Chain Monte Carlo
- Introduction: This notebook studies the deterministic Lorenz-63 inverse
  problem of estimating the physical parameters `theta = (sigma, rho)` from
  noisy observations of the first state component. It keeps the likelihood,
  prior, MAP objective, Laplace approximation, and augmented-state EnKF
  equations explicit, then compares MCMC posterior samples, a MAP-Laplace
  approximation, and an augmented EnKF parameter ensemble on the same data.
- Colab: [Open in Colab](https://colab.research.google.com/github/wispcarey/ML_for_IP_and_DA/blob/main/Demos/Lorenz63_Parameter_Estimation.ipynb)

### 10. Learning Model Error with MCEM and ADKF

- Notebook: `Demos/Learning_Model_Error_With_MCEM_And_ADKF.ipynb`
- Textbook reference: Chapter 8, Section 8.2: Expectation Maximization; Chapter
  8, Section 8.3: Auto-Differentiable Kalman Filters
- Introduction: This notebook extends Lorenz-63 parameter estimation to a
  stochastic dynamics setting where both the physical parameters
  `vartheta = (sigma, rho)` and the model-error covariance `Sigma` are
  unknown. It compares a Monte Carlo expectation-maximization workflow, using
  approximate smoothing samples and covariance updates, with an
  auto-differentiable Kalman-filter likelihood objective for learning model
  parameters from partial noisy observations.
- Colab: [Open in Colab](https://colab.research.google.com/github/wispcarey/ML_for_IP_and_DA/blob/main/Demos/Learning_Model_Error_With_MCEM_And_ADKF.ipynb)

### 11. Lorenz-96 Localized EnKF and Learned Regularization

- Notebook: `Demos/Lorenz96_Localized_EnKF_And_Learned_Regularization.ipynb`
- Textbook reference: Chapter 6, Section 6.3.5: Ensemble Kalman Filter; Chapter
  9, Section 9.2.3: Learning Localization And Inflation In EnKF
- Introduction: This notebook moves to the high-dimensional Lorenz-96 filtering
  benchmark and explains why limited ensembles require covariance localization
  and post-analysis inflation. It introduces the perturbed-observation EnKF,
  Gaspari-Cohn tapering, and multiplicative inflation, then learns the
  regularization parameters `theta_reg = {alpha, r}` by minimizing
  supervised filtering RMSE over training trajectories.
- Colab: [Open in Colab](https://colab.research.google.com/github/wispcarey/ML_for_IP_and_DA/blob/main/Demos/Lorenz96_Localized_EnKF_And_Learned_Regularization.ipynb)

### 12. Variational Filtering with a Learned Gain

- Notebook: `Demos/Variational_Filtering_With_Learned_Gain.ipynb`
- Textbook reference: Chapter 7, Section 7.2: Variational Formulation Of
  Filtering; Chapter 9, Section 9.2.1: Learning The Gain In 3DVar; Chapter 9,
  Section 9.2.2: Learning The Gain In EnKF
- Introduction: This notebook presents filtering as a sequence of variational
  approximations and learns a fixed gain matrix `K` for a reduced Lorenz-96
  filtering problem. It keeps the predictive Gaussian recursion, KL-based
  filtering objective, learned-gain analysis update, covariance update, and RMSE
  diagnostic explicit, then compares the learned gain with a fixed baseline
  gain.
- Colab: [Open in Colab](https://colab.research.google.com/github/wispcarey/ML_for_IP_and_DA/blob/main/Demos/Variational_Filtering_With_Learned_Gain.ipynb)

### 13. Likelihood-Free Ensemble Transport Filtering

- Notebook: `Demos/Likelihood_Free_Ensemble_Transport_Filtering.ipynb`
- Textbook reference: Chapter 10, Section 10.3.1: Minimizing The Energy
  Distance; Chapter 10, Section 10.3.2: Maximum Likelihood Estimation; Chapter
  11, Section 11.1.5: Maximum Mean Discrepancy And Energy Distance
- Introduction: This notebook studies ensemble transport filtering for
  stochastic Lorenz-63 dynamics when the analysis update is learned from joint
  forecast-state and synthetic-observation samples. It compares an
  energy-distance residual transport map with a maximum-likelihood conditional
  composed map, showing how each learned map turns forecast particles and a true
  observation into an approximate analysis ensemble.
- Colab: [Open in Colab](https://colab.research.google.com/github/wispcarey/ML_for_IP_and_DA/blob/main/Demos/Likelihood_Free_Ensemble_Transport_Filtering.ipynb)

### 14. Filtering Visualization on the Doubling Map

- Notebook: `Demos/Filtering_Visualization_Doubling_Map.ipynb`
- Textbook reference: Chapter 6, Section 6.3.6: Bootstrap Particle Filter;
  Chapter 6, Section 6.3.8: Evaluating Probabilistic Estimation
- Introduction: This notebook uses a one-dimensional doubling map with a
  nonlinear cosine observation to emphasize visualization of filtering
  distributions. It compares particle-filter posterior densities with EnKF
  ensembles through posterior snapshots, circular means, and time-density
  heatmaps, illustrating why distributional plots can reveal multimodality that
  point estimates hide.
- Colab: [Open in Colab](https://colab.research.google.com/github/wispcarey/ML_for_IP_and_DA/blob/main/Demos/Filtering_Visualization_Doubling_Map.ipynb)
