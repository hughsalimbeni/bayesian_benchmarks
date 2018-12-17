# Bayesian Benchmarks

[![Build Status](https://travis-ci.org/hughsalimbeni/bayesian_benchmarks.svg?branch=master)](https://travis-ci.org/hughsalimbeni/bayesian_benchmarks)
[![codecov](https://codecov.io/gh/hughsalimbeni/bayesian_benchmarks/branch/master/graph/badge.svg)](https://codecov.io/gh/hughsalimbeni/bayesian_benchmarks)

This is a set of tools for evaluating Bayesian models, together with benchmark implementations and results.

Motivations:
* There is a lack of standardized tasks that meaningfully assess the quality of uncertainty quantification for Bayesian black-box models.
* Variations between tasks in the literature make a direct comparison between methods difficult.
* Implementing competing methods takes considerable effort, and there little incentive to do a good job.
* Published papers may not always provide complete details of implementations due to space considerations.

Aims:
* Curate a set of benchmarks that meaningfully compare the efficacy of Bayesian models in real-world tasks.
* Maintain a fair assessment of benchmark methods, with full implementations and results.

Tasks:
* Classification and regression
* Density estimation (real world and synthetic) (TODO)
* Active learning
* Adversarial robustness (TODO)

Current implementations:
* Sparse variational GP, for [Gaussian](http://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf) and [non-Gaussian](http://proceedings.mlr.press/v38/hensman15.pdf) likelihoods
* Sparse variational GP, with [minibatches](https://arxiv.org/pdf/1309.6835.pdf)
* 2 layer Deep Gaussian process, with [doubly-stochastic variational inference](http://papers.nips.cc/paper/7045-doubly-stochastic-variational-inference-for-deep-gaussian-processes.pdf)
* A variety of sklearn models

See the models folder for instruction for adding new models. 

Coming soon:
* [Structured Variational Learning of Bayesian Neural Networks with Horseshoe Priors](https://arxiv.org/pdf/1806.05975.pdf)
* [Differentiable Compositional Kernel Learning for Gaussian Processes](https://arxiv.org/abs/1806.04326)
* [Deep Gaussian Processes using Stochastic Gradient Hamiltonian Monte Carlo
](https://arxiv.org/pdf/1806.05490.pdf)

