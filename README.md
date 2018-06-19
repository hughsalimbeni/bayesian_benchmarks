# Bayesian Benchmarks

[![Build Status](https://travis-ci.org/hughsalimbeni/bayesian_benchmarks.svg?branch=master)](https://travis-ci.org/hughsalimbeni/bayesian_benchmarks)
[![codecov](https://codecov.io/gh/hughsalimbeni/bayesian_benchmarks/branch/master/graph/badge.svg)](https://codecov.io/gh/hughsalimbeni/bayesian_benchmarks)

This is a set of tools for evaluating Bayesian models, together with benchmark results.

Problems this repository attempts to solve:
* Variations between tasks in the literature make a fair comparison between methods difficult.
* Implementing competing methods takes considerable effort, and there is an obvious incentive to do a poor job.
* Published papers may not always provide complete details of implementations due to space considerations.
* There is a lack of standardized tasks that meaningfully assess the quality of uncertainty quantification.

The aims of this repository is to:
* Curate a set of benchmarks that meaningfully compare the efficacy of Bayesian models in real-world tasks.
* Maintain a fair assessment of benchmark methods, with full implementations and results.

Current tasks:
* Classification and regression
* Density estimation (real world and synthetic) (TODO)
* Active learning
* Adversarial robustness (TODO)
