Regression models:
* Less than 5000 points: SGPR [Titsias 2009] with lbfgs.
* More than 5000 points: SVGP [Hensman 2013] with natural gradients [Salimbeni 2018] for the variational parameters and Adam optimizer the remaining parameters.

For classification models the model is SVGP [Hensman 2015] with a Bernoulli likelihood, and Robust Max for multiclass classification, with natural gradients [Salimbeni 2018] for the variational parameters and Adam optimizer the remaining parameters.


@article{titsias2009variational,
  title={Variational learning of inducing variables in sparse Gaussian processes},
  author={Titsias, Michalis},
  journal={Artificial Intelligence and Statistics},
  year={2009}
}

@article{hensman2013gaussian,
  title={Gaussian processes for big data},
  author={Hensman, James and Fusi, Nicolo and Lawrence, Neil D},
  journal={Uncertainty in Artificial Intelligence},
  year={2013}
}

@article{hensman2015scalable,
  title={Scalable variational Gaussian process classification},
  author={Hensman, James and Matthews, Alexander G de G and Ghahramani, Zoubin},
  journal={Artificial Intelligence and Statistics},
  year={2015}
}

@article{salimbeni2018natural,
  title={Natural Gradients in Practice: Non-Conjugate Variational Inference in Gaussian Process Models},
  author={Salimbeni, Hugh and Eleftheriadis, Stefanos and Hensman, James},
  journal={Artificial Intelligence and Statistics},
  year={2018}
}
