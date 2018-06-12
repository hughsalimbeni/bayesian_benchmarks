from setuptools import setup

requirements = [
    'numpy>=1.10.0',
    'scipy>=0.18.0',
    'xlrd>=1.1.0',
]

setup(name='gpflow',
      version='alpha',
      author="Hugh Salimbeni",
      author_email="hrs13@ic.ac.uk",
      description=("Bayesian benchmarking"),
      license="Apache License 2.0",
      keywords="machine-learning bayesian-methods",
      url="https://github.com/hughsalimbeni/bayesian_benchmarks",
      install_requires=requirements,
      include_package_data=True,
      classifiers=[
          'License :: OSI Approved :: Apache Software License',
          'Natural Language :: English',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ])
