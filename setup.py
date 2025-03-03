import os

import numpy as np
from setuptools import find_packages
from setuptools import setup


with open(os.path.join(os.path.dirname(__file__), "README.md"),
          encoding='utf-8') as readme:
    long_description = readme.read()

try:
    from Cython.Distutils.extension import Extension
    from Cython.Distutils import build_ext
except ImportError:
    from setuptools.extension import Extension
    USING_CYTHON = False
else:
    USING_CYTHON = True

ext = '.pyx' if USING_CYTHON else '.c'

genome_module = Extension(
    "selene_sdk.sequences._sequence",
    ["selene_sdk/sequences/_sequence" + ext],
    include_dirs=[np.get_include()])

genomic_features_module = Extension(
    "selene_sdk.targets._genomic_features",
    ["selene_sdk/targets/_genomic_features" + ext],
    include_dirs=[np.get_include()])

ext_modules = [genome_module, genomic_features_module]
cmdclass = {'build_ext': build_ext} if USING_CYTHON else {}

setup(name="selene-sdk",
      version="0.3.0",
      long_description=long_description,
      long_description_content_type='text/markdown',
      description=("framework for developing sequence-level "
                   "deep learning networks"),
      packages=find_packages(),
      url="https://github.com/FunctionLab/selene",
      package_data={
        "selene_sdk.interpret": [
            "data/gencode_v28_hg38/*",
            "data/gencode_v28_hg19/*"
        ],
        "selene_sdk.sequences": [
            "data/*"
        ]
      },
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
      ],
      ext_modules=ext_modules,
      cmdclass=cmdclass,
      install_requires=[
        "cython>=0.27.3",
        "h5py",
        "matplotlib>=2.2.3",
        "numpy",
        "pandas",
        "plotly",
        "pyfaidx",
        "pytabix",
        "pyyaml",
        "scikit-learn",
        "scipy",
        "seaborn",
        "statsmodels",
        "torch>=0.4.1",
        "torchvision"
    ])
