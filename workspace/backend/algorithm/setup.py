from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='cython_mcts',
    ext_modules=cythonize("../cpython/cython_mcts.pyx", build_dir="../cpython"),
    include_dirs=[np.get_include()],
    zip_safe=False,
) 