from setuptools import setup
from Cython.Build import cythonize
import numpy as np

# compile with python setup.py build_ext --inplace

if __name__ == "__main__":
    setup(
        ext_modules=cythonize(["cproblem.pyx"],
                              annotate=True,
                              compiler_directives={
                                  'language_level': '3',
                                  'initializedcheck': False,
                                  'boundscheck': False,
                                  'cdivision': True,
                                  'nonecheck': False,
                              }),
        include_dirs=[np.get_include()],
    )
