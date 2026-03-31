
# Compile cython functions for pheromone swarm

# python3 setup.py build_ext --inplace
import os
import sys
sys.path.insert(0, os.path.abspath("compiled_packages"))

from setuptools import setup, Extension
from Cython.Build import cythonize # type: ignore[reportUnknownVariableType]
import numpy as np

# compile_args = ["-fopenmp", "-O3", "-march=native"]
# link_args = ["-fopenmp", "-march=native"]

compile_args = ["-O3"]
link_args = []

print(f"{compile_args = }")
print(f"{link_args = }")

extensions = [
    Extension(
        'phero_c',
        ['phero_c.pyx'],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=[np.get_include()],
        language='c++'
    )
]

setup(ext_modules = cythonize(extensions, language_level='3str')) # Ensure GCC is used # type: ignore[reportUnknownArgumentType]