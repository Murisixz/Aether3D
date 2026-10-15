from setuptools import setup, Extension
from Cython.Build import cythonize
setup(ext_modules=cythonize([Extension("world_cy", ["world_cy.pyx"], extra_compile_args=["-O3", "-march=native", "-ffp-contract=off"])]))
