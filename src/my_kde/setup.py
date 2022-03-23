from distutils.core import setup
from Cython.Build import cythonize
import numpy
setup(
    name = 'my_kde',
    ext_modules = cythonize("my_kde.pyx"),
    include_dirs = [numpy.get_include()]
)