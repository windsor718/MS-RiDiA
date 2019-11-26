from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extentions = [
    Extension('calc_storage',
              sources=['./cysrc/calc_storage.pyx'],
              include_dirs=[],
              extra_compile_args=['-O3'],
              extra_link_args=[])
]

setup(
    name='calc_storage',
    ext_modules=cythonize(extentions)
)

extentions = [
    Extension('camavec',
              sources=['./cysrc/camavec.pyx'],
              include_dirs=[np.get_include()],
              extra_compile_args=['-O3'],
              extra_link_args=[])
]

setup(
    name='camavec',
    ext_modules=cythonize(extentions)
)
