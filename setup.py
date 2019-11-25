from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

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
