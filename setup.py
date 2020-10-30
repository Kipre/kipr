import setuptools
import numpy as np
from sys import platform

extra_args = {}

if platform == "linux" or platform == "linux2":
    extra_args = {
      # 'extra_compile_args': ['-mavx2']
    }
elif platform == "darwin":
    pass
elif platform == "win32":
    extra_args = {
      'extra_compile_args': ['/std:c++latest','/Zi', '/Od', "/arch:AVX2"],
      'extra_link_args': ['/DEBUG']
    }

native_side = setuptools.Extension(name='kipr_nat',
                                   sources=['src/natmodule.cpp'],
                                   library_dirs=['C:\\Program Files\\Python39\\libs'])

arrays = setuptools.Extension(name='kipr_array',
                              sources=['src/arraymodule.cpp'],
                              include_dirs=[np.get_include()],
                              library_dirs=['C:\\Program Files\\Python39\\libs'],
                              **extra_args)

setuptools.setup(name='kipr',
                 version='0.0',
                 author='Cyprien', 
                 description='Personal toolbox',
                 packages=['kipr'],
                 ext_modules=[native_side, arrays])