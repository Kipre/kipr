import setuptools
import numpy as np
from sys import platform

extra_args = {}

debug = True

if platform in ["linux", "linux2"]:
    extra_args = {}
elif platform == "darwin":
    pass
elif platform == "win32":
    extra_args = {
      'extra_compile_args': ['/std:c++latest', "/arch:AVX2", '/Zc:strictStrings-'],
      'extra_link_args': []
    }
    if debug:
        extra_args['extra_compile_args'] += ['/Zi', '/Od']
        extra_args['extra_link_args'] += ['/DEBUG']
    else:
        extra_args['extra_compile_args'] += ['/openmp', '/O2']
else:
    raise Exception(f'Unknown platform {platform}.')

with open('src/arraymodule.cpp', 'w') as f:
    f.write(f'// {np.random.randn}\n')
    f.write('#include "arraymodule.hpp"\n')


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
                 ext_modules=[arrays])
