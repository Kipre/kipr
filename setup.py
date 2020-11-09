import setuptools
import numpy as np
from sys import platform

extra_args = {}

debug = True

if platform == "linux" or platform == "linux2":
    extra_args = {}
elif platform == "darwin":
    pass
elif platform == "win32":
    extra_args = {
      'extra_compile_args': ['/std:c++latest', "/arch:AVX2"],
      'extra_link_args': []
    }
    if debug:
        extra_args['extra_compile_args'] += ['/Zi', '/Od']
        extra_args['extra_link_args'] += ['/DEBUG']

# amalgamation 
with open('src/arraymodule.cpp', 'w') as amalgamation:
    # include real header
    amalgamation.write('#include "arraymodule.hpp" \n')

    # include allcode 
    for src_file in ['src/python_boilerplate.cpp', 'src/utils.cpp',
                     'src/karray.cpp', 'src/shape.cpp', 'src/filter.cpp',
                     'src/kernels.cpp', 'src/members.cpp',
                     'src/math_ops.cpp', 'src/module_functions.cpp']:
        with open(src_file, 'r') as src:
            amalgamation.write(src.read())

    # include test suite
    amalgamation.write('#include "test.hpp" \n')

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
