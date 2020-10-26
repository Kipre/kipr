import setuptools
import numpy as np

native_side = setuptools.Extension(name='kipr_nat',
                                   sources=['src/natmodule.cpp'],
                                   library_dirs=['C:\\Program Files\\Python39\\libs'])

arrays = setuptools.Extension(name='kipr_array',
                              sources=['src/arraymodule.cpp'],
                              include_dirs=[np.get_include()],
                              library_dirs=['C:\\Program Files\\Python39\\libs'],
                              extra_compile_args=['/std:c++latest','/Zi', '/Od', "/arch:AVX2"],
                              extra_link_args=['/DEBUG'])

setuptools.setup(name='kipr',
                 version='0.0',
                 author='Cyprien', 
                 description='Personal toolbox',
                 packages=['kipr'],
                 ext_modules=[native_side, arrays])