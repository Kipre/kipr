# amalgamation 
with open('src/arraymodule.cpp', 'w') as amalgamation:
    amalgamation.write('#include "arraymodule.hpp" \n')

    for src_file in ['src/debug.cpp', 'src/utils.cpp',
                     'src/members.cpp',
                     'src/module_functions.cpp',
                     'src/kernels.cpp', 'src/math_ops.cpp']:
        with open(src_file, 'r') as src:
            amalgamation.write(src.read())
    
    amalgamation.write('#include "test.hpp" \n')