"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

cpp_dir = os.path.join('torchwnn', 'cpp')
ext_modules = []
ext_modules.append(CppExtension("torchwnn.cpp.functional", [os.path.join(cpp_dir, "functional.cpp")], extra_compile_args=['-O3']))

setup(
    name="torchwnn",
    version="1.0.2",
    author="Leandro Santiago de Araújo",
    description="Torcwnn is a Python library for Weightless Neural Network",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/leandro-santiago/torchwnn",
    license="MIT",      
    #package_dir={"": "torchwnn"},
    #packages=find_packages(where="torchwnn"),  
    packages=find_packages(exclude=["examples"]),
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    keywords = ['wisard', 'weithgless', 'neural', 'network'],
    install_requires=[
        'torch'
    ],
)
