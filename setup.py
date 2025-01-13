"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages

setup(
    name="torchwnn",
    version="1.0.1",
    author="Leandro Santiago de Araújo",
    description="Torcwnn is a Python library for Weightless Neural Network",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/leandro-santiago/torchwnn",
    license="MIT",
    install_requires=[
        "torch",
        "ucimlrepo",
        "pandas",
        "numpy",        
    ],    
    packages=find_packages(exclude=["examples"]),
)
