"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages

# Read the version without importing any dependencies
version = {}
with open("torchwnn/version.py") as f:
    exec(f.read(), version)

setup(
    name="torchwnn",
    version=version["__version__"],
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
    package_dir={"": "torchwnn"},
    packages=find_packages(where="torchwnn", exclude=["examples"]),
)
