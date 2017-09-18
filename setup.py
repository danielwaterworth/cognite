import os
from setuptools import setup

setup(
    name = "cognite",
    version = "0.0.3",
    author = "Daniel Waterworth",
    author_email = "daniel@manganizeme.com",
    description = "",
    keywords = "deep-learning",
    url = "https://github.com/danielwaterworth/cognite",
    packages=['cognite'],
    long_description="",
    classifiers=[],
    install_requires=['mxnet', 'numpy', 'toposort'],
)
