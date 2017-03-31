"""Setup script."""

from setuptools import setup

setup(
    name="keras-fcn",
    version="0.0.1",
    author="Jihong Ju",
    author_email="daniel.jihong.ju@gmail.com",
    description=("A reimplemtenation of fully convolutional networks"),
    packages=['fcn'],
    install_requires=['keras>=2.0.0']
    )
