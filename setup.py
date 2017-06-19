"""Setup script."""

import setuptools

setuptools.setup(
    author="Jihong Ju",
    author_email="daniel.jihong.ju@gmail.com",
    extras_require={
        "test": [
            "pandas==0.19.2",
            "tensorflow",
            "codecov",
            "mock",
            "pytest",
            "pytest-cov",
            "pytest-pep8",
            "pytest-runner",
            "pydot",
            "graphviz"
        ],
    },
    install_requires=[
        'keras>=2.0.0'
    ],
    name="keras-fcn",
    description=("A reimplemtenation of fully convolutional networks"),
    packages=['keras_fcn'],
    url="https://github.com/JihongJu/keras-fcn",
    version="0.0.1",
)
