import os
from setuptools import setup, find_packages

setup(
    name="nextqsm",
    version="1.0.2",
    packages=find_packages(),
    package_dir={'nextqsm': 'nextqsm'},
    package_data={
        'nextqsm': ['checkpoints/checkpoint', 'checkpoints/params.json']
    },
    install_requires=[
        "tensorflow",
        "packaging",
        "osfclient"
    ],
    entry_points={
        'console_scripts': [
            'nextqsm = nextqsm.predict_all:cli_main'
        ]
    },
)

