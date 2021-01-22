# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

requirements = [
    'click==7.0',
    'numpy>=1.16.5,<1.19',
    'keras==2.2.5',
    'keras-rl==0.4.2',
    'tensorflow==1.14.0',
    'cloudpickle==1.2',
    'gym[atari]==0.14.0',
    'pandas==1.0.0',
    'h5py==2.10'
]

test_requirements = [
    'flake8',
    'nose2'
]

setup(
    name='rlsp',
    version='1.1.0',
    description='Self-Learning Network and Service Coordination Using Deep Reinforcement Learning',
    url='https://github.com/RealVNF/deep-rl-network-service-coordination',
    author='RealVNF',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    python_requires=">=3.6.*",
    install_requires=requirements + test_requirements,
    tests_require=test_requirements,
    test_suite='nose2.collector.collector',
    zip_safe=False,
    entry_points={
        'console_scripts': [
            "rlsp=rlsp.agents.main:cli",
        ],
    },
)
