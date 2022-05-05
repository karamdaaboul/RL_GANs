from distutils.core import setup
from setuptools import find_packages

setup(
    name='rl_gans',
    packages=find_packages(),
    version='0.0.1',
    description='GANS RL',
    long_description=open('./README.md').read(),
    author='KD',
    zip_safe=True,
    license='MIT'
)
