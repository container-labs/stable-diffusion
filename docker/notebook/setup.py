from setuptools import setup, find_packages

setup(
    name='stable-app',
    version='0.0.0',
    description='',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)
