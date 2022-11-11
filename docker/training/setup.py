from setuptools import find_packages, setup

setup(
    name='stable-train',
    version='0.0.0',
    description='',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)
