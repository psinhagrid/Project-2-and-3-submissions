from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='Query Classifier',
    version='1.0.0',
    packages=find_packages(),
    install_requires=requirements,
    author='Prakhar Sinha',
    description='A Python package for performing classification of user query into several classes to optimize output. ',
    url='https://github.com/psinhagrid/Project_2_submission',
)