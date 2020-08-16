import os
import sys
from setuptools import setup, find_packages


def read_requirements(filename):
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', filename)
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


with open('README.md', 'rb') as f:
    readme = f.read().decode('utf8', 'ignore')


if "--raspi" in sys.argv:
    filename = 'requirements_raspi.txt'
    sys.argv.remove("--raspi")
else:
    filename = 'requirements.txt'

setup(
    name='roboskin',
    version='0.0.1',
    description='Codes for roboskin project',
    long_description=readme,
    author='Kandai Watanabe',
    author_email='kandai.watanabe@colorado.edu',
    url='https://github.com/HIRO-group/roboskin',
    install_requires=read_requirements(filename),
    packages=find_packages(exclude=('tests', 'docs')),
    test_suite='tests'
)
