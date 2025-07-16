from setuptools import find_packages, setup

# Read requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='eeg_analysis',
    packages=find_packages(),
    version='0.1.0',
    description='EEG analysis for depression remission prediction.',
    author='Your name (or your organization/company/team)',
    license='MIT',
    install_requires=requirements
)