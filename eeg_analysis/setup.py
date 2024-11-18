from setuptools import setup, find_packages

setup(
    name="eeg_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.22.4",
        "pandas>=1.3.5",
        "scipy>=1.10.0",
        "scikit-learn>=1.0.2",
        "mlflow>=2.8.0",
        "nolds>=0.5.2",
        "pyarrow>=14.0.1",
        "pyyaml>=6.0.1",
        "python-json-logger>=2.0.7",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "pylint>=2.17.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "eeg_pipeline=eeg_analysis.run_pipeline:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="EEG analysis pipeline for depression remission prediction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/eeg_analysis",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)