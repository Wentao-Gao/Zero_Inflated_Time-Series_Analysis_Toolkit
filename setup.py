#!/usr/bin/env python3
"""
Setup script for Zero-Inflated Time Series Analysis Toolkit
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="zero-inflated-timeseries",
    version="1.0.0",
    author="Zero-Inflated Research Team",
    author_email="contact@zero-inflated-toolkit.org",
    description="A comprehensive toolkit for zero-inflated time series analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/zero-inflated-comprehensive",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "viz": [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "zi-timeseries=zero_inflated_comprehensive.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "time series",
        "zero inflated",
        "machine learning",
        "deep learning",
        "forecasting",
        "statistical modeling",
        "pytorch",
        "scikit-learn",
    ],
    project_urls={
        "Documentation": "https://zero-inflated-timeseries.readthedocs.io/",
        "Bug Reports": "https://github.com/your-username/zero-inflated-comprehensive/issues",
        "Source": "https://github.com/your-username/zero-inflated-comprehensive",
        "Changelog": "https://github.com/your-username/zero-inflated-comprehensive/blob/main/CHANGELOG.md",
    },
)