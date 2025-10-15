"""
Setup script for WR_ITSP package
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="wr_itsp",
    version="0.2.0",
    author="George Crowley",
    author_email="gcrowley1@sheffield.ac.uk",
    description="Information-Theoretic Sensor Placement for Large-Scale Sewer Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jojo0094/WR_ITSP",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
    ],
)
