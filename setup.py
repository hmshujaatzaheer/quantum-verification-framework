"""
Setup script for Unified Quantum Verification Framework

Author: H M Shujaat Zaheer
Email: shujabis@gmail.com
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantum-verification-framework",
    version="0.1.0",
    author="H M Shujaat Zaheer",
    author_email="shujabis@gmail.com",
    description="Fault-tolerant verification of quantum random sampling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hmshujaatzaheer/quantum-verification-framework",
    project_urls={
        "Bug Tracker": "https://github.com/hmshujaatzaheer/quantum-verification-framework/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
)
