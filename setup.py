#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="qp_rl",
    version="0.1.0",
    description="Quasi-Probability Reinforcement Learning for Bidirectional Time",
    author=".",
    author_email=".",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
