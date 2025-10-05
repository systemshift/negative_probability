"""
Quasi-Probability Reinforcement Learning (QP-RL) v2

A clean research implementation of RL with negative probabilities.
"""

from .quasi_probability_agent import QuasiProbabilityAgent, ClassicalQAgent
from .grid_environment import GridWorld

__version__ = "2.0.0"
__all__ = ["QuasiProbabilityAgent", "ClassicalQAgent", "GridWorld"]
