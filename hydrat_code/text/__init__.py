"""
Module for interacting with pre-trained text classifiers.
This bypasses some of the hydrat classifier framework,
providing classes that allow for interacting with tools
that expect raw text and produce a class label from an
externally-specified set.
"""

from interface import TextClassifier, ProxyExperiment, DatasetExperiment
