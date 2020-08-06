# Lint as: python3
"""Compatibility layer for predictions_lib.

This file imports the predictions_lib either from google or public and
exports the build function.
"""
from deepmath.public import predictions_builder

build = predictions_builder.build
