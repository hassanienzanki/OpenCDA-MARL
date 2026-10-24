'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-30 12:33:46
FilePath     : /OpenCDA-MARL/opencda_marl/adapter/exception.py
Description  : Custom exceptions for MARL adapter
Copyright (c) 2025-2026 Lihao Guo and the Radar Lab, University of Arizona
SPDX-License-Identifier: Apache-2.0
'''
class CollisionException(Exception):
    """Exception raised when a vehicle collides"""
    pass


class PlannerFailureException(Exception):
    """Exception raised when planner/path construction fails for a vehicle."""
    pass
