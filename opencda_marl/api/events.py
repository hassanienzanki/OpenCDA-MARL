'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-09-05 15:46:15
FilePath     : /OpenCDA-MARL/opencda_marl/core/events.py
Description  : 
Copyright (c) 2025-2026 Lihao Guo and the Radar Lab, University of Arizona
SPDX-License-Identifier: Apache-2.0
'''
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union


@dataclass
class StepEvent:
    step: int
    event_id: str
    vehicle_id: Union[int, str]
    event_type: str
    metadata: Optional[Dict[str, Any]] = None
