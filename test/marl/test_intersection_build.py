'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-28 17:05:42
FilePath     : /OpenCDA-MARL/test/marl/test_intersection_build.py
Description  : 
Copyright (c) 2025-2026 Lihao Guo and the Radar Lab, University of Arizona
SPDX-License-Identifier: Apache-2.0
'''
import sys
import os
from pathlib import Path
import pytest
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# set log level to INFO
logger.remove()
logger.add(sys.stdout, level="INFO")


def _build_intersection_scenario():
    from opencda_marl._kernel.scenarios import ScenarioBuilder
    from opencda.core.common.cav_world import CavWorld
    
    scenario_builder = ScenarioBuilder()
    cfg = scenario_builder.create_intersection_scenario(
        weather='clear',
        town='intersection'
    )
    
    # add visualization to cfg
    if 'scenario' not in cfg:
        cfg['scenario'] = {}
    if 'traffic' not in cfg['scenario']:
        cfg['scenario']['traffic'] = {}
    if 'planner' not in cfg['scenario']['traffic']:
        cfg['scenario']['traffic']['planner'] = {}
    cfg['scenario']['traffic']['planner']['visualize'] = {
        'route_line': False,
        'junction_center': True,
        'text': True,
        'life_time': 1000000
    }
    
    cav_world = CavWorld(apply_ml=cfg['opt']['apply_ml'])
    scenario_manager = scenario_builder.build_from_config(
        config=cfg,
        apply_ml=cfg['opt']['apply_ml'],
        cav_world=cav_world
    )
    return scenario_manager


@pytest.mark.skipif(
    os.environ.get("OPENCDA_RUN_CARLA_INTEGRATION") != "1",
    reason="requires a running CARLA server and local intersection map",
)
def test_intersection_build():
    scenario_manager = _build_intersection_scenario()
    assert scenario_manager.world is not None

if __name__ == "__main__":
    from opencda_marl._kernel.sensing import CarlaSpectator
    
    scenario_manager = _build_intersection_scenario()
    
    world = scenario_manager.world
    spectator = CarlaSpectator(world=world, config={
        'preset': 'intersection_bird_eye'
    })
    
    traffic_manager = scenario_manager.traffic_manager
    planner = traffic_manager.planner
    
    junctions = planner.get_junctions()
    for junction_id, junction_data in junctions.items():
        # show how many routes in each junction
        print(f"Junction {junction_id} has {len(junction_data['routes'])} routes")
        
        # check first route
        route = junction_data['routes'][0]
        print(route)
