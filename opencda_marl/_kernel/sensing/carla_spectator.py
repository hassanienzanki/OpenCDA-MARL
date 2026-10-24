'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-08-28 20:22:37
FilePath     : /OpenCDA-MARL/opencda_marl/sensing/carla_spectator.py
Description  : 
Copyright (c) 2025-2026 Lihao Guo and the Radar Lab, University of Arizona
SPDX-License-Identifier: Apache-2.0
'''
import carla

from loguru import logger
from typing import Dict, Any, Optional


class CarlaSpectator:
    """
    Manager for CARLA spectator camera control.
    """

    def __init__(self, world: carla.World, config: Optional[Dict[str, Any]] = None):
        self.world = world
        self.spectator = world.get_spectator()
        self.config = config or {}

        # Current view state
        self.target_agent_id = None
        self.follow_offset = carla.Vector3D(0, 0, 50)  # Default follow offset
        
        self._view_presets = {
            'intersection_bird_eye': {
                'position': carla.Location(x=0, y=0, z=100),
                'rotation': carla.Rotation(pitch=-90, yaw=0, roll=0),
                'description': 'Bird\'s eye view of intersection center'
            }
        }

        if self._apply_configured_fixed_view():
            self.world.tick()
            return

        preset = self.config.get('preset')
        if preset and preset in self._view_presets:
            self.use_preset(preset)
            self.world.tick()

    def _apply_configured_fixed_view(self) -> bool:
        position_cfg = self.config.get('position') or self.config.get('location')
        rotation_cfg = self.config.get('rotation')
        if position_cfg is None or rotation_cfg is None:
            return False
        position = self._location_from_config(position_cfg)
        rotation = self._rotation_from_config(rotation_cfg)
        success = self.set_fixed_view(position, rotation)
        if success:
            logger.debug("Applied configured fixed spectator view")
        return success

    @staticmethod
    def _cfg_float(cfg: Any, key: str, default: float) -> float:
        if hasattr(cfg, "get"):
            return float(cfg.get(key, default))
        return float(getattr(cfg, key, default))

    @classmethod
    def _location_from_config(cls, cfg: Any) -> carla.Location:
        return carla.Location(
            x=cls._cfg_float(cfg, "x", 0.0),
            y=cls._cfg_float(cfg, "y", 0.0),
            z=cls._cfg_float(cfg, "z", 0.0),
        )

    @classmethod
    def _rotation_from_config(cls, cfg: Any) -> carla.Rotation:
        return carla.Rotation(
            pitch=cls._cfg_float(cfg, "pitch", 0.0),
            yaw=cls._cfg_float(cfg, "yaw", 0.0),
            roll=cls._cfg_float(cfg, "roll", 0.0),
        )

    def use_preset(self, preset_name: str) -> bool:
        """
        Apply predefined view preset.
        """
        if preset_name not in self._view_presets:
            available = list(self._view_presets.keys())
            logger.error(f"Unknown preset '{preset_name}'. Available: {available}")
            return False
        
        preset = self._view_presets[preset_name]
        success = self.set_fixed_view(preset['position'], preset['rotation'])
        
        if success:
            logger.debug(f"Applied preset '{preset_name}': {preset['description']}")
        
        return success
    
    # --------------------------------------------------------------------- #
    # Set view
    # --------------------------------------------------------------------- #    
    def set_fixed_view(self, position: carla.Location, 
                      rotation: carla.Rotation) -> bool:
        """
        Set fixed camera position and rotation.
        """
        try:
            transform = carla.Transform(position, rotation)
            self.spectator.set_transform(transform)
            
            logger.debug(f"Set fixed view at {position}, rotation {rotation}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set fixed view: {e}")
            return False
    
    def get_current_view(self) -> Dict[str, Any]:
        """
        Get current camera view information.
        
        Returns
        -------
        view_info : dict
            Current view mode, position, and rotation
        """
        current_transform = self.spectator.get_transform()
        
        return {
            'position': current_transform.location,
            'rotation': current_transform.rotation,
        }
