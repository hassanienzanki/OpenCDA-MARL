# -*- coding: utf-8 -*-
"""
Controller interface
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import importlib


class ControlManager(object):
    """
    Controller manager that is used to select
    and call different controller's functions.

    Parameters
    ----------
    control_config : dict
        The configuration dictionary of the control manager module.

    Attributes
    ----------
    controller : opencda object.
        The controller object of the OpenCDA framwork.
    """

    def __init__(self, control_config):
        controller_type = control_config['type']
        controller = getattr(
            importlib.import_module(
                "opencda.core.actuation.%s" %
                controller_type), 'Controller')
        self.controller = controller(control_config['args'])

    def update_info(self, ego_pos, ego_speed):
        """
        Update ego vehicle information for controller.
        """
        self.controller.update_info(ego_pos, ego_speed)

    def run_step(self, target_speed, waypoint):
        """
        Execute current controller step.
        """
        control_command = self.controller.run_step(target_speed, waypoint)
        return control_command

    # <OpenCDA-MARL> debug info for TensorBoard and CARLA overlay
    #
    # The PID controller records steer/throttle/brake each tick (near-zero cost).
    #
    # To display in CARLA, wire it through VehicleManager.run_step():
    #
    #   # in VehicleManager.run_step(), after self.controller.run_step():
    #   if hasattr(self.agent, 'record_controller_debug'):
    #       self.agent.record_controller_debug(self.controller.get_last_debug_info())
    #
    # Then in VanillaAgent, implement record_controller_debug() and use the
    # dict in draw_debug_preview() — gated by local_planner.debug_trajectory.
    #
    # Config: agents.vanilla.local_planner.debug_trajectory: true
    # Keys:   target_speed, current_speed, target_location, steer, throttle, brake
    def get_last_debug_info(self):
        """Return latest controller debug info dict from the PID controller."""
        return getattr(self.controller, "last_debug_info", {})
    # </OpenCDA-MARL>
