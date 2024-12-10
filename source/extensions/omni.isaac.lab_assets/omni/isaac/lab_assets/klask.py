import numpy as np
import os

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg


# Configuration for Klask articulation

KLASK_PARAMS = {
    "player_goal": (0.0, -0.176215, 0.01905),
    "opponent_goal": (0.0, 0.176215, 0.01905),
    #"player_goal": (0.0, -0.176215, 0.08),
    #"opponent_goal": (0.0, 0.176215, 0.08),
    "ball_restitution": 0.8,
    "ball_mass_initial": 0.001,
    "ball_mass_dist": (0.001, 0.005),
    "max_ball_vel": 5.0,
    "reward_player_in_goal": 0.0,
    "reward_goal_scored": 10.0,
    "reward_goal_conceded": -10.0,
    "reward_proximity_to_ball": 0.0,
    "reward_ball_speed": 0.0
}

KLASK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.getcwd(), "source/extensions/omni.isaac.lab_assets/omni/isaac/lab_assets/klask.usd"),
    ),
    actuators={
        "peg_1x_actuator": IdealPDActuatorCfg(
            joint_names_expr=["slider_to_peg_1"],
            stiffness=0.0,
            damping=10.0,
            velocity_limit=1.0,
            effort_limit=1.0
        ),
        "peg_1y_actuator": IdealPDActuatorCfg(
            joint_names_expr=["ground_to_slider_1"],
            stiffness=0.0,
            damping=10.0,
            velocity_limit=1.0,
            effort_limit=1.0
        ),
        "peg_2x_actuator": IdealPDActuatorCfg(
            joint_names_expr=["slider_to_peg_2"],
            stiffness=0.0,
            damping=10.0,
            velocity_limit=1.0,
            effort_limit=1.0
        ),
        "peg_2y_actuator": IdealPDActuatorCfg(
            joint_names_expr=["ground_to_slider_2"],
            stiffness=0.0,
            damping=10.0,
            velocity_limit=1.0,
            effort_limit=1.0
        ),        
    },
)