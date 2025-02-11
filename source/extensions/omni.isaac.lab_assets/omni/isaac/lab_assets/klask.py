import numpy as np
import os

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg, DelayedPDActuatorCfg


# Configuration for Klask articulation

KLASK_PARAMS = {
    "decimation": 4,
    "physics_dt": 0.005,
    "player_goal": (0.0, -0.176215, 0.01905),
    "opponent_goal": (0.0, 0.176215, 0.01905),
    #"player_goal": (0.0, -0.176215, 0.08),
    #"opponent_goal": (0.0, 0.176215, 0.08),
    "ball_restitution": 0.8,
    "ball_static_friction": 0.3,
    "ball_dynamic_friction": 0.2,
    "ball_mass_initial": 0.001,
    "ball_mass_dist": (0.001, 0.005),
    "max_ball_vel": 5.0,    # maximum speed the ball may have for a goal to be counted
    "rewards": {
        "player_in_goal": 0.0,
        "goal_scored": 10.0,
        "goal_conceded": -10.0,
        "distance_player_goal": 0.0,
        "distance_ball_opponent_goal": 1.0,
        "ball_speed": 0.0,
        "distance_player_ball_own_half": 0.0,
        "ball_stationary": 0.0,
        "collision_player_ball": 0.0,
        "ball_in_own_half": 0.0
    },
    "terminations": {
        "goal_scored": False,
        "goal_conceded": False,
        "player_in_goal": False
    }
}

KLASK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.getcwd(), "source/extensions/omni.isaac.lab_assets/omni/isaac/lab_assets/klask.usd"),
    ),
    actuators={
        "peg_1x_actuator": DelayedPDActuatorCfg(
            joint_names_expr=["slider_to_peg_1"],
            stiffness=0.0,
            damping=10.0,
            velocity_limit=1.0,
            effort_limit=1.0,
            min_delay=int(0.2/KLASK_PARAMS["physics_dt"]),
            max_delay=int(0.2/KLASK_PARAMS["physics_dt"])
        ),
        "peg_1y_actuator": DelayedPDActuatorCfg(
            joint_names_expr=["ground_to_slider_1"],
            stiffness=0.0,
            damping=10.0,
            velocity_limit=1.0,
            effort_limit=1.0,
            min_delay=int(0.2/KLASK_PARAMS["physics_dt"]),
            max_delay=int(0.2/KLASK_PARAMS["physics_dt"])
        ),
        "peg_2x_actuator": DelayedPDActuatorCfg(
            joint_names_expr=["slider_to_peg_2"],
            stiffness=0.0,
            damping=10.0,
            velocity_limit=1.0,
            effort_limit=1.0,
            min_delay=int(0.2/KLASK_PARAMS["physics_dt"]),
            max_delay=int(0.2/KLASK_PARAMS["physics_dt"])
        ),
        "peg_2y_actuator": DelayedPDActuatorCfg(
            joint_names_expr=["ground_to_slider_2"],
            stiffness=0.0,
            damping=10.0,
            velocity_limit=1.0,
            effort_limit=1.0,
            min_delay=int(0.2/KLASK_PARAMS["physics_dt"]),
            max_delay=int(0.2/KLASK_PARAMS["physics_dt"])
        ),        
    },
)