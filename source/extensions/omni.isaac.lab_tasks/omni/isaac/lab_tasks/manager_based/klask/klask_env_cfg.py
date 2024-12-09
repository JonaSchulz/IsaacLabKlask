import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import SceneEntityCfg

from omni.isaac.lab_assets.klask import KLASK_CFG, KLASK_PARAMS
from .utils_manager_based import *


@configclass
class KlaskSceneCfg(InteractiveSceneCfg):
    """Configuration for Klask scene."""

    # ground plane
    # ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.007,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=KLASK_PARAMS["ball_mass_initial"]),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg(restitution=KLASK_PARAMS["ball_restitution"])
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    klask = KLASK_CFG.replace(prim_path="{ENV_REGEX_NS}/Klask")


@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    #player_actions = mdp.JointEffortActionCfg(asset_name="klask", 
    #                                           joint_names=["slider_to_peg_1", "ground_to_slider_1"],)

    #opponent_actions = mdp.JointEffortActionCfg(asset_name="klask", 
    #                                           joint_names=["slider_to_peg_2", "ground_to_slider_2"],)

    player_x = mdp.JointVelocityActionCfg(
        asset_name="klask", 
        joint_names=["slider_to_peg_1"]
    )

    player_y = mdp.JointVelocityActionCfg(
        asset_name="klask", 
        joint_names=["ground_to_slider_1"]
    )

    opponent_x = mdp.JointVelocityActionCfg(
        asset_name="klask", 
        joint_names=["slider_to_peg_2"]
    )

    opponent_y = mdp.JointVelocityActionCfg(
        asset_name="klask", 
        joint_names=["ground_to_slider_2"]
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

     # TODO: noise corruption
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # observation terms (order preserved)
        peg_1_pos = ObsTerm(func=body_xy_pos_w, params={"asset_cfg": SceneEntityCfg(
            name="klask", 
            body_names=["Peg_1"]
        )}, )#scale=2/BOARD_WIDTH)

        peg_2_pos = ObsTerm(func=body_xy_pos_w, params={"asset_cfg": SceneEntityCfg(
            name="klask", 
            body_names=["Peg_2"],
        )}, )#scale=2/BOARD_LENGTH)

        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg(
            name="klask", 
            joint_names=["ground_to_slider_1", "slider_to_peg_1", "ground_to_slider_2", "slider_to_peg_2"]
        )}, )

        ball_pos_rel = ObsTerm(func=root_xy_pos_w, params={"asset_cfg": SceneEntityCfg(name="ball")})
        
        ball_vel_rel = ObsTerm(func=root_lin_xy_vel_w, params={"asset_cfg": SceneEntityCfg(name="ball")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class AchievedGoalCfg(ObsGroup):
        ball_pos_rel = ObsTerm(func=root_xy_pos_w, params={"asset_cfg": SceneEntityCfg(name="ball")})
        ball_vel_rel = ObsTerm(func=root_lin_xy_vel_w, params={"asset_cfg": SceneEntityCfg(name="ball")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DesiredGoalCfg(ObsGroup):
        ball_in_goal = ObsTerm(func=opponent_goal_obs, params={"goal": (0.0, 0.176215)})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    observation: PolicyCfg = PolicyCfg()
    desired_goal: DesiredGoalCfg = DesiredGoalCfg()
    achieved_goal: AchievedGoalCfg = AchievedGoalCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # on reset
    add_ball_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "mass_distribution_params": KLASK_PARAMS["ball_mass_dist"],
            "operation": "add",
        },
    )

    reset_x_position_peg_1 = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["slider_to_peg_1"]),
            "position_range": (-0.14, 0.14),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_x_position_peg_2 = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["slider_to_peg_2"]),
            "position_range": (-0.14, 0.14),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_y_position_peg_1 = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["ground_to_slider_1"]),
            "position_range": (-0.2, -0.01),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_y_position_peg_2 = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["ground_to_slider_2"]),
            "position_range": (0.01, 0.2),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_ball_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.18, 0.18)},
            "velocity_range": {}
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    player_in_goal = RewTerm(
        func=in_goal, 
        params={
            "asset_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
            "goal": KLASK_PARAMS["player_goal"]
        },
        weight=KLASK_PARAMS["reward_player_in_goal"]
    )

    goal_scored = RewTerm(
        func=ball_in_goal, 
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "goal": KLASK_PARAMS["opponent_goal"],
            "max_ball_vel": KLASK_PARAMS["max_ball_vel"]
        },
        weight=KLASK_PARAMS["reward_goal_scored"]
    )

    goal_conceded = RewTerm(
        func=ball_in_goal, 
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "goal": KLASK_PARAMS["player_goal"],
            "max_ball_vel": KLASK_PARAMS["max_ball_vel"]
        },
        weight=KLASK_PARAMS["reward_goal_conceded"]
    )

    proximity_to_ball = RewTerm(
        func=distance_player_ball, 
        params={
            "player_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
            "ball_cfg": SceneEntityCfg("ball"),
        },
        weight=KLASK_PARAMS["reward_proximity_to_ball"]
    )

    ball_speed = RewTerm(
        func=speed, 
        params={
            "asset_cfg": SceneEntityCfg("ball"),
        },
        weight=KLASK_PARAMS["reward_ball_speed"]
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    goal_scored = DoneTerm(
        func=ball_in_goal, 
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "goal": KLASK_PARAMS["opponent_goal"],
            "use_delay_buffer": False
        }
    )

    goal_conceded = DoneTerm(
        func=ball_in_goal, 
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "goal": KLASK_PARAMS["player_goal"],
            "use_delay_buffer": False
        }
    )

    player_in_goal = DoneTerm(
        func=in_goal, 
        params={
            "asset_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
            "goal": KLASK_PARAMS["player_goal"], 
            "use_delay_buffer": False
        }
    )


@configclass
class KlaskEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene = KlaskSceneCfg(num_envs=1, env_spacing=1.0)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    episode_length_s = 3.0

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = (0.0, 0.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz
        self.termination_delay = {}