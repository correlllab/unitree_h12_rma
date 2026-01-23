import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from agile.rl_env.mdp.actuators.actuators_cfg import DelayedDCMotorCfg, DelayedImplicitActuatorCfg

MAX_DELAY_PHY_STEPS = 4
MIN_DELAY_PHY_STEPS = 0

H1_2_USD_PATH = "/home/jensen/unitree_model/H12_handless/h12_handless/usd/h1_2_handless.usd"
H1_2_URDF_PATH = "/home/niraj/isaac_projects/unitree_h12_sim2sim/isaaclab/source/unitree_h12_sim2sim/unitree_h12_sim2sim/robots/h1-2/h1_2_handless.urdf"

LEG_JOINT_NAMES = [
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
]
ANKLE_JOINT_NAMES = [
    ".*_ankle_.*_joint",
]
WAIST_JOINT_NAMES = [
    "torso_joint",
]
ARM_JOINT_NAMES = [
    ".*_shoulder_.*_joint",
    ".*_elbow_joint",
    ".*_wrist_.*_joint",
]
HAND_JOINT_NAMES = [
    ".*_hand_.*",
]
FEET_LINK_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]
DEFAULT_PELVIS_HEIGHT = 1.0

# Using the delayed DC motor model.
# replaced USD with URDF ! 
H1_2_27DOF_DELAYED_DC_MOTOR = ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
        # Use the local USD asset bundled in this package (unitree_model/usd/...)
        # usd_path= "/home/niraj/isaac_projects/H12_Bullet_Time/h12_bullet_time/source/h12_bullet_time/h12_bullet_time/assets/robots/unitree_model/usd/h1_2_handless/h1_2_handless.usd",
        fix_base = False,
        replace_cylinders_with_capsules=True,

        #laptop path - fixed with isaac_projects
        asset_path= H1_2_URDF_PATH,
        activate_contact_sensors=True,

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4
        ),

        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    soft_joint_pos_limit_factor=0.9,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.00),
        joint_pos={
            ".*_hip_pitch_joint": -0.10,
            ".*_knee_joint": 0.30,
            ".*_ankle_pitch_joint": -0.20,
        },
        joint_vel={".*": 0.0},
    ),
    # this can be lower than true values but starting with true values
    actuators={
        "legs": DelayedDCMotorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 220.0,
                ".*_hip_roll_joint": 220.0,
                ".*_hip_pitch_joint": 220.0,
                ".*_knee_joint": 360.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 32.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 220.0,
                ".*_hip_roll_joint": 220.0,
                ".*_hip_pitch_joint": 220.0,
                ".*_knee_joint": 300.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.5,
                ".*_hip_roll_joint": 2.5,
                ".*_hip_pitch_joint": 2.5,
                ".*_knee_joint": 4.0,
            },
            armature={
                ".*_hip_.*": 0.02,
                ".*_knee_joint": 0.02,
            },
            saturation_effort=180.0,
            min_delay=0,
            max_delay=MAX_DELAY_PHY_STEPS,
        ),
        "feet": DelayedDCMotorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness={
                ".*_ankle_pitch_joint": 40.0,
                ".*_ankle_roll_joint": 40.0,
            },
            damping={
                ".*_ankle_pitch_joint": 2.0,
                ".*_ankle_roll_joint": 2.0,
            },
            effort_limit_sim={
                ".*_ankle_pitch_joint": 75.0,
                ".*_ankle_roll_joint": 75.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 37.0,
                ".*_ankle_roll_joint": 37.0,
            },
            armature=0.02,
            saturation_effort=80.0,
            min_delay=0,
            max_delay=MAX_DELAY_PHY_STEPS,
        ),
        "waist": DelayedDCMotorCfg(
            joint_names_expr=[
                "torso_joint",
            ],
            effort_limit_sim={
                "torso_joint": 220.0,
            },
            velocity_limit_sim={
                "torso_joint": 37.0,
            },
            stiffness={
                "torso_joint": 300.0,
            },
            damping={
                "torso_joint": 5.0,
            },
            armature=0.02,
            saturation_effort=120.0,
            min_delay=0,
            max_delay=MAX_DELAY_PHY_STEPS,
        ),
        "arms": DelayedDCMotorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 120.0,
                ".*_shoulder_roll_joint": 120.0,
                ".*_shoulder_yaw_joint": 75.0,
                ".*_elbow_joint": 120.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 25.0,
                ".*_wrist_yaw_joint": 25.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 120.0,
                ".*_shoulder_roll_joint": 120.0,
                ".*_shoulder_yaw_joint": 75.0,
                ".*_elbow_joint": 120.0,
                ".*_wrist_.*_joint": 25.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 2.0,
                ".*_shoulder_roll_joint": 1.0,
                ".*_shoulder_yaw_joint": 0.4,
                ".*_elbow_joint": 1.0,
                ".*_wrist_.*_joint": 0.2,
            },
            armature={
                ".*_shoulder_.*": 0.02,
                ".*_elbow_.*": 0.02,
                ".*_wrist_.*_joint": 0.02,
            },
            saturation_effort=40.0,
            min_delay=0,
            max_delay=0,
        ),
    },
)

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
# H1_2
ARMATURE_M107_15 = 0.063259741
ARMATURE_M107_24 = 0.160478022

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2
# H1_2
STIFFNESS_M107_15 = ARMATURE_M107_15 * NATURAL_FREQ**2
STIFFNESS_M107_24 = ARMATURE_M107_24 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ
# H1_2
DAMPING_M107_15 = 2.0 * DAMPING_RATIO * ARMATURE_M107_15 * NATURAL_FREQ
DAMPING_M107_24 = 2.0 * DAMPING_RATIO * ARMATURE_M107_24 * NATURAL_FREQ

H1_2_27DOF = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=H1_2_USD_PATH,
        # Disable hands to accelerate training.
        variants={"Physics": "PhysX", "left_hand": "None", "right_hand": "None"},
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    soft_joint_pos_limit_factor=0.9,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        joint_pos={
            ".*_hip_pitch_joint": -0.10,
            ".*_knee_joint": 0.30,
            ".*_ankle_pitch_joint": -0.20,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 220.0,
                ".*_hip_roll_joint": 220.0,
                ".*_hip_pitch_joint": 220.0,
                ".*_knee_joint": 360.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_M107_24,
                ".*_hip_roll_joint": STIFFNESS_M107_24,
                ".*_hip_yaw_joint": STIFFNESS_M107_15,
                ".*_knee_joint": STIFFNESS_M107_24,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_M107_24,
                ".*_hip_roll_joint": DAMPING_M107_24,
                ".*_hip_yaw_joint": DAMPING_M107_15,
                ".*_knee_joint": DAMPING_M107_24,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_M107_24,
                ".*_hip_roll_joint": ARMATURE_M107_24,
                ".*_hip_yaw_joint": ARMATURE_M107_15,
                ".*_knee_joint": ARMATURE_M107_24,
            },
            min_delay=MIN_DELAY_PHY_STEPS,
            max_delay=MAX_DELAY_PHY_STEPS,
        ),
        "feet": DelayedImplicitActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_7520_14,
            damping=2.0 * DAMPING_7520_14,
            armature=2.0 * ARMATURE_7520_14,
            min_delay=MIN_DELAY_PHY_STEPS,
            max_delay=MAX_DELAY_PHY_STEPS,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "torso_.*_joint",
            ],
            effort_limit_sim={
                "torso_joint": 200.0,
            },
            velocity_limit_sim={
                "torso_joint": 37.0,
            },
            stiffness={
                "torso_joint": 300.0,
            },
            damping={
                "torso_joint": 5.0,
            },
            armature=0.03,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 120.0,
                ".*_shoulder_roll_joint": 120.0,
                ".*_shoulder_yaw_joint": 120.0,
                ".*_elbow_joint": 120.0,
                ".*_wrist_roll_joint": 75.0,
                ".*_wrist_pitch_joint": 75.0,
                ".*_wrist_yaw_joint": 75.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 90.0,
                ".*_shoulder_roll_joint": 60.0,
                ".*_shoulder_yaw_joint": 20.0,
                ".*_elbow_joint": 90.0,
                ".*_wrist_.*_joint": 4.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 2.0,
                ".*_shoulder_roll_joint": 1.0,
                ".*_shoulder_yaw_joint": 0.4,
                ".*_elbow_joint": 1.0,
                ".*_wrist_.*_joint": 0.2,
            },
            armature={
                ".*_shoulder_.*": 0.03,
                ".*_elbow_.*": 0.03,
                ".*_wrist_.*_joint": 0.03,
            },
        ),
    },
)

H1_2_ACTION_SCALE_LOWER = {}
for actuator_name, actuator_cfg in H1_2_27DOF.actuators.items():
    e = actuator_cfg.effort_limit_sim
    s = actuator_cfg.stiffness
    if actuator_name != "legs" and actuator_name != "feet":
        continue
    names = actuator_cfg.joint_names_expr
    if not isinstance(e, dict):
        e = dict.fromkeys(names, e)
    if not isinstance(s, dict):
        s = dict.fromkeys(names, s)
    for n in names:
        if n in e and n in s and s[n]:
            H1_2_ACTION_SCALE_LOWER[n] = 0.25 * e[n] / s[n]