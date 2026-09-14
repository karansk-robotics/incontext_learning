"""Layout constants for the ROBOTIS AI Worker FFW-SG2 <-> ICRT bridge.

Two vector layouts matter here.

``JOINT`` (22-D) is what ``physical_ai_server`` records into LeRobot datasets for
``ffw_sg2_rev1``.  Both ``action`` and ``observation.state`` use it: the converter
in ``physical_ai_tools`` orders every field by the per-group ``joint_order`` lists
and concatenates the groups in ``joint_list`` order.

``CARTESIAN`` (20-D) is what we feed ICRT: each arm as an end-effector pose in the
``arm_base_link`` frame, rotation in the 6D continuous representation ICRT already
uses, plus a normalised gripper scalar.  Head, lift and mobile base are dropped --
they are held at their episode-start pose during rollout.
"""

# --- 22-D joint layout, as recorded by physical_ai_server ---------------------

ARM_L_JOINTS = [f'arm_l_joint{i}' for i in range(1, 8)]
ARM_R_JOINTS = [f'arm_r_joint{i}' for i in range(1, 8)]
GRIPPER_L_JOINT = 'gripper_l_joint1'
GRIPPER_R_JOINT = 'gripper_r_joint1'
HEAD_JOINTS = ['head_joint1', 'head_joint2']
LIFT_JOINT = 'lift_joint'
BASE_DIMS = ['linear_x', 'linear_y', 'angular_z']

JOINT_ORDER = (
    ARM_L_JOINTS + [GRIPPER_L_JOINT]
    + ARM_R_JOINTS + [GRIPPER_R_JOINT]
    + HEAD_JOINTS + [LIFT_JOINT] + BASE_DIMS
)
JOINT_DIM = len(JOINT_ORDER)  # 22, the full FFW-SG2 recording

# Not every AI Worker records all 22. ROBOTIS's own published datasets show the
# width varies by robot_type -- ffw_bg2_rev4_custom and ffw_arm_only both record
# 16, being the two arms and their grippers with no head, lift or mobile base.
#
# Crucially the arm/gripper block is a stable PREFIX of the full layout: indices
# 0-15 mean the same thing in both. So anything >= 16 wide can be retargeted,
# and only the trailing head/lift/base entries come and go. The 20-D Cartesian
# space we feed ICRT is built purely from this prefix, which is why it is
# unaffected by the difference.
ARMS_ONLY_DIM = 16

# Slices into the 22-D joint vector.
SLICE_ARM_L = slice(0, 7)
SLICE_GRIP_L = 7
SLICE_ARM_R = slice(8, 15)
SLICE_GRIP_R = 15
SLICE_HEAD = slice(16, 18)
SLICE_LIFT = 18
SLICE_BASE = slice(19, 22)

# --- 20-D Cartesian layout, as fed to ICRT -----------------------------------

ARM_CART_DIM = 10  # xyz(3) + rot6d(6) + gripper(1)
CARTESIAN_DIM = 2 * ARM_CART_DIM  # 20

SLICE_CART_L = slice(0, 10)
SLICE_CART_R = slice(10, 20)

# --- Extra channels beyond the two arms --------------------------------------
#
# The arms are the whole story for CONTROL, but not for CONTEXT. Two joints the
# policy does not command still matter to it:
#
#   head (2)  carries cam_head. It does not touch end-effector geometry -- the
#             head is a separate branch from the arms -- but it decides what the
#             policy's main camera sees. Observed so the policy can interpret an
#             image without inferring the viewpoint from pixels alone. NEVER
#             commanded: it stays wherever the operator set it.
#
#   lift (1)  sits INSIDE the arm chain, so with poses in base_link a different
#             torso height means a different z for the same arm pose. Observed
#             AND commanded -- if a demonstrator raised the torso to reach a high
#             box, that is part of the skill, and a policy that cannot reproduce
#             it will try to reach with the arm alone and fail.
#
# So proprio carries head+lift; action carries lift only.
# The head was here until 2026-09-14 and was REMOVED, deliberately.
#
# Including it looked right — tell the policy its own gaze so it can interpret
# the camera image rather than inferring viewpoint from pixels. But each task in
# this data was recorded at its own fixed head angle:
#     blue box   head_joint1 = +0.4663   (spread 0.0001 across 50 episodes)
#     ECU part   head_joint1 = +0.6427   (spread 0.0001 across 39 episodes)
#     yellow     head_joint1 = +0.7839   (spread 0.0001 across 31 episodes)
# Zero overlap. One number, to three decimals, identifies the task exactly.
#
# That is a task LABEL in the input vector. ICRT's whole claim is that the task
# is inferred from the demonstration; gradient descent would take the free answer
# instead, because it is the cheapest feature available. The scene is a
# correlated cue, which is bad enough — head angle is an exact one.
#
# Put it back if same-scene/different-task episodes are ever recorded at one
# camera angle: the channel becomes constant, the leakage vanishes, and the
# original argument for including it holds again.
PROPRIO_EXTRA = [LIFT_JOINT]                    # 1
ACTION_EXTRA = [LIFT_JOINT]                     # 1

PROPRIO_DIM = CARTESIAN_DIM + len(PROPRIO_EXTRA)   # 23
ACTION_DIM = CARTESIAN_DIM + len(ACTION_EXTRA)     # 21 (before ICRT's eos channel)

# Slices into the extra tail. Index into the 22-D recorded joint vector that each
# extra channel is copied from, in order.
SLICE_PROPRIO_EXTRA = slice(CARTESIAN_DIM, PROPRIO_DIM)
SLICE_ACTION_EXTRA = slice(CARTESIAN_DIM, ACTION_DIM)
PROPRIO_EXTRA_SRC = [18]            # lift_joint only; head removed, see above
ACTION_EXTRA_SRC = [18]             # lift_joint

# Within one 10-D arm block.
SLICE_BLOCK_POS = slice(0, 3)
SLICE_BLOCK_ROT6D = slice(3, 9)
BLOCK_GRIPPER = 9

# --- Kinematics --------------------------------------------------------------

# End-effector poses are expressed in base_link -- the robot's canonical frame,
# between the wheels -- not in arm_base_link at the shoulders.
#
# The consequence is that lift_joint sits INSIDE each arm's kinematic chain, so a
# chain has 8 actuated joints (lift + 7 revolute) rather than 7, and EEF height
# includes the torso position. We do not command the lift: it is read from the
# recording for FK, and held fixed while IK solves the 7 arm joints. See
# FFWKinematics for how that is enforced.
FK_ROOT_LINK = 'base_link'
LIFT_CHAIN_JOINT = 'lift_joint'
EEF_LINK_L = 'end_effector_l_link'
EEF_LINK_R = 'end_effector_r_link'

# RH-P12-RN gripper: the commanded joint travels [0, 1.1] rad over a 0-107.6 mm
# span, and the sense is **0 = OPEN, 1.1 = CLOSED** -- the fingers close as the
# value rises.
#
# Verified against the ECU recordings rather than assumed: in episode_000000 the
# normalised signal sits at 0.09 while the arm approaches, rises to 0.72 for the
# transport phase, and drops back at release. The wrist camera at those two
# frames shows fingers wide apart at 0.09 and clamped on the part at 0.72.
#
# We normalise to [0, 1] preserving that sense, so **0 = open, 1 = closed**.
# The mapping is monotonic and round-trips, so the model is indifferent -- but
# the sense matters wherever a threshold is applied to the raw value, e.g.
# ICRT's `binary_gripper` option, which treats `> 0.5` as one discrete state.
GRIPPER_LIMITS = (0.0, 1.1)

# Revolute limits straight out of ffw_sg2_follower.urdf. Note joint2 is
# asymmetric between arms -- the left elbow opens positive, the right negative.
ARM_JOINT_LIMITS_L = [
    (-3.14, 3.14), (0.0, 3.14), (-3.14, 3.14), (-2.9361, 1.0786),
    (-3.14, 3.14), (-1.57, 1.57), (-1.5804, 1.8201),
]
ARM_JOINT_LIMITS_R = [
    (-3.14, 3.14), (-3.14, 0.0), (-3.14, 3.14), (-2.9361, 1.0786),
    (-3.14, 3.14), (-1.57, 1.57), (-1.8201, 1.5804),
]

# Every FFW-SG2 revolute joint reports the same velocity ceiling.
JOINT_VELOCITY_LIMIT = 4.8  # rad/s

# --- Cameras -----------------------------------------------------------------
# Keys as physical_ai_server names them; ICRT stacks them on the camera axis in
# exactly this order, so it must stay stable between conversion and inference.
CAMERA_KEYS = ['cam_head', 'cam_wrist_left', 'cam_wrist_right']
NUM_CAMERAS = len(CAMERA_KEYS)

# ROBOTIS has two naming conventions for the same physical cameras. Datasets
# recorded with physical_ai_tools use the names above; cyclo_intelligence's robot
# configs (shared/robot_configs/ffw_*_config.yaml) name them differently:
#
#   cyclo               physical_ai_tools   topic
#   cam_left_head       cam_head            /zed/zed_node/left/image_rect_color
#   cam_left_wrist      cam_wrist_left      /camera_left/...
#   cam_right_wrist     cam_wrist_right     /camera_right/...
#   cam_right_head      cam_head_right      /zed/zed_node/right/image_rect_color
#
# The head "pair" is one ZED stereo camera, not two viewpoints. We take the LEFT
# eye only and ignore the right, which keeps a single consistent head view across
# every dataset regardless of which tool recorded it.
CAMERA_ALIASES = {
    'cam_head': ['cam_head', 'cam_left_head'],
    'cam_wrist_left': ['cam_wrist_left', 'cam_left_wrist'],
    'cam_wrist_right': ['cam_wrist_right', 'cam_right_wrist'],
}

# Deliberately unused — the ZED's right eye. Named so a reader can see the
# omission is a decision rather than an oversight.
CAMERA_IGNORED = ['cam_head_right', 'cam_right_head']
