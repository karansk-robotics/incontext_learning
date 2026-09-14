"""Bring up the ICRT dual-arm policy against the FFW-SG2 follower.

    ros2 launch /workspace/launch/icrt_policy.launch.py \
        checkpoint:=/workspace/checkpoints/icrt_dualarm.pth \
        train_yaml:=/workspace/config/train_dualarm.yaml \
        prompt_npz:=/workspace/prompts/pick_place.npz

The policy node publishes to the LEADER topics:
    /leader/joint_trajectory_command_broadcaster_{left,right}/joint_trajectory
ffw_sg2_follower_ai.launch.py remaps those onto arm_{l,r}_controller at spawn
time, so the policy is a drop-in for the physical leader arm — the robot comes
up exactly as it does for teleop and nothing needs relaunching.

Start ffw_sg2_follower_ai.launch.py, NOT ffw_sg2_ai.launch.py: the latter also
starts the LG2 leader, which would fight the policy for the same topics. The
file lives at ffw_bringup/launch/ffw_sg2_follower_ai.launch.py in
ROBOTIS-GIT/ai_worker (confirmed against branch main).

init_position:=false skips joint homing and the swerve-steering controller swap
on a warm restart, which otherwise costs ~20 s before the cameras come up.

The node idles until ~/start is called. ICRT's KV cache has to be prompted with
a demonstration before the first action means anything, so autostart defaults to
false and should stay that way:

    ros2 service call /icrt_policy/start  std_srvs/srv/Trigger
    ros2 service call /icrt_policy/stop   std_srvs/srv/Trigger
    ros2 service call /icrt_policy/reset  std_srvs/srv/Trigger   # re-prompt
"""
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, LogInfo)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

# Node parameters forwarded as `-p name:=<value>`; each is also a launch arg.
_PARAMS = [
    # name               default                          description
    ('train_yaml',       '',                              'ICRT training config the checkpoint was produced with'),
    ('checkpoint',       '',                              'ICRT .pth checkpoint'),
    ('urdf',             '',                              'FFW-SG2 follower URDF (blank -> assets/ffw_sg2)'),
    ('prompt_npz',       '',                              'converted demo used to fill the KV cache'),
    ('inference_rate',   '15.0',                          'Hz; ICRT advances 2 KV tokens per step'),
    ('device',           'cuda',                          'torch device'),
    ('binary_gripper',   'false',                         'threshold the gripper channel'),
    ('use_temporal',     'true',                          'temporal ensembling of actions'),
    ('sync_slop',        '0.05',                          'ApproximateTimeSynchronizer tolerance, s'),
    ('max_joint_step',   '0.25',                          'safety clamp, rad per control step'),
    ('autostart',        'false',                         'LEAVE FALSE — needs a prompt first'),
]


def generate_launch_description():
    args = [
        DeclareLaunchArgument(name, default_value=default, description=desc)
        for name, default, desc in _PARAMS
    ]

    args += [
        DeclareLaunchArgument(
            'bringup', default_value='true',
            description='also start the FFW-SG2 follower bringup'),
        DeclareLaunchArgument(
            'bringup_package', default_value='ffw_bringup',
            description='package holding ffw_sg2_follower_ai.launch.py'),
        DeclareLaunchArgument(
            'bringup_file', default_value='ffw_sg2_follower_ai.launch.py',
            description='follower bringup; NOT ffw_sg2_ai.launch.py, which '
                        'also starts the LG2 leader and fights for our topics'),
        DeclareLaunchArgument(
            'log_level', default_value='info'),

        # --- args belonging to ffw_sg2_follower_ai.launch.py itself ---
        DeclareLaunchArgument(
            'model', default_value='ffw_sg2_rev1_follower',
            description='model dir under ffw_description/urdf'),
        DeclareLaunchArgument(
            'launch_cameras', default_value='true'),
        DeclareLaunchArgument(
            'launch_lidar', default_value='true'),
        DeclareLaunchArgument(
            'init_position', default_value='true',
            description='home every joint then swap the swerve steering '
                        'controller; adds ~20s before cameras come up. Set '
                        'false on a warm restart.'),
    ]

    follower = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare(LaunchConfiguration('bringup_package')),
            'launch',
            LaunchConfiguration('bringup_file'),
        ])),
        condition=IfCondition(LaunchConfiguration('bringup')),
        launch_arguments={
            'model':          LaunchConfiguration('model'),
            'launch_cameras': LaunchConfiguration('launch_cameras'),
            'launch_lidar':   LaunchConfiguration('launch_lidar'),
            'init_position':  LaunchConfiguration('init_position'),
        }.items(),
    )

    # aiworker_icrt is a plain Python package on PYTHONPATH (/workspace), not an
    # ament package, so there is no `ros2 run` executable to point Node() at.
    cmd = ['python3', '-m', 'aiworker_icrt.ros_node', '--ros-args',
           '--log-level', LaunchConfiguration('log_level')]
    for name, _default, _desc in _PARAMS:
        cmd += ['-p', [name, ':=', LaunchConfiguration(name)]]

    policy = ExecuteProcess(
        cmd=cmd,
        name='icrt_policy',
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription(args + [
        LogInfo(msg='icrt_policy idles until /icrt_policy/start is called — '
                    'prompt the KV cache first.'),
        follower,
        policy,
    ])
