"""ROS 2 node running the dual-arm Cartesian ICRT policy on the AI Worker.

Wiring
------
Subscribes to the three camera topics and ``/joint_states``, and publishes
``JointTrajectory`` on the *leader* command topics::

    /leader/joint_trajectory_command_broadcaster_left/joint_trajectory
    /leader/joint_trajectory_command_broadcaster_right/joint_trajectory

That choice is deliberate. ``ffw_sg2_follower_ai.launch.py`` spawns the arm
controllers with those topics remapped onto ``arm_l_controller``/
``arm_r_controller``, so publishing there makes the policy a drop-in replacement
for the physical leader -- the robot can be brought up exactly as it is for
teleoperation, with nothing relaunched and no controller reconfigured.

Transport is whatever RMW is configured; under ``rmw_zenoh_cpp`` a single
``rmw_zenohd`` router must be running on the network.

Rate
----
Inference runs on a timer rather than in the subscription callback. ICRT advances
its KV cache by two tokens per step, so the cadence has to be regular: driving it
from camera callbacks would let a backlog stretch the effective control period
and desynchronise the cache from wall-clock time.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import message_filters

from aiworker_icrt import constants as C

DEFAULT_CAMERA_TOPICS = {
    'cam_head': '/zed/zed_node/left/image_rect_color/compressed',
    'cam_wrist_left': '/camera_left/camera_left/color/image_rect_raw/compressed',
    'cam_wrist_right': '/camera_right/camera_right/color/image_rect_raw/compressed',
}
LEFT_CMD_TOPIC = '/leader/joint_trajectory_command_broadcaster_left/joint_trajectory'
RIGHT_CMD_TOPIC = '/leader/joint_trajectory_command_broadcaster_right/joint_trajectory'


class ICRTPolicyNode(Node):

    def __init__(self):
        super().__init__('icrt_policy')

        p = self.declare_parameter
        p('train_yaml', '')
        p('checkpoint', '')
        p('urdf', '')
        p('prompt_npz', '')          # converted demo used to fill the KV cache
        p('inference_rate', 15.0)    # Hz
        p('device', 'cuda')
        p('binary_gripper', False)
        p('use_temporal', True)
        p('sync_slop', 0.05)         # s, ApproximateTimeSynchronizer tolerance
        p('max_joint_step', 0.25)    # rad per control step, safety clamp
        p('autostart', False)
        for key, topic in DEFAULT_CAMERA_TOPICS.items():
            p(f'topic.{key}', topic)
        p('topic.joint_states', '/joint_states')

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self._rate = float(g('inference_rate'))
        self._max_step = float(g('max_joint_step'))
        self._binary_gripper = bool(g('binary_gripper'))
        self._use_temporal = bool(g('use_temporal'))

        self._lock = threading.Lock()
        self._latest: Optional[tuple] = None
        self._running = bool(g('autostart'))
        self._warned_missing = False

        # --- policy -------------------------------------------------------- #
        from aiworker_icrt.policy import DualArmICRT
        self.get_logger().info('loading ICRT checkpoint (this builds the '
                               'transformer on GPU and takes a moment)...')
        self.policy = DualArmICRT(
            train_yaml_path=g('train_yaml'),
            checkpoint_path=g('checkpoint'),
            urdf_path=g('urdf') or None,
            device=g('device'),
        )
        self.get_logger().info('checkpoint loaded')

        self._prompt_npz = g('prompt_npz')
        if self._prompt_npz:
            self._load_prompt(self._prompt_npz)

        # --- I/O ------------------------------------------------------------ #
        # Images are sensor traffic: best-effort, keep-last-1. joint_states comes
        # from joint_state_broadcaster with default reliable QoS.
        img_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST)
        js_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                            history=HistoryPolicy.KEEP_LAST)

        self._cam_keys = list(C.CAMERA_KEYS)
        subs = [message_filters.Subscriber(self, CompressedImage, g(f'topic.{k}'),
                                           qos_profile=img_qos)
                for k in self._cam_keys]
        subs.append(message_filters.Subscriber(self, JointState, g('topic.joint_states'),
                                               qos_profile=js_qos))
        self._sync = message_filters.ApproximateTimeSynchronizer(
            subs, queue_size=5, slop=float(g('sync_slop')))
        self._sync.registerCallback(self._on_obs)

        self._pub_l = self.create_publisher(JointTrajectory, LEFT_CMD_TOPIC, 10)
        self._pub_r = self.create_publisher(JointTrajectory, RIGHT_CMD_TOPIC, 10)

        self.create_service(Trigger, '~/start', self._srv_start)
        self.create_service(Trigger, '~/stop', self._srv_stop)
        self.create_service(Trigger, '~/reset', self._srv_reset)

        self.create_timer(1.0 / self._rate, self._on_tick)
        self.get_logger().info(
            f'icrt_policy ready at {self._rate} Hz '
            f'({"running" if self._running else "idle - call ~/start"})')

    # -- prompting ---------------------------------------------------------- #

    def _load_prompt(self, path: str) -> None:
        """Load a converted demonstration and fill the KV cache.

        Expects the npz written by the dataset converter: ``images`` (T,N,H,W,3
        uint8), ``joint_states`` (T,22), ``actions_cartesian`` (T,20).
        """
        d = np.load(path)
        images = [
            {k: d['images'][t, i] for i, k in enumerate(C.CAMERA_KEYS)}
            for t in range(d['images'].shape[0])
        ]
        self.policy.prompt(images, d['joint_states'], d['actions_cartesian'])
        self.get_logger().info(
            f'prompted with {len(images)} frames from {path}')

    # -- callbacks ---------------------------------------------------------- #

    def _on_obs(self, *msgs) -> None:
        *img_msgs, js = msgs
        with self._lock:
            self._latest = (img_msgs, js)

    def _on_tick(self) -> None:
        if not self._running:
            return
        with self._lock:
            latest = self._latest
            self._latest = None
        if latest is None:
            return
        img_msgs, js = latest

        joints = self._joint_state_to_vector(js)
        if joints is None:
            return
        images = {k: self._decode(m) for k, m in zip(self._cam_keys, img_msgs)}

        try:
            target, _cart, info = self.policy.step(
                images, joints,
                binary_gripper=self._binary_gripper,
                use_temporal=self._use_temporal,
            )
        except Exception as exc:  # a policy fault must not take the node down
            self.get_logger().error(f'inference failed: {exc}')
            return

        for side in ('left', 'right'):
            if not info[side]['converged']:
                self.get_logger().warn(
                    f'{side} IK did not converge: '
                    f'pos_err={info[side]["pos_err"]*1000:.2f}mm '
                    f'rot_err={info[side]["rot_err"]*1000:.2f}mrad'
                )

        target = self._clamp_step(joints, target)
        self._publish(target)

    # -- helpers ------------------------------------------------------------ #

    def _joint_state_to_vector(self, js: JointState) -> Optional[np.ndarray]:
        """Map a JointState onto the canonical 22-D layout by name.

        Only the 19 upper-body entries exist in /joint_states; the three mobile
        dims come from /odom, which this action space does not use, so they stay
        zero and are never commanded.
        """
        pos = dict(zip(js.name, js.position))
        out = np.zeros(C.JOINT_DIM)
        missing = []
        for i, name in enumerate(C.JOINT_ORDER):
            if name in C.BASE_DIMS:
                continue
            if name not in pos:
                missing.append(name)
            else:
                out[i] = pos[name]
        if missing:
            if not self._warned_missing:
                self.get_logger().error(
                    f'/joint_states is missing {missing}; is the follower '
                    f'bringup running?')
                self._warned_missing = True
            return None
        self._warned_missing = False
        return out

    @staticmethod
    def _decode(msg: CompressedImage) -> np.ndarray:
        import cv2
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)   # BGR
        if img is None:
            raise ValueError('failed to decode CompressedImage')
        return img[:, :, ::-1]                      # -> RGB

    def _clamp_step(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Bound per-step joint motion.

        IK is warm-started and normally moves ~0.03 rad/step, so this never binds
        in healthy operation. It exists for the case where the policy emits a
        wild pose -- near a singularity, or off-distribution -- and IK dutifully
        solves for it. Better a slewed approach than a full-speed lunge.
        """
        out = target.copy()
        for sl in (C.SLICE_ARM_L, C.SLICE_ARM_R):
            delta = target[sl] - current[sl]
            worst = np.abs(delta).max()
            if worst > self._max_step:
                self.get_logger().warn(
                    f'clamping joint step {worst:.3f} -> {self._max_step:.3f} rad')
                out[sl] = current[sl] + delta * (self._max_step / worst)
        return out

    def _publish(self, target: np.ndarray) -> None:
        dt = 1.0 / self._rate
        stamp = Duration(sec=int(dt), nanosec=int((dt % 1.0) * 1e9))
        for pub, joints, arm_sl, grip_i in (
            (self._pub_l, C.ARM_L_JOINTS + [C.GRIPPER_L_JOINT],
             C.SLICE_ARM_L, C.SLICE_GRIP_L),
            (self._pub_r, C.ARM_R_JOINTS + [C.GRIPPER_R_JOINT],
             C.SLICE_ARM_R, C.SLICE_GRIP_R),
        ):
            msg = JointTrajectory()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.joint_names = joints
            positions = np.concatenate([target[arm_sl], [target[grip_i]]])
            # One point, one control period out: the 100 Hz JointTrajectoryController
            # interpolates between here and there, so a 15 Hz policy still yields
            # smooth motion on the DYNAMIXEL bus.
            msg.points = [JointTrajectoryPoint(
                positions=positions.tolist(), time_from_start=stamp)]
            pub.publish(msg)

    # -- services ----------------------------------------------------------- #

    def _srv_start(self, _req, res):
        self._running = True
        res.success, res.message = True, 'policy running'
        return res

    def _srv_stop(self, _req, res):
        self._running = False
        res.success, res.message = True, 'policy stopped'
        return res

    def _srv_reset(self, _req, res):
        self._running = False
        if self._prompt_npz:
            self._load_prompt(self._prompt_npz)
        else:
            self.policy.reset()
        res.success, res.message = True, 'kv cache reset; call ~/start to resume'
        return res


def main(args=None):
    rclpy.init(args=args)
    node = ICRTPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
