"""Dual-arm Cartesian ICRT policy for the AI Worker.

Replaces ``icrt.models.policy.icrt_wrapper.ICRTWrapper``, which hardcodes the
single-arm DROID setup: two cameras named side/wrist, a 10-D proprio vector, and
euler->rot_6d conversion on one end-effector. Here we have three cameras, two
arms, and joint-space hardware that has to be retargeted through
:mod:`aiworker_icrt.kinematics` on the way in and out.

The model still sees exactly what ICRT expects -- a sequence of (state, action)
pairs in a Cartesian end-effector space -- so DROID pretraining stays meaningful
for each arm individually.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import yaml

from aiworker_icrt import constants as C
from aiworker_icrt.kinematics import FFWKinematics


class DualArmICRT:
    """Loads a trained ICRT checkpoint and steps it from raw robot observations.

    Parameters
    ----------
    train_yaml_path
        The ``ExperimentConfig`` dumped alongside the checkpoint during training.
    checkpoint_path
        Trained weights.
    urdf_path
        FFW-SG2 URDF used for the FK/IK retargeting.
    device
        Torch device. ICRT constructs its transformer under a CUDA default tensor
        type, so this is effectively cuda-only.
    """

    def __init__(
        self,
        train_yaml_path: Union[str, Path],
        checkpoint_path: Union[str, Path],
        urdf_path: Optional[str] = None,
        device: str = 'cuda',
        action_exec_horizon: Optional[int] = None,
    ):
        # Imported lazily: these pull in the whole ICRT/timm stack, which we do
        # not want at module import time (the converter imports this package too).
        import timm
        from torchvision import transforms

        import icrt.util.misc as misc
        from icrt.util.args import ExperimentConfig
        from icrt.util.model_constructor import model_constructor

        args: ExperimentConfig = yaml.load(
            Path(train_yaml_path).read_text(), Loader=yaml.Loader
        )
        self.args = args
        self.device = torch.device(device)

        num_arms = getattr(args.shared_cfg, 'num_arms', 1)
        if num_arms != 2:
            raise ValueError(
                f'{train_yaml_path} was trained with num_arms={num_arms}; '
                'DualArmICRT requires a bimanual checkpoint (num_arms=2)'
            )
        if args.shared_cfg.num_cameras != C.NUM_CAMERAS:
            raise ValueError(
                f'checkpoint expects {args.shared_cfg.num_cameras} cameras, '
                f'but this robot provides {C.NUM_CAMERAS}: {C.CAMERA_KEYS}'
            )
        if not args.shared_cfg.rot_6d:
            raise ValueError('rot_6d=False is not supported by the Cartesian bridge')

        model = model_constructor(
            model_config=args.model_cfg,
            shared_config=args.shared_cfg,
            train=False,
        )

        # Match the vision preprocessing to whatever encoder the checkpoint used;
        # a mismatch here is silent and ruins the policy.
        timm_cfg = timm.data.resolve_data_config(model.vision_encoder.model.pretrained_cfg)
        full = timm.data.create_transform(**timm_cfg)
        self.mean, self.std = timm_cfg['mean'], timm_cfg['std']
        # Drop ToTensor/ColorJitter: frames arrive as CHW float tensors already,
        # and jitter is a training-time augmentation.
        self.preprocess = transforms.Compose([
            t for t in full.transforms
            if not isinstance(t, (transforms.ToTensor, transforms.ColorJitter))
        ])

        model.to(self.device)
        misc.load_model(model, str(checkpoint_path))
        model.eval()
        self.model = model

        self.kin = FFWKinematics(urdf_path) if urdf_path else FFWKinematics()
        self.num_pred_steps = args.shared_cfg.num_pred_steps
        self.action_exec_horizon = action_exec_horizon or self.num_pred_steps

        # Joint targets for the dims this action space does not control. Captured
        # at prompt/reset time and held for the episode.
        self._held_joints: Optional[np.ndarray] = None
        self._last_joints: Optional[np.ndarray] = None
        # Torso movement beyond this is reported: the training data has the lift
        # constant, so anything else is a regime the policy has not seen.
        self.lift_drift_warn_m = 0.01
        self._lift_drift = 0.0
        self.reset()

    # -- lifecycle ---------------------------------------------------------- #

    def reset(self, joint_state: Optional[np.ndarray] = None) -> None:
        """Clear the KV cache and latch the uncontrolled joints."""
        self.model.reset(self.action_exec_horizon)
        if joint_state is not None:
            self._held_joints = np.asarray(joint_state, dtype=np.float64).copy()
            self._last_joints = self._held_joints.copy()

    # -- observation plumbing ----------------------------------------------- #

    def _stack_images(self, images: Dict[str, np.ndarray]) -> torch.Tensor:
        """{cam_key: HWC uint8 or CHW float} -> (1, T=1, N, 3, H, W).

        Camera order follows constants.CAMERA_KEYS and must match the order used
        when the training data was converted -- ICRT gives each camera its own
        attention-pooling adapter, so they are not interchangeable.
        """
        frames = []
        for key in C.CAMERA_KEYS:
            if key not in images:
                raise KeyError(f'missing camera {key!r}; have {sorted(images)}')
            img = images[key]
            t = torch.as_tensor(np.ascontiguousarray(img))
            if t.ndim != 3:
                raise ValueError(f'{key}: expected a single HWC/CHW frame, got {tuple(t.shape)}')
            if t.shape[-1] in (1, 3):       # HWC -> CHW
                t = t.permute(2, 0, 1)
            if t.dtype == torch.uint8:
                t = t.float() / 255.0
            frames.append(self.preprocess(t))
        return torch.stack(frames, dim=0)[None, None].float()

    def _proprio(self, joint_state: np.ndarray) -> torch.Tensor:
        """(22,) joint positions -> (1, 1, 23) proprio tensor.

        Arms in Cartesian, then head(2) and lift(1) appended raw — the two things
        the policy must be aware of but, for the head, cannot change.
        """
        pro = self.kin.joints_to_proprio(np.asarray(joint_state, dtype=np.float64))
        return torch.as_tensor(pro, dtype=torch.float32)[None, None]

    def _to_device(self, obs: dict) -> dict:
        return {k: (v.to(self.device, non_blocking=True) if v is not None else None)
                for k, v in obs.items()}

    # -- prompting ---------------------------------------------------------- #

    @torch.inference_mode()
    def prompt(
        self,
        images: Sequence[Dict[str, np.ndarray]],
        joint_states: np.ndarray,
        actions_cartesian: np.ndarray,
    ) -> None:
        """Fill the KV cache with one or more demonstrations.

        This is the in-context part: the demos are not trained on, they are read.
        ``actions_cartesian`` is (T, 20) -- already retargeted, since a demo comes
        from a converted dataset rather than from live hardware.
        """
        joint_states = np.asarray(joint_states, dtype=np.float64)
        T = len(images)
        if joint_states.shape[0] != T:
            raise ValueError(f'{T} frames but {joint_states.shape[0]} joint states')

        frames = torch.cat([self._stack_images(im) for im in images], dim=1)  # 1,T,N,3,H,W
        proprio = torch.as_tensor(
            self.kin.joints_to_proprio(joint_states), dtype=torch.float32
        )[None]
        action = torch.as_tensor(np.asarray(actions_cartesian), dtype=torch.float32)[None]
        # ICRT appends a zero eos channel to actions it is prompted with.
        action = torch.cat([action, torch.zeros(1, action.shape[1], 1)], dim=-1)

        self.model.reset(self.action_exec_horizon)
        self.model.prompt(self._to_device({
            'observation': frames, 'proprio': proprio, 'action': action,
        }))
        self._held_joints = joint_states[-1].copy()
        self._last_joints = joint_states[-1].copy()

    # -- rollout ------------------------------------------------------------ #

    @torch.inference_mode()
    def step(
        self,
        images: Dict[str, np.ndarray],
        joint_state: np.ndarray,
        binary_gripper: bool = False,
        use_temporal: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """One control step.

        Returns ``(joint_target_22, cartesian_action_20, ik_info)``. The joint
        target's head/lift/base entries are whatever was latched at reset -- this
        action space does not command them.
        """
        joint_state = np.asarray(joint_state, dtype=np.float64)
        if self._held_joints is None:
            self.reset(joint_state)

        obs = self._to_device({
            'observation': self._stack_images(images),
            'proprio': self._proprio(joint_state),
            'action': None,
        })
        action = self.model.get_action_eval(
            obs,
            binary_gripper=binary_gripper,
            use_temporal=use_temporal,
        )
        cart = action.detach().float().cpu().numpy()

        # Seed IK from the previous solution when we have one: warm-starting is
        # what keeps IK at ~1 iteration and stops the elbow flipping between
        # neighbouring timesteps.
        seed = self._last_joints if self._last_joints is not None else joint_state
        seed = seed.copy()

        # The lift MUST come from the live measurement, not from the value
        # latched at reset.
        #
        # base_link-referenced EEF poses depend on the torso height, so FK (which
        # reads the measured lift) and IK (which solves against it) have to agree
        # about where the shoulders are. Seeding IK with a stale lift makes them
        # disagree silently: the arm is then solved for a torso that is not where
        # the robot's torso actually is, and the hand lands off by roughly the
        # lift error -- with every convergence check still reporting success,
        # because the solver hit the pose it was asked for in the wrong frame.
        #
        # Using the live value means a torso that moves is COMPENSATED FOR: the
        # arm adjusts and the hand stays where the policy asked. We still never
        # command the lift; ros_node only publishes arm and gripper joints.
        seed[C.SLICE_LIFT] = joint_state[C.SLICE_LIFT]
        seed[C.SLICE_HEAD] = joint_state[C.SLICE_HEAD]
        seed[C.SLICE_BASE] = joint_state[C.SLICE_BASE]

        # Compensating is correct but the situation is still off-distribution:
        # the policy was trained on recordings where the torso never moved, so a
        # moving lift is a regime it has never seen. Say so rather than silently
        # tracking it.
        lift_drift = abs(float(joint_state[C.SLICE_LIFT])
                         - float(self._held_joints[C.SLICE_LIFT]))
        if lift_drift > self.lift_drift_warn_m:
            self._lift_drift = lift_drift

        # action_to_joints applies the COMMANDED lift before solving the arms:
        # solving against the old torso height and then moving it would displace
        # the hands by exactly the lift delta.
        joint_target, info = self.kin.action_to_joints(cart, seed)
        info['lift_drift_m'] = lift_drift
        self._last_joints = joint_target.copy()
        return joint_target, cart, info
