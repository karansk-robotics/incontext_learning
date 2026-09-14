// pybind11 bindings for cyclo_control's kinematics + bimanual Cartesian QP.
//
// WHY THIS EXISTS
// ---------------
// aiworker_icrt/kinematics.py carries a damped-least-squares IK solver written
// for dataset conversion. It is not what runs on the robot: the AI Worker is
// driven by cyclo_control, a velocity-level QP that enforces joint position and
// VELOCITY limits, singularity avoidance and self-collision as hard constraints.
//
// Validating a checkpoint against the DLS solver therefore measures a proxy for
// the consumer rather than the consumer -- the failure mode this project has hit
// repeatedly (validations.md section 5). Concretely, the DLS solver reports
// "joint velocity 2.4x over the cap" for poses that cyclo_control could never
// execute that way: its QP would clamp to the bound and lag instead. Same input,
// different and much more benign failure, and only one of the two is what the
// hardware would actually do.
//
// These bindings expose the real thing so validate_checkpoint.py can ask the
// deployment solver directly.
//
// USAGE (mirrors the C++ control loop -- solveQP() builds cost and constraints
// internally, so nothing else needs calling between steps):
//
//     import cyclo_py as cy
//     kin  = cy.KinematicsSolver(urdf_path, srdf_path)
//     ctrl = cy.AIWorkerBimanualMoveLController(kin, dt=1/15)
//     ctrl.set_controller_params(slack_penalty=1e3, cbf_alpha=1.0,
//                                buffer_distance=0.05, safe_distance=0.02)
//     ctrl.set_weight({"arm_l_link7": w6, "arm_r_link7": w6}, w_damping)
//     while ...:
//         kin.update_state(q, qdot)
//         ctrl.set_desired_task_vel({"arm_l_link7": xdot_l, "arm_r_link7": xdot_r})
//         ok, qdot_opt = ctrl.get_opt_joint_vel()
//         q = q + qdot_opt * dt
//
// `ok == False` means the QP was infeasible and qdot_opt is zero -- that is the
// signal validate_checkpoint.py should gate on, not a DLS residual.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <memory>
#include <string>
#include <map>
#include <vector>

#include "common/type_define.hpp"
#include "kinematics/kinematics_solver.hpp"
#include "controllers/ai_worker/ai_worker_bimanual_movel_controller.hpp"

namespace py = pybind11;

using cyclo_motion_controller::kinematics::KinematicsSolver;
using cyclo_motion_controller::controllers::AIWorkerBimanualMoveLController;
using cyclo_motion_controller::common::Vector6d;
using cyclo_motion_controller::common::collision_checker::MinDistResult;

PYBIND11_MODULE(cyclo_py, m)
{
  m.doc() = "cyclo_control kinematics and bimanual Cartesian QP, exposed to Python";

  py::class_<MinDistResult>(m, "MinDistResult")
    .def_readonly("distance", &MinDistResult::distance,
                  "Minimum distance between the pair [m]")
    .def_readonly("grad", &MinDistResult::grad,
                  "Gradient of the distance w.r.t. joint positions")
    .def_readonly("grad_dot", &MinDistResult::grad_dot,
                  "Time derivative of the gradient")
    .def("__repr__", [](const MinDistResult & r) {
      return "<MinDistResult distance=" + std::to_string(r.distance) + ">";
    });

  // Held by shared_ptr because the controller takes a shared_ptr to it; this
  // also keeps the solver alive for as long as any controller references it.
  py::class_<KinematicsSolver, std::shared_ptr<KinematicsSolver>>(m, "KinematicsSolver")
    .def(py::init<const std::string &, const std::string &>(),
         py::arg("urdf_path"), py::arg("srdf_path"),
         "The SRDF supplies the collision pairs; without it self-collision "
         "constraints are inert.")

    .def("update_state", &KinematicsSolver::updateState,
         py::arg("q"), py::arg("qdot"),
         "Must be called before each get_opt_joint_vel().")

    // Affine3d is returned to Python as a plain 4x4 homogeneous matrix.
    .def("compute_pose",
         [](KinematicsSolver & s, const Eigen::VectorXd & q, const std::string & link) {
           return Eigen::Matrix4d(s.computePose(q, link).matrix());
         }, py::arg("q"), py::arg("link_name"))
    .def("get_pose",
         [](const KinematicsSolver & s, const std::string & link) {
           return Eigen::Matrix4d(s.getPose(link).matrix());
         }, py::arg("link_name"),
         "Pose from the last update_state(); call compute_pose() for an "
         "arbitrary q without disturbing solver state.")

    .def("compute_jacobian", &KinematicsSolver::computeJacobian,
         py::arg("q"), py::arg("link_name"))
    .def("get_jacobian", &KinematicsSolver::getJacobian, py::arg("link_name"))

    .def("get_urdf_path", &KinematicsSolver::getURDFPath)
    .def("get_link_frames", &KinematicsSolver::getLinkFrameVector)
    .def("has_link_frame", &KinematicsSolver::hasLinkFrame, py::arg("name"))
    .def("get_joint_frames", &KinematicsSolver::getJointFrameVector)
    .def("has_joint_frame", &KinematicsSolver::hasJointFrame, py::arg("name"))
    .def("get_joint_names", &KinematicsSolver::getJointNames,
         "Model joint order -- this is what q and qdot must be indexed by, and "
         "it is NOT the dataset's JOINT_ORDER. Map explicitly.")
    .def("get_root_link_name", &KinematicsSolver::getRootLinkName)
    .def("get_dof", &KinematicsSolver::getDof)
    .def("get_joint_position", &KinematicsSolver::getJointPosition)
    .def("get_joint_velocity", &KinematicsSolver::getJointVelocity)
    .def("get_joint_position_limit", &KinematicsSolver::getJointPositionLimit,
         "(lower, upper)")
    .def("get_joint_velocity_limit", &KinematicsSolver::getJointVelocityLimit,
         "(lower, upper) -- these are QP constraints, so a solution can never "
         "exceed them. A DLS solver has no such guarantee.")
    .def("set_joint_velocity_bounds_by_index",
         &KinematicsSolver::setJointVelocityBoundsByIndex,
         py::arg("idx"), py::arg("lower"), py::arg("upper"))

    .def("get_collision_pair_count", &KinematicsSolver::getCollisionPairCount)
    .def("get_collision_pair_distances", &KinematicsSolver::getCollisionPairDistances,
         py::arg("with_grad") = false, py::arg("with_graddot") = false,
         py::arg("verbose") = false,
         "Real per-pair self-collision distances. validate_checkpoint.py "
         "currently approximates this with end-effector separation, which "
         "cannot see an elbow crossing the torso.");

  py::class_<AIWorkerBimanualMoveLController>(m, "AIWorkerBimanualMoveLController")
    .def(py::init<std::shared_ptr<KinematicsSolver>, double>(),
         py::arg("robot_data"), py::arg("dt"))

    .def("set_weight", &AIWorkerBimanualMoveLController::setWeight,
         py::arg("link_w_tracking"), py::arg("w_damping"),
         "link_w_tracking maps link name -> 6-vector of task-space tracking "
         "weights; w_damping penalises joint velocity.")

    .def("set_desired_task_vel", &AIWorkerBimanualMoveLController::setDesiredTaskVel,
         py::arg("link_xdot_desired"),
         "Desired task-space velocity per link as a 6-vector. This is a "
         "resolved-rate servo, not a pose solver: convert a pose error to a "
         "velocity yourself (e.g. xdot = K * error) before calling.")

    .def("set_controller_params", &AIWorkerBimanualMoveLController::setControllerParams,
         py::arg("slack_penalty"), py::arg("cbf_alpha"),
         py::arg("buffer_distance"), py::arg("safe_distance"))

    .def("set_constraint_links", &AIWorkerBimanualMoveLController::setConstraintLinks,
         py::arg("right_link"), py::arg("left_link"))

    .def("set_rigid_grasp_pose_constraint",
         [](AIWorkerBimanualMoveLController & c, bool active,
            const Eigen::Matrix4d & right_to_left_in_right) {
           Eigen::Affine3d T;
           T.matrix() = right_to_left_in_right;
           c.setRigidGraspPoseConstraint(active, T);
         },
         py::arg("active"), py::arg("right_to_left_in_right") = Eigen::Matrix4d::Identity(),
         "Equality constraint holding the two hands in a fixed relative pose, "
         "for carrying one object with both arms.")

    // The C++ signature writes into an out-parameter; Python gets a tuple.
    .def("get_opt_joint_vel",
         [](AIWorkerBimanualMoveLController & c) {
           Eigen::VectorXd qdot;
           const bool ok = c.getOptJointVel(qdot);
           return py::make_tuple(ok, qdot);
         },
         "Returns (ok, qdot). ok=False means the QP was infeasible and qdot is "
         "zero -- the honest 'this command cannot be executed' signal.");
}
