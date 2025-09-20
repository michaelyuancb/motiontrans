import zerorpc
from polymetis import RobotInterface
import scipy.spatial.transform as st
import numpy as np
import torch


# robot_description_path: "franka_panda/panda_arm.urdf"
# controlled_joints:  [0, 1, 2, 3, 4, 5, 6]
# num_dofs: 7
# ee_link_idx: 7
# ee_link_name: panda_link8
# rest_pose: [-0.13935425877571106, -0.020481698215007782, -0.05201413854956627, -2.0691256523132324, 0.05058913677930832, 2.0028650760650635, -0.9167874455451965]
# joint_limits_low: [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
# joint_limits_high: [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
# joint_damping: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
# torque_limits: [87., 87., 87., 87., 12., 12., 12.]


class FrankaInterface:
    def __init__(self):
        self.robot = RobotInterface(ip_address='localhost')
        self.joint_limits_high = torch.Tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        self.joint_limits_low = torch.Tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])


    def get_ee_pose(self):
        data = self.robot.get_ee_pose()
        pos = data[0].numpy()
        quat_xyzw = data[1].numpy()
        rot_vec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        return np.concatenate([pos, rot_vec]).tolist()
    
    def get_joint_positions(self):
        return self.robot.get_joint_positions().numpy().tolist()
    
    def get_joint_velocities(self):
        return self.robot.get_joint_velocities().numpy().tolist()
    
    def move_to_joint_positions(self, positions, time_to_go):
        self.robot.move_to_joint_positions(
            positions=torch.Tensor(positions),
            time_to_go=time_to_go
        )
    
    def start_cartesian_impedance(self, Kx, Kxd):
        self.robot.start_cartesian_impedance(
            Kx=torch.Tensor(Kx),
            Kxd=torch.Tensor(Kxd)
        )
    
    def start_joint_impedance(self, Kp, Kd):
        self.robot.start_joint_impedance(
            Kp=torch.Tensor(Kp),
            Kd=torch.Tensor(Kd)
        )

    def update_desired_ee_pose(self, pose):
        pose = np.asarray(pose)
        self.robot.update_desired_ee_pose(
            position=torch.Tensor(pose[:3]),
            orientation=torch.Tensor(st.Rotation.from_rotvec(pose[3:]).as_quat())
        )

    def update_desired_joint_positions(self, position):
        self.robot.update_desired_joint_positions((torch.Tensor(position)))

    def terminate_current_policy(self):
        self.robot.terminate_current_policy()
        
    def go_home(self):
        self.robot.go_home()
    
    def ik_exist(self, position, orientation, q0, tol: float = 0.001):
        position = np.asarray(position)
        orientation = np.asarray(orientation)
        q0 = np.asarray(q0)
        ik_solution = self.robot.solve_inverse_kinematics(position=torch.Tensor(position),
                                                          orientation=torch.Tensor(orientation),
                                                          q0=torch.Tensor(q0),
                                                          tol=tol)
        result_desc = "Success"
        ik_solution, ik_solution_flag = ik_solution
        if ik_solution_flag is True:
            return ik_solution_flag.item(), result_desc, None
        
        low_flag = ik_solution < self.joint_limits_low
        high_flag = ik_solution > self.joint_limits_high
        out_range_flag = low_flag.any() or high_flag.any()
        if out_range_flag:
            result_desc = "Joint Position Out-of-Range"
        elif ik_solution_flag == 0:
            result_desc = "No IK Solution Found"
        final_ik_flag = (not out_range_flag)
        try: 
            ik_solution = ik_solution.numpy().tolist()
        except:
            ik_solution = None
        return final_ik_flag, result_desc, ik_solution




s = zerorpc.Server(FrankaInterface())
s.bind("tcp://0.0.0.0:4242")
s.run()