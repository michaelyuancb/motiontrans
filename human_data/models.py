import numpy as np 
from scipy.spatial.transform import Rotation as R


class Transform:
    def __init__(self, pose_info_str_list=None):
        if type(pose_info_str_list) is str:
            pose_info_str_list = pose_info_str_list.split(",")
        if pose_info_str_list is None:
            self.pos = np.array([0., 0., 0.])
            self.orn = np.array([0., 0., 0., 1.]) 
        else:
            self.pos = np.array([float(data) for data in pose_info_str_list[:3]])
            self.orn = np.array([float(data) for data in pose_info_str_list[3:]])
    
    def return_pose(self):
        return np.concatenate([self.pos, self.orn])

    def return_6d_pose(self):
        return np.concatenate([self.pos, R.from_quat(self.orn).as_euler("xyz")])
    
    def return_matrix(self):
        mat = np.eye(4)
        mat[:3, :3] = R.from_quat(self.orn).as_matrix()
        mat[:3, 3] = self.pos
        return mat
    
    def set_pose(self, pose):
        assert len(pose) == 7
        self.pos = pose[:3]
        self.orn = pose[3:]
    
    def set_6d_pose(self, pose):
        assert len(pose) == 6
        self.pos = pose[:3]
        self.orn = R.from_euler("xyz", pose[3:]).as_quat()

    def set_pose_mat(self, mat):
        assert mat.shape == (4, 4)
        self.pos = mat[:3, 3]
        self.orn = R.from_matrix(mat[:3, :3]).as_quat()

    def mul_mat_left(self, mat):
        mat_self = self.return_matrix()
        mat = mat @ mat_self
        self.set_pose_mat(mat)

    def mul_mat_right(self, mat):
        mat_self = self.return_matrix()
        mat = mat_self @ mat
        self.set_pose_mat(mat)

    def get_pose_str(self):
        info = str(self.pos[0]) + "," + str(self.pos[1]) + "," + str(self.pos[2]) + "," + str(self.orn[0]) + "," + str(self.orn[1]) + "," + str(self.orn[2]) + "," + str(self.orn[3])
        return info
        

# Hand Structure: https://docs.unity3d.com/Packages/com.unity.xr.hands@1.1/manual/hand-data/xr-hand-data-model.html      

class XRHand:
    def __init__(self, hand_pose_str_list=None):
        if hand_pose_str_list is not None:
            self.status = hand_pose_str_list[:1]
            # H: High Confidence
            # L: Low Confidence
            # F: Failed
            info = hand_pose_str_list[1:].split(",") if isinstance(hand_pose_str_list[1:], str) else hand_pose_str_list[1:]
            self.palm  = Transform(info[:7])
            self.wrist = Transform(info[7:14])
            # metacarpal, proximal, distal, tip
            self.thumb1, self.thumb2, self.thumb3, self.thumb_tip = Transform(info[14:21]), Transform(info[21:28]), Transform(info[28:35]), Transform(info[35:42])
            # metacarpal, proximal, intermediate, distal, tip
            self.index0, self.index1, self.index2, self.index3, self.index_tip = Transform(info[42:49]), Transform(info[49:56]), Transform(info[56:63]), Transform(info[63:70]), Transform(info[70:77])
            self.middle0, self.middle1, self.middle2, self.middle3, self.middle_tip = Transform(info[77:84]), Transform(info[84:91]), Transform(info[91:98]), Transform(info[98:105]), Transform(info[105:112])
            self.ring0, self.ring1, self.ring2, self.ring3, self.ring_tip = Transform(info[112:119]), Transform(info[119:126]), Transform(info[126:133]), Transform(info[133:140]), Transform(info[140:147])
            self.pinky0, self.pinky1, self.pinky2, self.pinky3, self.pinky_tip = Transform(info[147:154]), Transform(info[154:161]), Transform(info[161:168]), Transform(info[168:175]), Transform(info[175:182])
        else:
            self.palm, self.wrist = Transform(), Transform()
            self.thumb1, self.thumb2, self.thumb3, self.thumb_tip = Transform(), Transform(), Transform(), Transform()
            self.index0, self.index1, self.index2, self.index3, self.index_tip = Transform(), Transform(), Transform(), Transform(), Transform()
            self.middle0, self.middle1, self.middle2, self.middle3, self.middle_tip = Transform(), Transform(), Transform(), Transform(), Transform()
            self.ring0, self.ring1, self.ring2, self.ring3, self.ring_tip = Transform(), Transform(), Transform(), Transform(), Transform()
            self.pinky0, self.pinky1, self.pinky2, self.pinky3, self.pinky_tip = Transform(), Transform(), Transform(), Transform(), Transform()
        self.hand_pose = [self.palm, self.wrist, self.thumb1, self.thumb2, self.thumb3, self.thumb_tip, self.index0, self.index1, self.index2, self.index3, self.index_tip, self.middle0, self.middle1, self.middle2, self.middle3, self.middle_tip, self.ring0, self.ring1, self.ring2, self.ring3, self.ring_tip, self.pinky0, self.pinky1, self.pinky2, self.pinky3, self.pinky_tip]
        self.hand_pose_str_list = hand_pose_str_list

    def return_pose(self):
        return self.hand_pose

    def mul_mat_left(self, mat):
        for pose in self.hand_pose:
            pose.mul_mat_left(mat)

    def mul_mat_right(self, mat):
        for pose in self.hand_pose:
            pose.mul_mat_right(mat)

    def get_hand_pose_str(self):
        info = []
        for pose in self.hand_pose:
            info.append(pose.get_pose_str())
        info = ",".join(info)
        info = self.status + info
        return info
    
    # def get_hand_pose_array(self):
    #     arr = []
    #     for pose in self.hand_pose:
    #         arr.append(pose.return_pose())
    #     arr = np.concatenate(arr)
    #     return arr
    
    # def set_hand_pose_array(self, arr):
    #     for i, pose in enumerate(self.hand_pose):
    #         pose.set_pose(arr[i*6:i*6+7])

    def get_hand_6d_pose_array(self):
        arr = []
        for pose in self.hand_pose:
            arr.append(pose.return_6d_pose())
        arr = np.concatenate(arr)
        return arr
    
    def set_hand_6d_pose_array(self, arr):
        for i, pose in enumerate(self.hand_pose):
            pose.set_6d_pose(arr[i*6:i*6+6])