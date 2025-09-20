
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d



def plt_plot_pose(transformation, ax):
    """
    绘制位姿
    :param transformation: 位姿数组，前三个为位置，后三个为欧拉角（弧度）
    :param ax: Matplotlib 3D坐标轴
    """
    # 提取位置和欧拉角
    position = transformation[:3]
    roll, pitch, yaw = transformation[3:]
    
    # 使用 scipy.spatial.transform.Rotation 创建旋转矩阵
    rotation = R.from_euler('xyz', [roll, pitch, yaw])
    R_matrix = rotation.as_matrix()
    
    # 基准坐标系原点
    origin = np.array([0, 0, 0])
    
    # 绘制基准坐标系
    ax.quiver(origin[0], origin[1], origin[2], 1, 0, 0, color='r', length=1.0, arrow_length_ratio=0.1, label='X-axis (ref)')
    ax.quiver(origin[0], origin[1], origin[2], 0, 1, 0, color='g', length=1.0, arrow_length_ratio=0.1, label='Y-axis (ref)')
    ax.quiver(origin[0], origin[1], origin[2], 0, 0, 1, color='b', length=1.0, arrow_length_ratio=0.1, label='Z-axis (ref)')
    
    # 绘制位姿坐标系
    ax.quiver(position[0], position[1], position[2], R_matrix[0, 0], R_matrix[1, 0], R_matrix[2, 0], color='r', length=1.0, arrow_length_ratio=0.1, label='X-axis (pose)')
    ax.quiver(position[0], position[1], position[2], R_matrix[0, 1], R_matrix[1, 1], R_matrix[2, 1], color='g', length=1.0, arrow_length_ratio=0.1, label='Y-axis (pose)')
    ax.quiver(position[0], position[1], position[2], R_matrix[0, 2], R_matrix[1, 2], R_matrix[2, 2], color='b', length=1.0, arrow_length_ratio=0.1, label='Z-axis (pose)')
    
    # 绘制位姿原点与基准坐标系原点的连线
    ax.plot([origin[0], position[0]], [origin[1], position[1]], [origin[2], position[2]], 'k--', label='Connection line')
    
    # 设置坐标轴标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Pose Visualization')
    
    # 设置坐标轴范围以放大显示
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    
    ax.legend()


def create_sphere(center, radius=0.05, color=[1, 0, 0]):
    if center.shape[0] == 6:
        center = center[:3]
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color) 
    return sphere


def create_coordinate(pose, size=0.1):
    origin, orientation = pose[:3], pose[3:]
    print(origin, orientation)
    orientation = R.from_euler('xyz', orientation).as_matrix()
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    # coordinate.rotate(orientation, center=-size/2*np.ones(3))
    coordinate.rotate(orientation, center=np.zeros(3))
    coordinate.translate(origin)
    return coordinate
