
import serial
import numpy as np
import time
from real.inspire_hand_utils.inspire_hand_serial_control import read6, write6, openSerial

class InspireHandAgent:
    def __init__(self,
                 port,
                 hand_id,
                 baudrate,
                 speed=1000):
        super().__init__()
        self.serial = openSerial(port, baudrate)
        self.id = hand_id

        # limit angle of hand tip (degree)
        self.finger_limit = [19, 176.7] #[0.3314, 3.0824]
        self.thumb_band_limit = [-13, 53.6] #[0.2238, 0.9350]
        self.thumb_rot_limit = [90, 165] #[1.5708, 2.8783]

        # limit angle of hand joint (Radians)
        self.finger_joint_limit = [0.3377, 1.6467]
        self.thumb_band_joint_limit = [0.0, 0.4712]
        self.thumb_rot_joint_limit = [0.3367, 1.4990]

        if speed < 0 or speed > 1000:
            raise ValueError("Joint speed out of range!")
        self.joint_speed = speed

    def convert_joint_radians_to_servo(self, radian_position: np.ndarray):
        servo_angle = list()
        for i, joint in enumerate(radian_position):
            if i <= 3:
                servo_angle.append(
                    int((radian_position[i] - self.finger_joint_limit[1]) /
                        (self.finger_joint_limit[0] - self.finger_joint_limit[1]) * 1000)
                )
            elif i == 4:
                servo_angle.append(
                    int((radian_position[i] - self.thumb_band_joint_limit[1]) /
                        (self.thumb_band_joint_limit[0] - self.thumb_band_joint_limit[1]) * 1000)
                )
            elif i == 5:
                servo_angle.append(
                    int((radian_position[i] - self.thumb_rot_joint_limit[1]) /
                        (self.thumb_rot_joint_limit[0] - self.thumb_rot_joint_limit[1]) * 1000)
                )
            else:
                raise ValueError("position list should no longer than 6!")

        return servo_angle

    def convert_servo_to_joint_radians(self, servo_position: np.ndarray):
        radians_angle = list()
        for i, joint in enumerate(servo_position):
            if i <= 3:
                radians_angle.append(
                    ((servo_position[i] - 0) *
                     (self.finger_joint_limit[0] - self.finger_joint_limit[1]) / 1000) + self.finger_joint_limit[1]
                )
            elif i == 4:
                radians_angle.append(
                    ((servo_position[i] - 0) *
                     (self.thumb_band_joint_limit[0] - self.thumb_band_joint_limit[1]) / 1000) + self.thumb_band_joint_limit[1]
                )
            elif i == 5:
                radians_angle.append(
                    ((servo_position[i] - 0) *
                     (self.thumb_rot_joint_limit[0] - self.thumb_rot_joint_limit[1]) / 1000) + self.thumb_rot_joint_limit[1]
                )
            else:
                raise ValueError("position list should no longer than 6!")

        return radians_angle

    def convert_degree_to_servo(self, degree_position: np.ndarray):
        servo_angle = list()
        for i, joint in enumerate(degree_position):
            if i <= 3:
                servo_angle.append(
                    int((degree_position[i] - self.finger_limit[0]) /
                        (self.finger_limit[1] - self.finger_limit[0]) * 1000)
                )
            elif i == 4:
                servo_angle.append(
                    int((degree_position[i] - self.thumb_band_limit[0]) /
                        (self.thumb_band_limit[1] - self.thumb_band_limit[0]) * 1000)
                )
            elif i == 5:
                servo_angle.append(
                    int((degree_position[i] - self.thumb_rot_limit[0]) /
                        (self.thumb_rot_limit[1] - self.thumb_rot_limit[0]) * 1000)
                )
            else:
                raise ValueError("position list should no longer than 6!")

        return servo_angle

    def convert_servo_to_degree(self, servo_position: np.ndarray):
        degree_angle = list()
        for i, joint in enumerate(servo_position):
            if i <= 3:
                degree_angle.append(
                    ((servo_position[i] - 0) *
                     (self.finger_limit[1] - self.finger_limit[0]) / 1000) + self.finger_limit[0]
                )
            elif i == 4:
                degree_angle.append(
                    ((servo_position[i] - 0) *
                     (self.thumb_band_limit[1] - self.thumb_band_limit[0]) / 1000) + self.thumb_band_limit[0]
                )
            elif i == 5:
                degree_angle.append(
                    ((servo_position[i] - 0) *
                     (self.thumb_rot_limit[1] - self.thumb_rot_limit[0]) / 1000) + self.thumb_rot_limit[0]
                )
            else:
                raise ValueError("position list should no longer than 6!")

        return degree_angle



    def set_joint_position2(self, radian_position: np.ndarray):
        '''
        using joint angle
        '''
        assert len(radian_position) == 6

        # print(f"Setting degree position is {degree_position}")
        servo_position = self.convert_joint_radians_to_servo(radian_position)
        # print(f"Setting servo position is {servo_position}")

        # if any(angle < -1 or angle > 1000 for angle in servo_position):
        #     raise ValueError("Joint angle of inspire hand out of range!")
        servo_position = np.clip(servo_position, 0, 1000)

        write6(self.serial, self.id, 'angleSet', servo_position)


    def set_tip_position(self, radian_position: np.ndarray):
        '''
        using tip angle
        '''
        assert len(radian_position) == 6

        degree_position = np.degrees(radian_position)
        # print(f"Setting degree position is {degree_position}")
        servo_position = self.convert_degree_to_servo(degree_position)
        # print(f"Setting servo position is {servo_position}")

        # if any(angle < -1 or angle > 1000 for angle in servo_position):
        #     raise ValueError("Joint angle of inspire hand out of range!")
        servo_position = np.clip(servo_position, 0, 1000)

        write6(self.serial, self.id, 'angleSet', servo_position)

    def set_joint_servo_position(self, servo_position: np.ndarray):
        assert len(servo_position) == 6

        # servo_position = np.clip(servo_position, 0, 1000)
        if any(angle < -1 or angle > 1000 for angle in servo_position):
            raise ValueError("Joint angle of inspire hand out of range!")

        servo_set = list()
        for i, position in enumerate(servo_position):
            servo_set.append(int(servo_position[i]))

        write6(self.serial, self.id, 'angleSet', servo_set)

    def read_joint_servo_position(self):
        servo_position = read6(self.serial, self.id, 'angleAct')
        return  servo_position

    def set_joint_speed(self):
        write6(self.serial, self.id, 'speedSet', np.full(6, self.joint_speed))

    def read_tip_position(self):
        servo_angle = read6(self.serial, self.id, 'angleAct')
        # print(f"Reading servo position is {servo_angle}")

        degree_angle = self.convert_servo_to_degree(np.array(servo_angle, dtype=np.float64))
        # print(f"Reading degree position is {degree_angle}")

        return np.deg2rad(degree_angle)

    def read_joint_position2(self):
        servo_angle = read6(self.serial, self.id, 'angleAct')
        radians_angle = self.convert_servo_to_joint_radians(np.array(servo_angle, dtype=np.float64))
        return radians_angle

    def read_joint_force(self):
        joint_force = read6(self.serial, self.id, 'forceAct')
        for i, force in enumerate(joint_force):
            if force > 60000:
                joint_force[i] = joint_force[i] - 65536
        return joint_force

