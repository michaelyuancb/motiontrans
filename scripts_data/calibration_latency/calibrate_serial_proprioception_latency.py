import time
import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import click
from real.active_vision_utils.TeleVision import OpenTeleVision
from real.active_vision_utils.dynamixel.dynamixel_robot import DynamixelRobot
from real.active_vision_utils.dynamixel.active_cam import DynamixelAgent
from real.inspire_hand_utils.inspire_hand_agent import InspireHandAgent
from real.active_vision_utils.constants_vuer import *

@click.command()
@click.option('-r', '--robot', type=str, default="active_vision")
def main(robot):
    if robot == "active_vision":
        active_vision_agent = DynamixelAgent(port="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA3H8CB-if00-port0")
    elif robot == "inspire_hand":
        inspire_hand_agent = InspireHandAgent(port='/dev/ttyUSB0',
                                              hand_id=1,
                                              baudrate=115200
                                             )
    else:
        raise ValueError("Robot type wrong!")
    send_time = time.monotonic()
    if robot == "active_vision":
        pose = active_vision_agent._robot.read_joint_state()
    else:
        pose = inspire_hand_agent.read_joint_servo_position()
    receive_time = time.monotonic()
    delta_time = receive_time - send_time
    print(f"received pose is: {pose}")
    print(f"send and receive signal time interval is: {delta_time}")
    print(f"Proprioception latency is: {0.5 * delta_time}")

if __name__ == '__main__':
    main()
