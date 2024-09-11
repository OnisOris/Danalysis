from SwarmControl import Drone
from config2 import CONFIG
from pioneer_sdk import Pioneer
import time
from loguru import logger
import sys
import datetime

# python .\scripts\data_collection.py 114 ./data/

port_ = CONFIG['standard_port']

drone_number = int(sys.argv[1])

ip_ = f"10.1.100.{drone_number}"


pioneer = Pioneer(name='pioneer', ip=ip_, mavlink_port=port_, logger=True)

drone = Drone(CONFIG, drone=pioneer, joystick_on=False, apply=True)
if drone.body.real_point[2] < 0.5:
    pioneer.reboot_board()
time.sleep(5)

pioneer.arm()
time.sleep(2)
pioneer.takeoff()
pioneer._mavlink_send_number = 1
time.sleep(5)
# pioneer.go_to_local_point(-3, 0, 1.5, 0)
drone.goto([-4, 0.0, 1.5], apply=True)

time.sleep(14)

drone.set_coord_check()
drone.set_v(ampl=1)

drone.body.v = [1, 0, 0]

time.sleep(5)

drone.body.v = [0, 0, 0]
logger.debug("0 0 0 --------------------------------------------")
time.sleep(3)
logger.debug("sleep --------------------------------------------")
drone.speed_flag = False
# drone.xyz_flag = False
time.sleep(2)
drone.land()

drone.stop()


time.sleep(10)

drone.disarm()
drone.stop()

current_date = datetime.date.today().isoformat()
current_time = str(datetime.datetime.now().time())
symbols_to_remove = ":"
# current_time = current_time[:7]

from os.path import isdir
path = str(sys.argv[2])
if not isdir(path):
    from os import makedirs
    makedirs(path, exist_ok=True)

for symbol in symbols_to_remove:
    current_time = current_time.replace(symbol, "-")
drone.save_data(f'{path}data_1_{drone_number}_{current_date}_{current_time}.csv')