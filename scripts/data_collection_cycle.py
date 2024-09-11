from SwarmControl import Drone
from config import CONFIG
from pioneer_sdk import Pioneer
import time
from loguru import logger
import sys
import datetime
port_ = CONFIG['standard_port']

drone_number = int(sys.argv[1])

ip_ = f"10.1.100.{drone_number}"


pioneer = Pioneer(name='pioneer', ip=ip_, mavlink_port=port_, logger=True)

drone = Drone(CONFIG, drone=pioneer, joystick_on=False, apply=True)
# if drone.body.real_point[2] < 0.5:
#     pioneer.reboot_board()
# time.sleep(5)
pioneer.arm()
time.sleep(2)
pioneer.takeoff()
time.sleep(5)
# Количество повторений эксперимента
n = 2
for i in range(n):
    pioneer.go_to_local_point(-4.52890205, 0.06610023, 1.5, 0)
    drone.goto([-3, 0.0, 1.5], apply=True)
    time.sleep(14)
    drone.set_coord_check()
    drone.set_v(ampl=2)
    drone.body.v = [1, 0, 0]
    time.sleep(3)
    drone.body.v = [0, 0, 0]
    logger.debug("0 0 0 --------------------------------------------")
    time.sleep(3)
    logger.debug("sleep --------------------------------------------")
    drone.speed_flag = False

    current_date = datetime.date.today().isoformat()
    current_time = str(datetime.datetime.now().time())
    symbols_to_remove = ":"
    # current_time = current_time[:7]
    for symbol in symbols_to_remove:
        current_time = current_time.replace(symbol, "-")
    time.sleep(3)
    drone.save_data(f'../data/data_1_{drone_number}_{current_date}_{current_time}.csv')
    time.sleep(5)

time.sleep(2)
pioneer.land()
drone.stop()
time.sleep(10)
drone.disarm()
drone.stop()