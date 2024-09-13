from config2 import CONFIG
import time
from loguru import logger
import sys
import datetime
from pion import Pion
import numpy as np

"""
Запускать с такими параметрами:
python ./scripts/data_collection.py 114 ./data/
"""

port_ = CONFIG['standard_port']

drone_number = int(sys.argv[1])

ip_ = f"10.1.100.{drone_number}"

pion = Pion(ip=ip_, mavlink_port=port_)

time.sleep(2)

pion.arm()
time.sleep(2)
pion.takeoff()
pion._mavlink_send_number = 1
time.sleep(5)
pion.goto(-4, 0.0, 1.5)

time.sleep(14)

pion.set_attitude_check()
pion.set_v(ampl=1)
pion.t_speed = np.array([1, 0, 0, 0])

time.sleep(5)

pion.t_speed = np.array([0, 0, 0, 0])
logger.debug("0 0 0 --------------------------------------------")
time.sleep(3)
logger.debug("sleep --------------------------------------------")
pion.speed_flag = False
time.sleep(2)
pion.land()

time.sleep(10)

pion.disarm()

pion.check_attitude_flag = False

current_date = datetime.date.today().isoformat()
current_time = str(datetime.datetime.now().time())
symbols_to_remove = ":"

from os.path import isdir

path = str(sys.argv[2])
if not isdir(path):
    from os import makedirs

    makedirs(path, exist_ok=True)

for symbol in symbols_to_remove:
    current_time = current_time.replace(symbol, "-")
pion.save_data(f'{path}data_1_{drone_number}_{current_date}_{current_time}.npy')
