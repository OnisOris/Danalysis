import time
from loguru import logger
import sys
import datetime
from pion import Spion, Pion
import numpy as np



"""
-n - номер дрона (стандартно 203)
-p - путь до сохранения результатов (стандартное ./data/)
-s - время подачи вектора скорости (стандартно 3 с)
-g - посылать ли дрон в стартовую точку [-4, 0, 1.5] (стандартно отключено)
-ip - ip дрона можно задать здесь (стандартный 10.1.100.), номер дрона - последняя часть ip
-port - задание порта (стандартный 5656)
Запускать с такими параметрами:
python data_c.py -n 203 -s 3 
"""
args = sys.argv
drone_number = 105
path = "./data/"
time_of_exp = 3
goto = False
ip_ = None
port_ = 5656
spion = False
for arg in args:
    match arg:
        case '-n':
            drone_number = args[args.index(arg)+1]
        case '-p':
            path = args[args.index(arg)+1]
        case '-s':
            time_of_exp = float(args[args.index(arg)+1])
        case '-g':
            goto = True
        case '-ip':
            ip_= args[args.index(arg)+1] 
        case '-port':
            port_ = int(args[args.index(arg)+1])
        case '-sim':
            spion = True


if ip_ is None:
    ip_ = f"10.1.100.{drone_number}"
else:
    ip_ = f"{ip_}{drone_number}"
if spion:
    pion = Spion(ip=ip_, mavlink_port=port_, logger=True)
else:
    pion = Pion(ip=ip_, mavlink_port=port_, logger=True)

pion.check_attitude_flag = True
time.sleep(2)
pion.led_control(255, 0, 255, 0)
pion.arm()
time.sleep(2)
pion.takeoff()
time.sleep(7)
pion.set_v()
if goto:
    pion.goto_from_outside(-4, 0.0, 1.5, 0)

time.sleep(4)

pion.t_speed = np.array([-1, 0, 0, 0])
logger.debug(f"speed = {pion.t_speed}, time_of_exp = {time_of_exp}-----------------------------------")

time.sleep(time_of_exp)

pion.t_speed = np.array([0, 0, 0, 0])
logger.debug(f"speed = {pion.t_speed}, end--------------------------------------------")
time.sleep(3)
logger.debug("sleep --------------------------------------------")
pion.speed_flag = False
time.sleep(2)
pion.land()

time.sleep(13)

pion.disarm()

pion.check_attitude_flag = False

current_date = datetime.date.today().isoformat()
current_time = str(datetime.datetime.now().time())
symbols_to_remove = ":"

from os.path import isdir

if not isdir(path):
    from os import makedirs

    makedirs(path, exist_ok=True)

for symbol in symbols_to_remove:
    current_time = current_time.replace(symbol, "-")
pion.save_data(f'{path}data_1_{drone_number}_{current_date}_{current_time}.npy')
pion.led_control(255, 0, 0, 0)
pion.stop()
