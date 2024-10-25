from loguru import logger
import datetime
from pion import Apion
import sys
import asyncio
import numpy as np
from os.path import isdir


"""
-n - номер дрона (стандартно 203)
-p - путь до сохранения результатов (стандартное ./data/)
-s - время подачи вектора скорости (стандартно 3 с)
-g - посылать ли дрон в стартовую точку [-4, 0, 1.5] (стандартно отключено)
-ip - ip дрона можно задать здесь (стандартный 10.1.100.), номер дрона - последняя часть ip
-port - задание порта (стандартный 5656)
Запускать с такими параметрами:
python data_c_async.py -n 203 
"""
args = sys.argv
drone_number = 203
path = "./data/"
time_of_exp = 3
goto = False
ip_ = None
port_ = 5656
for arg in args:
    if arg == '-n':
        drone_number = args[args.index(arg) + 1]
    if arg == '-p':
        path = args[args.index(arg) + 1]
    if arg == '-s':
        time_of_exp = float(args[args.index(arg) + 1])
    if arg == '-g':
        goto = True
    if arg == '-ip':
        ip_ = args[args.index(arg) + 1]
    if arg == '-port':
        port_ = int(args[args.index(arg) + 1])
if ip_ is None:
    ip_ = f"10.1.100.{drone_number}"
else:
    ip_ = f"{ip_}{drone_number}"
drone = Apion(ip=ip_, mavlink_port=port_)
drone.land()
async def main():

    print('start')
    print(f"ip = {drone.ip}")

    drone.check_attitude_flag = True
    drone.arm()
    drone.takeoff()
    await asyncio.sleep(5)
    v_while_task = asyncio.create_task(drone.set_v_async())
    if goto:
        print("goto")
        await drone.goto_from_outside(-4, 0.0, 1.5, 0)
    # Запуск асинхронного цикла отправки скорости

    logger.debug(f"speed = {drone.t_speed}, time_of_exp = {time_of_exp}-----------------------------------")
    # Попробуем двигать дрон
    drone.t_speed = np.array([1, 0, 0, 0])

    await asyncio.sleep(time_of_exp)
    drone.t_speed = np.array([0, 0, 0, 0])
    logger.debug(f"speed = {drone.t_speed}, count = {drone.count} end-------------------------------------------- ")
    await asyncio.sleep(3)
    drone.land()

    # Остановим все задачи
    drone.stop()
    v_while_task.cancel()  # Отменим задачу, чтобы остановить цикл
    if not isdir(path):
        from os import makedirs

        makedirs(path, exist_ok=True)
    current_date = datetime.date.today().isoformat()
    current_time = str(datetime.datetime.now().time())
    symbols_to_remove = ":"
    for symbol in symbols_to_remove:
        current_time = current_time.replace(symbol, "-")
    drone.save_data(f'{path}data_1_{drone_number}_{current_date}_{current_time}.npy')
    drone.stop()


if __name__ == "__main__":
    asyncio.run(main())








