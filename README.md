# Danalysis
Данный модуль нужен для анализа поведения роботов и дронов.
Скрипты тестировались на Windows 11, Ubuntu 22, Manjaro linux.  

## Описание
В данный момент это просто набор скриптов для сбора информации с базовых пионеров и 
аналитика информации их поведения
## Зависимости
Перед первым запуском нужно создать виртуальное окружение и установить в него зависимости:
```Shell
pip install -r req.txt
```

## Запуск сбора информации с дронов

Допустим номер дрона 105.

В случае, если данные мы будем собирать в папке data, то:

Windows & Manjaro linux:
```Shell
python data_c.py -n 105
```

Ubuntu:
```Shell
python3 data_c.py -n 105
```
## Построение графиков

В аргументах указываем ключи: -p - если нужно отображение графиков, -d [Описание] - ключ для описания происходящего.


Windows & Manjaro linux:
```Shell
python analyze.py -p -d "Описание, которое будет записано в файл dicription.txt и отображаться на графике"
```

Ubuntu:
```Shell
python3 analyze.py -p -d "Описание"
```
## Изменение параметров программы

Поменять станлартные параметры можно аргументами:
-n - номер дрона (стандартно 105)
-p - путь до сохранения результатов (стандартное ./data/)
-s - время подачи вектора скорости (стандартно 3 с)
-g - посылать ли дрон в стартовую точку [-4, 0, 1.5] (стандартно отключено)
-ip - ip дрона можно задать здесь (стандартный 10.1.100.), номер дрона - последняя часть ip
-port - задание порта (стандартный 5656)
Запускать с такими параметрами:

```Shell
python data_c.py -n 105 -s 3 
```

Пример полной команды:

```Shell
python data_c.py -n 105 -p ./data/ -s 3 -g -ip 10.1.100. -port 5656
```




