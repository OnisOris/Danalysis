# Danalysis
Данный модуль нужен для анализа поведения роботов и дронов.
Скрипты тестировались на Windows 11, Ubuntu 22, Manjaro linux.  

## Описание
В данный момент это просто набор скриптов для сбора информации с базовых пионеров и 
аналитика информации их поведения
## Зависимости
Перед первым запуском нужно установить зависимости:
```Shell
pip install -r req.txt
```

## Запуск сбора информации с дронов

Допустим номер дрона 114.

В случае, если данные мы будем собирать в папке data, то:

Windows:
```Shell
python ./scripts/data_collection.py 114 ./data/
```

Ubuntu:
```Shell
python3 ./scripts/data_collection.py 114 ./data/
```
## Построение графиков

В аргументах указываем ключи: -p - если нужно отображение графиков, -d [Описание] - ключ для описания происходящего.


Windows:
```Shell
python analyze.py -p -d "Описание, которое будет записано в файл dicription.txt и отображаться на графике"
```

Ubuntu:
```Shell
python analyze.py -p -d "Описание"
```
