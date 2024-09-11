# Danalysis
Данный модуль нужен для анализа поведения роботов и дронов.
Скрипты тестировались на Windows 11 и Ubuntu 22.  

## Описание
В данный момент это просто набор скриптов для сбора информации с базовых пионеров и 
аналитика информации их поведения
## Зависимости
Перед первым запуском нужно установить зависимости:
```Shell
pip install -r req.txt
```

Обращаю внимание, что в моих экспериментах используется fork pioneer_sdk 
https://github.com/OnisOris/pioneer_sdk
а также модуль https://github.com/OnisOris/SwarmControl для более удобного управления дроном 

Если вы будете дополнительно пользоваться Jupyter блокнотом, то:
```Shell
pip install jupyter
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

В аргументах указываем папку data с данными формата csv,
после этого вторым аргументом указывается папка,
где будут созданы папки с данными.

Windows:
```Shell
python ./analyze.py ./data/ ./save/
```

Ubuntu:
```Shell
python3 ./analyze.py ./data/ ./save/
```
