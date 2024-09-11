from numpy import pi

CONFIG = {
    'standard_port': 5656,  # стандартный порт всех дронов
    'trajectory_write': True,  # Записывать ли каждому дрону траекторию, если не знаете, что это, лучше False
    'ip_3': '10.1.100.',
    'num_drone': 105,  # Это в приложении сбера не используется
    'period_send_v': 0.05,  # Задержка перед отправкой следующего вектора скорости
    'period_get_xyz': 0.05,  # Задержка перед приемом новой позиции
    # Размеры дрона
    'height': 0.12,
    'lenght': 0.29,
    'width': 0.29,
    # Режим тестирования (все работает, кроме старта дронов
    'test': True,
    # Повороты принимаемых координат и отправляемых векторов управления
    'rot_get_pos': 0,
    'rot_send_U': -pi / 2,
    # Рисуем ли в наземной станции график. На сервере без графической оболочки False
    'control_switch': True,  # Подавать ли упарвление на дроны - для проверки координат
    # Писать ли данные о задержках
    'write': True,
    # Время полета
    'time_exp': 60,
}
