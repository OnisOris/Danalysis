import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from os.path import isdir
import shutil
import sys

# python ./analyze.py ./data/ ./save/

path = sys.argv[1]
save = sys.argv[2]
files = os.listdir(path)
for file in files:
    path2 = f'{save}{file[:-4]}/'
    if not isdir(path2):
        from os import makedirs

        makedirs(path2, exist_ok=True)
    array = np.load(f'{path}{file}')
    df = pd.DataFrame(array, columns=['x', 'y', 'z', 'vx', 'vy', 'Vz', 'vx_c', 'vy_c', 'vz_c', 'v_yaw_c', 't'])
    df.plot(x='t')
    shutil.move(f'{path}{file}', f'{path2}{file}')
    title = f"{file}\n"
    df = df[1:]

    ### График 1
    df.plot(x='t', title=title)
    plt.xlabel('Время с начала инициализации [с]')
    plt.ylabel('Расстояние [м]')
    plt.savefig(f'{path2}all.png')
    ### График 2
    # array = df.to_numpy()
    tu = array[:, 10]
    x = array[:, 0]
    y = array[:, 1]

    t = tu.reshape(-1, 1)
    # Здесь потом нужно сделать алгоритм для поиска лучшей степени полинома
    best_degree_x = 25
    best_degree_y = 25
    # Нормализуем время t
    scaler = StandardScaler()
    t_scaled = scaler.fit_transform(t)
    # Теперь применим полиномиальную регрессию
    # x
    poly_x = PolynomialFeatures(best_degree_x)
    t_poly_x = poly_x.fit_transform(t_scaled)
    # y
    poly_y = PolynomialFeatures(best_degree_y)
    t_poly_y = poly_y.fit_transform(t_scaled)

    # Используем Ridge для регуляризации
    model_x = Ridge(alpha=1e-6)
    model_y = Ridge(alpha=1e-6)

    model_x.fit(t_poly_x, x)
    model_y.fit(t_poly_y, y)

    # Предсказанные значения
    x_pred = model_x.predict(t_poly_x)
    y_pred = model_y.predict(t_poly_y)

    # Визуализация
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(t, x, color='blue', label='Исходные данные x')
    plt.plot(t, x_pred, color='red', label='Полиномиальная регрессия x')
    plt.title(f'Полиномиальная регрессия для x (степень {best_degree_x})')
    plt.xlabel('Время (t)')
    plt.ylabel('Координата (x)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(t, y, color='green', label='Исходные данные y')
    plt.plot(t, y_pred, color='orange', label=f'Полиномиальная регрессия y (степень {best_degree_y})')
    plt.title(f'Полиномиальная регрессия для y (степень {best_degree_y})')
    plt.xlabel('Время (t)')
    plt.ylabel('Координата (y)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{path2}reg.png')
    # plt.show()

    ### График 3
    # Построение графика
    plt.figure()
    x_dot = np.gradient(x_pred, tu)
    y_dot = np.gradient(y_pred, tu)

    df_dot2 = pd.DataFrame(np.vstack([x_dot, tu]).T, columns=['x_dot', 't'])
    df_dot2.plot(x='t', title=title)

    #
    x_ddot = np.gradient(x_dot, tu)
    y_ddot = np.gradient(y_dot, tu)

    df_ddot_y = pd.DataFrame(np.vstack([y_ddot, array[:, 4], tu]).T, columns=['y_ddot', 'vy', 't'])

    df_ddot_x = pd.DataFrame(np.vstack([x_ddot, array[:, 3], tu]).T, columns=['x_ddot', 'vx', 't'])

    # Линия для x2
    plt.plot(df_ddot_x['t'], df_ddot_x['x_ddot'], label='x_ddot')

    # # Линия для x3
    plt.plot(df['t'], df['vx_c'], label='Vx (control)')
    plt.plot(df['t'], df['vx'], label='Vx (Locus)')
    # Настройки графика
    plt.xlabel('Time (s)')
    plt.title('Graphic of accelerations and velocities')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{path2}velocity_acceleration.png')
    str_stat = (f"Mean acceleration = {round(df_ddot_x.iloc[:, 0].mean(), 4)}, "
                f"\n Median acceleration - {round(df_ddot_x.iloc[:, 0].median(), 4)}")
    with open(f"{path2}stat.txt", "w") as file_stat:
        file_stat.write(str_stat)
    print(str_stat)
