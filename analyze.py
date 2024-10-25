import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import os
from os.path import isdir
import shutil
import sys

path = './data/'
save = './save/'
files = os.listdir(path)
args = sys.argv
discription = ''
without_acc = False
without_moving = False
if len(args) > 1:
    if "-d" in args:
        discription = f"{args[args.index('-d') + 1]}"
    if "-na" in args:
        without_acc = True
    if "-nm" in args:
        without_moving = True

for file in files:
    path2 = f'{save}{file[:-4]}/'
    if not isdir(path2):
        from os import makedirs

        makedirs(path2, exist_ok=True)

    array = np.load(f'{path}{file}')
    print(array.shape)
    df = pd.DataFrame(array,
                      columns=['x', 'y', 'z', 'vx', 'vy', 'Vz', 'pitch', 'roll', 'yaw', 'pitch_speed', 'roll_speed',
                               'yaw_speed', 'vx_c', 'vy_c', 'vz_c', 'v_yaw_c', 't'])
    if not without_moving:
        shutil.move(f'{path}{file}', f'{path2}{file}')
    title = f"{file}\n{discription}"

    ### График 1 (все данные)
    ax = df.plot(x='t', title=title, figsize=(10, 6))
    ax.legend(fontsize=8)
    plt.locator_params(axis='x', nbins=30)
    ax.set_xlabel('Время с начала инициализации [с]')
    ax.set_ylabel('Расстояние [м]')
    ax.grid(True)

    # Обычная числовая ось X
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{path2}all.png', dpi=300)

    ### График 2 (полиномиальная регрессия)
    tu = array[:, 16]
    x = array[:, 0]
    vx = array[:, 3]
    t = tu.reshape(-1, 1)

    # Масштабируем время t
    scaler = StandardScaler()
    t_scaled = scaler.fit_transform(t)

    # Перебор вручную для нахождения лучшей степени полинома
    best_degree_x = 1
    best_degree_vx = 1
    best_score_x = float('-inf')
    best_score_vx = float('-inf')

    for degree in range(2, 15):
        poly = PolynomialFeatures(degree)
        t_poly = poly.fit_transform(t_scaled)

        model_x = Ridge(alpha=1e-6)
        model_vx = Ridge(alpha=1e-6)

        scores_x = cross_val_score(model_x, t_poly, x, cv=5, scoring='neg_mean_squared_error')
        scores_vx = cross_val_score(model_vx, t_poly, vx, cv=5, scoring='neg_mean_squared_error')

        mean_score_x = np.mean(scores_x)
        mean_score_vx = np.mean(scores_vx)

        if mean_score_x > best_score_x:
            best_score_x = mean_score_x
            best_degree_x = degree

        if mean_score_vx > best_score_vx:
            best_score_vx = mean_score_vx
            best_degree_vx = degree

    # Трансформация данных с найденными лучшими степенями полиномов
    poly_x = PolynomialFeatures(best_degree_x)
    t_poly_x = poly_x.fit_transform(t_scaled)

    poly_vx = PolynomialFeatures(best_degree_vx)
    t_poly_vx = poly_vx.fit_transform(t_scaled)

    # Обучение модели на полиномиальных признаках
    model_x.fit(t_poly_x, x)
    model_vx.fit(t_poly_vx, vx)

    x_pred = model_x.predict(t_poly_x)
    vx_pred = model_vx.predict(t_poly_vx)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(t, x, color='blue', label='Исходные данные x')
    plt.plot(t, x_pred, color='red', label=f'Полиномиальная регрессия x (степень {best_degree_x})')
    plt.title(f'Полиномиальная регрессия для x (степень {best_degree_x})')
    plt.xlabel('Время (t)')
    plt.ylabel('Координата (x)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(t, vx, color='green', label='Исходные данные vx')
    plt.plot(t, vx_pred, color='orange', label=f'Полиномиальная регрессия vx (степень {best_degree_vx})')
    plt.title(f'Полиномиальная регрессия для vx (степень {best_degree_vx})')
    plt.xlabel('Время (t)')
    plt.ylabel('Скорость (vx)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{path2}reg.png', dpi=300)

    ### График 3 (Random Forest)
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(t_scaled, vx)
    vx_pred_rf = rf_model.predict(t_scaled)

    plt.figure(figsize=(10, 6))
    plt.scatter(t, vx, color='green', label='Исходные данные vx')
    plt.plot(t, vx_pred_rf, color='purple', label='Random Forest vx')
    plt.title('Random Forest для vx')
    plt.xlabel('Время (t)')
    plt.ylabel('Скорость (vx)')
    plt.legend()
    plt.savefig(f'{path2}random_forest.png', dpi=300)

    ### График 4 (скорость и ускорение с разными осями Y)

    plt.figure(figsize=(10, 6))

    # Создаем основной график для скорости
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Первая ось Y для скорости
    ax1.set_xlabel('Время (с)')
    ax1.set_ylabel('Скорость (м/с)', color='green')
    ax1.plot(tu, vx_pred, label='Vx (Reg)', color='orange')
    ax1.plot(df['t'], df['vx_c'], label='Vx (control)', color='red')
    ax1.plot(df['t'], df['vx'], label='Vx (Locus)', color='green')
    ax1.tick_params(axis='y', labelcolor='green')

    # Ограничим масштаб по скорости для улучшенной визуализации
    ax1.set_ylim(-2, 2)

    # Создаем вторую ось Y для ускорения
    ax2 = ax1.twinx()

    # Вычисляем ускорение и сглаживаем его
    vx_dot = np.gradient(vx_pred_rf, tu)
    vx_dot_smooth = gaussian_filter1d(vx_dot, sigma=5)

    # Вторая ось Y для ускорения
    ax2.set_ylabel('Ускорение (м/с²)', color='blue')
    ax2.plot(tu, vx_dot_smooth, label='Ax (gradient smoothed)', color='blue', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Ограничим масштаб по ускорению
    ax2.set_ylim(-10, 10)

    # Добавим легенды для обоих графиков
    fig.tight_layout()
    fig.suptitle(f'График скоростей и ускорений\n{discription}', y=1.05)

    # Сетка и настройки оси X
    ax1.grid(True)
    plt.locator_params(axis='x', nbins=30)
    plt.xticks(rotation=45)
    plt.legend()
    # Сохранение графика
    plt.tight_layout()
    plt.savefig(f'{path2}velocity_acceleration_dual_axis.png', dpi=300)

    if len(args) > 1:
        if "-p" in args:
            plt.show()
        if "-d" in args:
            discription = f"{args[args.index('-d') + 1]}"

    with open(f"{path2}discription.txt", "w") as file_stat:
        file_stat.write(discription)
