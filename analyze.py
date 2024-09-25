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

path = './data/'
save = './save/'
files = os.listdir(path)
args = sys.argv 
for file in files:
    path2 = f'{save}{file[:-4]}/'
    if not isdir(path2):
        from os import makedirs
        makedirs(path2, exist_ok=True)

    array = np.load(f'{path}{file}')
    df = pd.DataFrame(array, columns=['x', 'y', 'z', 'vx', 'vy', 'Vz', 'vx_c', 'vy_c', 'vz_c', 'v_yaw_c', 't'])
    shutil.move(f'{path}{file}', f'{path2}{file}')
    title = f"{file}\n"

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
    tu = array[:, 10]
    x = array[:, 0]
    y = array[:, 1]
    vx = array[:, 3]
    t = tu.reshape(-1, 1)

    best_degree_x = 25
    best_degree_y = 25
    best_degree_vx = 25

    scaler = StandardScaler()
    t_scaled = scaler.fit_transform(t)

    poly_x = PolynomialFeatures(best_degree_x)
    t_poly_x = poly_x.fit_transform(t_scaled)

    poly_y = PolynomialFeatures(best_degree_y)
    t_poly_y = poly_y.fit_transform(t_scaled)

    poly_vx = PolynomialFeatures(best_degree_vx)
    t_poly_vx = poly_vx.fit_transform(t_scaled)

    model_x = Ridge(alpha=1e-6)
    model_y = Ridge(alpha=1e-6)
    model_vx = Ridge(alpha=1e-6)

    model_x.fit(t_poly_x, x)
    model_y.fit(t_poly_y, y)
    model_vx.fit(t_poly_vx, vx)

    x_pred = model_x.predict(t_poly_x)
    y_pred = model_y.predict(t_poly_y)
    vx_pred = model_vx.predict(t_poly_vx)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(t, x, color='blue', label='Исходные данные x')
    plt.plot(t, x_pred, color='red', label='Полиномиальная регрессия x')
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

    ### График 3 (скорость и ускорение)
    plt.figure(figsize=(10, 6))
    vx_dot = np.gradient(vx_pred, tu)

    plt.plot(tu, vx_dot, label='Ax (gradient(Vx))', color='blue', linestyle='--')
    plt.plot(tu, vx_pred, label='Vx (Reg)', color='orange')
    plt.plot(df['t'], df['vx_c'], label='Vx (control)', color='red')
    plt.plot(df['t'], df['vx'], label='Vx (Locus)', color='green')

    plt.xlabel('Время (с)')
    plt.ylabel('Скорость (м/с) и ускорение (м/с²)')
    plt.title('График скоростей и ускорений')
    plt.legend()
    plt.grid(True)

    # Увеличиваем количество меток на оси X
    plt.locator_params(axis='x', nbins=30)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{path2}velocity_acceleration.png', dpi=300)
    discription = ''
    if len(args) > 1:
       if "-p" in args:
            plt.show()
       if "-d" in args:
            discription = f"{args[args.index('-d') + 1]}"
    with open(f"{path2}discription.txt", "w") as file_stat:
       file_stat.write(discription)
