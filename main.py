import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from subprocess import run
from sklearn.cluster import KMeans


R_BIN_PATH = 'C:\\Program Files\\R\\R-4.0.4\\bin'


input_matrix = []
filenames = []
eigenvectors_c = []
eigenvalues_c = []
eigenvectors_b = []
eigenvalues_b = []
input_df_list = []


def read_files():
    input_data = input('[PCA]: чтение входных файлов\n' +
                       '[PCA]: выберите параметра исследования:\n' +
                       '[PCA]: введите необходимый диапазон в формате "xx-xxxx, yy-yyyy, ..." = ')
    input_data = input_data.replace(' ', '')
    input_waves_temp = input_data.split(',')
    input_waves = []
    for i in input_waves_temp:
        temp = []
        for j in i.split('-'):
            temp.append(int(j))
        if temp[0] > temp[1]:
            temp[0], temp[1] = temp[1], temp[0]
        input_waves.append(temp)
    print(input_waves)

    normalization = 'y'
    number = input('[PCA]: нормализовать данные (y/n) (по умолчанию y): ')
    if number != '':
        normalization = number
    os.chdir(os.curdir)

    for file_path in glob.glob("input_dpt\\*.dpt"):
        input_df = pd.read_csv(file_path, header=None)
        input_df_list.append(input_df)
        filename = file_path.replace('input_dpt\\', '')
        filename = filename.replace('.dpt', '')
        filename = filename.replace('-', '')
        filename = filename[0:3]
        filename = filename.replace('_0', '')
        filename = filename.replace('_', '')
        filenames.append(filename)
        one_pat = []
        for i in range(len(input_waves)):
            input_df_temp = input_df.drop(input_df[input_df[0] < input_waves[i][0]].index)
            input_df_temp = input_df_temp.drop(input_df_temp[input_df_temp[0] > input_waves[i][1]].index)
            one_pat = np.concatenate((one_pat, input_df_temp[1].values.astype('float')))
        input_matrix.append(one_pat)
        print(input_matrix)
    cutting_spectra_and_finding_ratio(input_df_list)
    print('[PCA]: файлы прочитаны')
    if normalization == 'Y' or normalization == 'y':
        for index in range(len(input_matrix[0])):
            column = [row[index] for row in input_matrix]
            m = np.mean(column)
            s = np.std(column)
            normalization_function = lambda x: (x - m) / s
            normalized_column = normalization_function(column)
            for str_index in range(len(input_matrix)):
                input_matrix[str_index][index] = normalized_column[str_index]
        print('[PCA]: нормализация произведена')
    else:
        print('[PCA]: нормализация пропущена')


def cutting_spectra_and_finding_ratio(input_df_list):
    diapason = [[1600, 1650], [1580, 1600], [1550, 1580], [1500, 1510], [1440, 1470]]
    baseline = [1720, 2000]
    max_min = []
    ratio = []
    waves = []
    for input_df in input_df_list:
        # Создание диапазона для построения графиков сравнений
        intensities = []
        waves_in_diapason = []
        # из всего диапазона длин вол вырезаются нужные фрагменты с макс. и мин.
        for i in diapason:
            temp = input_df.drop(input_df[input_df[0] <= i[0]].index)
            temp = temp.drop(temp[temp[0] >= i[1]].index)
            intensities.append(temp[1].values.astype('float').tolist())
            waves_in_diapason.append(temp[0].values.astype('float').tolist())
        # базовая линия 2000-1700 см-1 и её среднее, которое вычтется из остального спектра
        base = input_df.drop(input_df[input_df[0] <= baseline[0]].index)
        base = base.drop(base[base[0] >= baseline[1]].index)
        base = base[1].values.astype('float')
        coefficient = np.mean(base)

        # находим макс. и мин., вычитаем из них среднее по базовой линии
        max_min_temp = []
        waves_temp = []
        max_min_temp.append(max(intensities[0]) - coefficient)
        waves_temp.append(waves_in_diapason[0][intensities[0].index(max(intensities[0]))])
        max_min_temp.append(min(intensities[1]) - coefficient)
        waves_temp.append(waves_in_diapason[1][intensities[1].index(min(intensities[1]))])
        max_min_temp.append(max(intensities[2]) - coefficient)
        waves_temp.append(waves_in_diapason[2][intensities[2].index(max(intensities[2]))])
        max_min_temp.append(min(intensities[3]) - coefficient)
        waves_temp.append(waves_in_diapason[3][intensities[3].index(min(intensities[3]))])
        max_min_temp.append(max(intensities[4]) - coefficient)
        waves_temp.append(waves_in_diapason[4][intensities[4].index(max(intensities[4]))])
        max_min.append(max_min_temp)
        waves.append(waves_temp)

        # находим отношения максимумов и минимумов попарно
        ratio_temp = []
        for i in range(len(max_min_temp)):
            for j in range(i + 1, len(max_min_temp)):
                ratio_temp.append(max_min_temp[i]/max_min_temp[j])
        ratio.append(ratio_temp)
    return waves, ratio


def sorting_ratio_and_waves_by_names():
    waves, ratio = cutting_spectra_and_finding_ratio(input_df_list)
    # сортируем все интенсивности из фаилов по донорам, секретирующим и несекретирующим пациентам
    donor = []
    waves_d = []
    patient = []
    waves_p = []
    non_secreting = []
    waves_n = []
    for i in range(len(filenames)):
        if (filenames[i][0] == 'D') or (filenames[i][0] == 'O') or (filenames[i][0] == 'B') or (filenames[i][0] == 'H'):
            donor.append(ratio[i])
            waves_d.append(waves[i])
        elif filenames[i][0] == 'P' or (filenames[i][0] == 'M'):
            patient.append(ratio[i])
            waves_p.append(waves[i])
        elif filenames[i][0] == 'N':
            non_secreting.append(ratio[i])
            waves_n.append(waves[i])
    return donor, waves_d, patient, waves_p, non_secreting, waves_n


def calculate_ratio():
    waves, ratio = cutting_spectra_and_finding_ratio(input_df_list)
    donor, waves_d, patient, waves_p, non_secreting, waves_n = sorting_ratio_and_waves_by_names()
    # находим среднее по каждой группе образцов
    normal = []
    result_d = [0] * len(ratio[0])
    for i in donor:
        result_d = [x + y for x, y in zip(result_d, i)]
    for i in range(len(result_d)):
        if len(donor) != 0:
            result_d[i] = result_d[i]/len(donor)
            normal.append(1 / result_d[i])
            result_d[i] = result_d[i]*(1/result_d[i])

    result_waves_d = [0] * len(waves[0])
    for i in waves_d:
        result_waves_d = [x + y for x, y in zip(result_waves_d, i)]
    for i in range(len(result_waves_d)):
        if len(waves_d) != 0:
            result_waves_d[i] = result_waves_d[i] / len(waves_d)

    result_p = [0] * len(ratio[0])
    for i in patient:
        result_p = [x + y for x, y in zip(result_p, i)]
    for i in range(len(result_p)):
        if len(patient) != 0:
            result_p[i] = (result_p[i]/len(patient)) * normal[i]

    result_waves_p = [0] * len(waves[0])
    for i in waves_p:
        result_waves_p = [x + y for x, y in zip(result_waves_p, i)]
    for i in range(len(result_waves_p)):
        if len(waves_p) != 0:
            result_waves_p[i] = result_waves_p[i] / len(waves_p)

    result_n = [0] * len(ratio[0])
    for i in non_secreting:
        result_n = [x + y for x, y in zip(result_n, i)]
    for i in range(len(result_n)):
        if len(non_secreting) != 0:
            result_n[i] = (result_n[i] / len(non_secreting)) * normal[i]

    result_waves_n = [0] * len(waves[0])
    for i in waves_n:
        result_waves_n = [x + y for x, y in zip(result_waves_n, i)]
    for i in range(len(result_waves_n)):
        if len(waves_n) != 0:
            result_waves_n[i] = result_waves_n[i] / len(waves_n)
    return result_d, result_waves_d, result_p, result_waves_p, result_n, result_waves_n, normal


def calculate_and_sort_eigenvalues_and_vectors():
    # вычисляем транспонированную матрицу от входной
    x_matrix = np.array(input_matrix)
    x_matrix_transpose = x_matrix.transpose()

    # перемножаем, x_matrix * x_matrix_transpose = c_matrix
    # перемножаем, x_matrix_transpose * x_matrix = b_matrix
    c_matrix = x_matrix @ x_matrix_transpose
    b_matrix = x_matrix_transpose @ x_matrix

    # записать c_matrix и b_matrix в файл
    c_matrix_dataframe = pd.DataFrame(c_matrix)
    b_matrix_dataframe = pd.DataFrame(b_matrix)
    c_matrix_dataframe.to_csv('R_script\\input\\c_matrix.csv', index=False, header=None)
    b_matrix_dataframe.to_csv('R_script\\input\\b_matrix.csv', index=False, header=None)

    # запуск скрипта на R для поиска
    # собственных чисел и собственных векторов c_matrix и b_matrix
    current_directory = os.path.abspath(os.curdir)
    run([R_BIN_PATH + '\\Rscript.exe', current_directory +
         '\\R_script\\eigenvalues_vectors.R', current_directory])

    # прочитать собственне числа и собственные вектора c_matrix и b_matrix
    eigenvalues_c_tmp = pd.read_csv('R_script\\output\\eigenvalues_c.csv', header=None).values.tolist()
    eigenvalues_b_tmp = pd.read_csv('R_script\\output\\eigenvalues_b.csv', header=None).values.tolist()
    eigenvectors_c_tmp = pd.read_csv('R_script\\output\\eigenvectors_c.csv', header=None).values.tolist()
    eigenvectors_b_tmp = pd.read_csv('R_script\\output\\eigenvectors_b.csv', header=None).values.tolist()

    os.remove('R_script\\input\\c_matrix.csv')
    os.remove('R_script\\input\\b_matrix.csv')
    os.remove('R_script\\output\\eigenvalues_c.csv')
    os.remove('R_script\\output\\eigenvalues_b.csv')
    os.remove('R_script\\output\\eigenvectors_c.csv')
    os.remove('R_script\\output\\eigenvectors_b.csv')

    for value in eigenvalues_c_tmp:
        eigenvalues_c.append(value[0])
    for value in eigenvalues_b_tmp:
        eigenvalues_b.append(value[0])
    for vector in eigenvectors_c_tmp:
        eigenvectors_c.append(vector)
    for vector in eigenvectors_b_tmp:
        eigenvectors_b.append(vector)


def calculate_t_and_p_matrix():
    u_values, u_vectors, v_values, v_vectors = eigenvalues_c, eigenvectors_c, eigenvalues_b, eigenvectors_b
    u_vectors = np.array(u_vectors).transpose()
    v_vectors = np.array(v_vectors).transpose()

    # строим матрицу S, изначально заполняем нулями
    s_matrix = np.zeros((len(u_values), len(v_values)))

    # какие-то числа совпадают, смотрим, где чисел меньше
    if len(u_values) > len(v_values):
        eigenvalues = v_values
    else:
        eigenvalues = u_values

    # на диагональ матрицы S ставим собственные числа
    for index in range(len(eigenvalues)):
        s_matrix[index][index] = (abs(eigenvalues[index])) ** (1 / 2)

    # находим матрицы T и P
    t_matrix = u_vectors @ s_matrix
    p_matrix = v_vectors
    # g = u_vectors @ s_matrix @ v_vectors
    return t_matrix, p_matrix


def show_graphic_of_eigenvalues_and_pc():
    u_values, u_vectors, v_values, v_vectors = eigenvalues_c, eigenvectors_c, eigenvalues_b, eigenvectors_b

    # какие-то числа совпадают, смотрим, где чисел меньше
    # if len(u_values) > len(v_values):
        # eigenvalues = v_values
    # else:
        # eigenvalues = u_values
    # eigenvalues_plot = np.insert(eigenvalues, 0, np.sum(eigenvalues))

    # x = [i for i in range(len(eigenvalues_plot))]
    # y = eigenvalues_plot
    # plt.plot(x, y, color='black', linestyle='solid', linewidth=2, marker='o', markerfacecolor='red', markersize=8)
    # plt.xlabel('номер значения')
    # plt.ylabel('значение')
    # plt.title('собственные значения')
    # print('[PCA]: вывожу график собственных значений...')
    # plt.show()

    # I = len(eigenvectors_b[0])
    # J = len(eigenvectors_c[0])
    # lambda_0 = sum(eigenvalues)
    # TRV = []
    # for i in range(0, len(eigenvalues)):
        # RV.append((1 / (I * J)) * (lambda_0 - sum(eigenvalues[0:i])))
    # ERV = []
    # for trv in TRV:
        # ERV.append(1 - (trv / ((1 / (I * J)) * lambda_0)))

    # x = [i for i in range(len(eigenvalues))]
    # y = TRV
    # plt.plot(x, y, color='red', linestyle='solid', linewidth=2, marker='o', markerfacecolor='black', markersize=8)
    # y = ERV
    # plt.plot(x, y, color='green', linestyle='solid', linewidth=2, marker='o', markerfacecolor='black', markersize=8)
    # plt.xlabel('номер значения')
    # plt.ylabel('TRV, ERV')
    # plt.title('главные компоненты\nTRV - красный\nERV - зеленый')
    # print('[PCA]: вывожу график главных компонент...')
    # plt.show()


def show_graphic_of_t_matrix():
    t_matrix, p_matrix = calculate_t_and_p_matrix()
    while True:
        command = int(input('[PCA]: какой график хотите построить:\n' +
                            '[PCA]: 1 - 2D график матрицы T\n' +
                            '[PCA]: 2 - 2D график матрицы P\n' +
                            '[PCA]: 3 - 3D график матрицы T\n' +
                            '[PCA]: 4 - Найти средние отношения поглощения \n' +
                            '[PCA]: 5 - Найти отношения поглощений для определенных \n' +
                            '[PCA]: 6 - закончить работу...\n' +
                            '[PCA]: '))

        if command == 1 or command == 2:
            first_column = int(input('[PCA]: введите номера столбцов для графика:\n' +
                                     '[PCA]: для оси X: '))
            second_column = int(input('[PCA]: для оси Y: '))
            first_column -= 1
            second_column -= 1
            if command == 1:
                x = t_matrix[:, first_column]
                y = t_matrix[:, second_column]
                ax = plt.gca()
                ax.spines['left'].set_position('center')
                ax.spines['bottom'].set_position('center')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                print('Файлов всего было {}'.format(len(filenames)))
                for index in range(len(filenames)):
                    if (filenames[index][0] == 'P') or (filenames[index][0] == 'M'):
                        plt.scatter(x[index], y[index], color="red", marker="o", s=50)
                        # plt.annotate(filenames[index], (x[index], y[index]))
                    elif filenames[index][0] == 'N':
                        plt.scatter(x[index], y[index], color="blue", marker="o", s=50)
                        # plt.annotate(filenames[index], (x[index], y[index]))
                    elif filenames[index][0] == 'D':
                        plt.scatter(x[index], y[index], color="green", marker="o", s=50)
                        # plt.annotate(filenames[index], (x[index], y[index]))
                    elif (filenames[index][0] == 'O') or (filenames[index][0] == 'B'):
                        plt.scatter(x[index], y[index], color="black", marker="o", s=50)
                        # plt.annotate(filenames[index], (x[index], y[index]))
                print('[PCA]: вывожу график...')
                plt.show()
            elif command == 2:
                x = p_matrix[:, first_column]
                y = p_matrix[:, second_column]
                ax = plt.gca()
                ax.spines['left'].set_position('center')
                ax.spines['bottom'].set_position('center')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.scatter(x, y, color="black", marker="o", s=12)
                print('[PCA]: вывожу график...')
                plt.show()
        elif command == 3:
            first_column = int(input('[PCA]: введите номера столбцов для графика:\n' +
                                     '[PCA]: для оси X: '))
            second_column = int(input('[PCA]: для оси Y: '))
            third_column = int(input('[PCA]: для оси Z: '))
            first_column -= 1
            second_column -= 1
            third_column -= 1
            x = t_matrix[:, first_column]
            y = t_matrix[:, second_column]
            z = t_matrix[:, third_column]
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title('Three principal components:')
            for index in range(len(filenames)):
                if (filenames[index][0] == 'P') or (filenames[index][0] == 'M'):
                    ax.scatter(x[index], y[index], z[index], color="red", marker="o", s=40)
                elif filenames[index][0] == 'N':
                    ax.scatter(x[index], y[index], z[index], color="blue", marker="o", s=40)
                elif filenames[index][0] == 'D':
                    ax.scatter(x[index], y[index], z[index], color="green", marker="o", s=40)
                elif (filenames[index][0] == 'O') or (filenames[index][0] == 'B'):
                    ax.scatter(x[index], y[index], z[index], color="black", marker="o", s=40)

            number_of_clusters = int(input('Введите число кластеров:'))
            kmeans_PCA = KMeans(n_clusters=number_of_clusters)
            X_kmeans_PCA = t_matrix[:, :3]
            y_kmeans_PCA = kmeans_PCA.fit_predict(X_kmeans_PCA)
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_kmeans_PCA[:, 0], X_kmeans_PCA[:, 1], X_kmeans_PCA[:, 2],
                       c=y_kmeans_PCA, cmap='viridis',
                       edgecolor='k', s=40, alpha=0.5)

            ax.set_title("First three PCA directions")
            ax.set_xlabel("1")
            ax.set_ylabel("2")
            ax.set_zlabel("3")
            ax.dist = 10

            ax.scatter(kmeans_PCA.cluster_centers_[:, 0], kmeans_PCA.cluster_centers_[:, 1],
                       kmeans_PCA.cluster_centers_[:, 2],
                       s=300, c='r', marker='*', label='Centroid')

            plt.autoscale(enable=True, axis='x', tight=True)

            plt.show()

        elif command == 4:
            result_d, result_waves_d, result_p, result_waves_p, result_n, result_waves_n, normal = calculate_ratio()
            # Пересчитываем погрешность на основе данных из расчётов в excel, которые тут не будут представлены
            error_radial = [0.5, 0.4, 0.001, 0.07, 0.01, 0.001, 0.001, 0.2, 0.03, 0.001]
            for i in range(len(error_radial)):
                error_radial[i] = error_radial[i] * normal[i]
            print('Прогрешность:', error_radial)

            # рисуем графики
            labels = ['A1/M1', 'A1/A2', 'A1/M2', 'A1/A3', 'M1/A2', 'M1/M2', 'M1/A3', 'A2/M2', 'A2/A3', 'M2/A3']
            theta = np.linspace(start=0, stop=2 * np.pi, num=len(result_d), endpoint=False)
            theta = np.concatenate((theta, [theta[0]]))
            result_d = np.append(result_d, result_d[0])
            result_p = np.append(result_p, result_p[0])
            result_n = np.append(result_n, result_n[0])
            error_radial = np.append(error_radial, error_radial[0])
            fig = plt.figure(figsize=(10, 10), facecolor='#f3f3f3')
            ax = fig.add_subplot(111, projection='polar')
            ax.errorbar(theta, result_d, linewidth=2, xerr=0, yerr=error_radial, color="green", ecolor='black')
            ax.errorbar(theta, result_p, linewidth=2, xerr=0, yerr=0, color="red")
            ax.errorbar(theta, result_n, linewidth=2, xerr=0, yerr=0, color="blue")
            ax.set_thetagrids(range(0, 360, int(360 / len(labels))), labels)
            plt.yticks(np.arange(0, 1.5, 0.2), fontsize=8)
            ax.set(facecolor='#f3f3f3')
            ax.set_theta_offset(np.pi / 2)

            pl = ax.yaxis.get_gridlines()
            for line in pl:
                line.get_path()._interpolation_steps = 5
            plt.show()

            cat_par = ['Amide-I', 'Min 1-2', 'Amide-II', 'Min 2-3', 'Amide-III']
            g1 = result_waves_d
            g2 = result_waves_p
            g3 = result_waves_n
            width = 0.3

            error_d = np.array([0.89, 0.364, 0.625, 0.483, 0.246]).T
            error_p = np.array([0.1, 0.3, 0.2, 0.4, 0.5]).T
            bottom = [1638.5, 1595.5, 1569.5, 1503.5, 1449.5]
            fig, axs = plt.subplots(1, 5, figsize=(10, 5), constrained_layout=True)
            for index in range(len(g1)):
                axs[index].bar(1 - width, g1[index] - bottom[index], width=0.3, bottom=bottom[index],
                               yerr=error_d[index], ecolor="black", alpha=0.6, color='g', edgecolor="blue",
                               linewidth=0.1)
                axs[index].bar(1, g2[index] - bottom[index], width=0.3, bottom=bottom[index], yerr=error_p[index],
                               ecolor="black", alpha=0.6, color='r', edgecolor="blue", linewidth=0.1)
                axs[index].bar(1 + width, g3[index] - bottom[index], width=0.3, bottom=bottom[index],
                               yerr=error_p[index], ecolor="black", alpha=0.6, color='b', edgecolor="blue",
                               linewidth=0.1)
                axs[index].set_title(cat_par[index])
            plt.show()
        elif command == 5:
            donor, waves_d, patient, waves_p, non_secreting, waves_n = sorting_ratio_and_waves_by_names()
            result_d, result_waves_d, result_p, result_waves_p, result_n, result_waves_n, normal = calculate_ratio()
            error_radial = [0.5, 0.4, 0.001, 0.07, 0.01, 0.001, 0.001, 0.2, 0.03, 0.001]
            for i in range(len(error_radial)):
                error_radial[i] = error_radial[i] * normal[i]
            print('Прогрешность:', error_radial)
            all_secr_numb = []
            all_nesecr_numb = []
            for index in range(len(filenames)):
                if (filenames[index][0] == 'P') or (filenames[index][0] == 'M'):
                    all_secr_numb.append(filenames[index][1:3])
                if filenames[index][0] == 'N':
                    all_nesecr_numb.append(filenames[index][1:3])
            print('Номера обработанных секретирующих:\n', all_secr_numb, '\n',
                  'Номера обработанных несекретирующих:\n', all_nesecr_numb)
            int_pat = input('Введите номера пациентов через запятую:')
            int_pat = int_pat.replace(' ', '')
            pat_numb_temp = int_pat.split(',')
            pat_numb = []
            secr_intensities = []
            nesecr_intensities = []
            secr_waves = []
            nesecr_waves = []
            print(patient)
            print(non_secreting)
            for i in pat_numb_temp:
                pat_numb.append(int(i))
            for index in range(len(pat_numb)):
                pat_name_secr = []
                pat_name_nesecr = []
                for file in filenames:
                    if (file[0] == 'P') or (file[0] == 'M'):
                        pat_name_secr.append(file)
                    if file[0] == 'N':
                        pat_name_nesecr.append(file)
                for index_2 in range(len(pat_name_secr)):
                    if int(pat_name_secr[index_2][1:3]) == int(pat_numb[index]):
                        secr_intensities.append(patient[index_2])
                        secr_waves.append(waves_p[index_2])
                for index_3 in range(len(pat_name_nesecr)):
                    if int(pat_name_nesecr[index_3][1:3]) == int(pat_numb[index]):
                        nesecr_intensities.append(non_secreting[index_3])
                        nesecr_waves.append(waves_n[index_3])
            for n in secr_intensities:
                for i in range(len(n)):
                    n[i] = n[i] * normal[i]
            for n in nesecr_intensities:
                for i in range(len(n)):
                    n[i] = n[i] * normal[i]

            labels = ['A1/M1', 'A1/A2', 'A1/M2', 'A1/A3', 'M1/A2', 'M1/M2', 'M1/A3', 'A2/M2', 'A2/A3', 'M2/A3']
            theta = np.linspace(start=0, stop=2 * np.pi, num=len(result_d), endpoint=False)
            theta = np.concatenate((theta, [theta[0]]))
            result_d = np.append(result_d, result_d[0])
            for g in secr_intensities:
                g.append(g[0])
            for g in nesecr_intensities:
                g.append(g[0])
            error_radial = np.append(error_radial, error_radial[0])
            fig = plt.figure(figsize=(10, 10), facecolor='#f3f3f3')
            ax = fig.add_subplot(111, projection='polar')
            for sample in secr_intensities:
                ax.errorbar(theta, sample, linewidth=2, xerr=0, yerr=0, color="red")
            ax.errorbar(theta, result_d, linewidth=2, xerr=0, yerr=error_radial, color="green", ecolor='black')
            for sample in nesecr_intensities:
                ax.errorbar(theta, sample, linewidth=2, xerr=0, yerr=0, color="blue")
            ax.set_thetagrids(range(0, 360, int(360 / len(labels))), labels)
            plt.yticks(np.arange(0, 1.5, 0.2), fontsize=8)
            ax.set(facecolor='#f3f3f3')
            ax.set_theta_offset(np.pi / 2)

            pl = ax.yaxis.get_gridlines()
            for line in pl:
                line.get_path()._interpolation_steps = 5

            cat_par = ['Amide-I', 'Min 1-2', 'Amide-II', 'Min 2-3', 'Amide-III']
            g1 = result_waves_d
            g2 = secr_waves
            g3 = nesecr_waves
            width = 0.3

            error_d = np.array([0.89, 0.364, 0.625, 0.483, 0.246]).T
            error_p = np.array([0.1, 0.3, 0.2, 0.4, 0.5]).T
            bottom = [1638.5, 1595.5, 1569.5, 1503.5, 1449.5]
            fig, axs = plt.subplots(1, 5, figsize=(10, 5), constrained_layout=True)
            for index in range(len(g1)):
                axs[index].bar(1 - width, g1[index] - bottom[index], width=0.3, bottom=bottom[index],
                               yerr=error_d[index], ecolor="black", alpha=0.6, color='g', edgecolor="blue",
                               linewidth=0.1)
                for sample in g2:
                    axs[index].bar(1 + width*int(g2.index(sample)), sample[index] - bottom[index], width=0.3, bottom=bottom[index], yerr=error_p[index],
                               ecolor="black", alpha=0.6, color='r', edgecolor="blue", linewidth=0.1)
                for sample in g3:
                    axs[index].bar(1 + width*(len(g2) + int(g3.index(sample))), sample[index] - bottom[index], width=0.3, bottom=bottom[index],
                               yerr=error_p[index], ecolor="black", alpha=0.6, color='b', edgecolor="blue",
                               linewidth=0.1)
                axs[index].set_title(cat_par[index])

            plt.show()

        elif command == 6:
            exit(0)
        else:
            print('[PCA]: некорректный ввод')


def write_eigenvalues_and_eigenvectors_in_files(research_name):
    eigenvalues_c_dataframe = pd.DataFrame(eigenvalues_c)
    eigenvectors_c_dataframe = pd.DataFrame(eigenvectors_c)
    eigenvalues_b_dataframe = pd.DataFrame(eigenvalues_b)
    eigenvectors_b_dataframe = pd.DataFrame(eigenvectors_b)
    filenames_dataframe = pd.DataFrame(filenames)
    current_datetime = str(datetime.now()).replace(':', '-')
    if research_name != '':
        files_directory = 'output_csv\\{} {}'.format(current_datetime, research_name)
    else:
        files_directory = 'output_csv\\{}'.format(current_datetime)
    os.mkdir(files_directory)
    eigenvalues_c_dataframe.to_csv(files_directory + '\\eigenvalues_c.csv'.format(current_datetime), index=False, header=None)
    eigenvectors_c_dataframe.to_csv(files_directory + '\\eigenvectors_c.csv'.format(current_datetime), index=False, header=None)
    eigenvalues_b_dataframe.to_csv(files_directory + '\\eigenvalues_b.csv'.format(current_datetime), index=False, header=None)
    eigenvectors_b_dataframe.to_csv(files_directory + '\\eigenvectors_b.csv'.format(current_datetime), index=False, header=None)
    filenames_dataframe.to_csv(files_directory + '\\filenames.csv'.format(current_datetime), index=False, header=None)
    print('[PCA]: исследование {} {} сохранено'.format(current_datetime, research_name))


def read_eigenvalues_and_eigenvectors_from_files():
    eigenvalues_c_tmp = pd.read_csv('input_csv\\eigenvalues_c.csv', header=None).values.tolist()
    eigenvectors_c_tmp = pd.read_csv('input_csv\\eigenvectors_c.csv', header=None).values.tolist()
    eigenvalues_b_tmp = pd.read_csv('input_csv\\eigenvalues_b.csv', header=None).values.tolist()
    eigenvectors_b_tmp = pd.read_csv('input_csv\\eigenvectors_b.csv', header=None).values.tolist()
    filenames_tmp = pd.read_csv('input_csv\\filenames.csv', header=None).values.tolist()
    for element in eigenvalues_c_tmp:
        eigenvalues_c.append(element[0])
    for element in eigenvalues_b_tmp:
        eigenvalues_b.append(element[0])
    for element in eigenvectors_c_tmp:
        eigenvectors_c.append(element)
    for element in eigenvectors_b_tmp:
        eigenvectors_b.append(element)
    for element in filenames_tmp:
        filenames.append(element[0])
    print('[PCA]: сохраненное исследование прочитано')


def main_menu():
    command = int(input('***** PCA (Python3 + R) v4.0 от dzagalskij X daniil432 *****\n' +
                        '***** ВНИМАНИЕ! *****\nПроверьте правильность R_BIN_PATH: {}\n'.format(R_BIN_PATH) +
                        '[PCA]: выберите источник данных:\n' +
                        '[PCA]: прочитать новый спектр (из input_dpt) (1)\n' +
                        '[PCA]: прочитать сохраненные собственные числа и вектора (из input_csv) (2)\n' +
                        '[PCA]: '))
    if command == 1:
        research_name = input('[PCA]: новое исследование\n' +
                              '[PCA]: введите название исследования: ')
        read_files()
        calculate_and_sort_eigenvalues_and_vectors()
        write_eigenvalues_and_eigenvectors_in_files(research_name)
    elif command == 2:
        read_eigenvalues_and_eigenvectors_from_files()
    else:
        print('[PCA]: некорректный ввод')
        exit(0)
    show_graphic_of_eigenvalues_and_pc()
    show_graphic_of_t_matrix()


if __name__ == '__main__':
    main_menu()