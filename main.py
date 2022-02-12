import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from subprocess import run
import copy
# from scipy.optimize import curve_fit


R_BIN_PATH = 'C:\\Program Files\\R\\R-4.0.4\\bin'


class Spectra_Anal:
    def read_files(self, input_data, normalization, path_dpt):

        self.input_matrix = []
        self.filenames = []
        self.input_df_list = []

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
        os.chdir(path_dpt)
        path_input = glob.glob("*.dpt")
        for file_path in path_input:
            # print(glob.glob("*.dpt"))
            input_df = pd.read_csv(file_path, header=None)
            self.input_df_list.append(input_df)
            filename = file_path.replace('input_dpt\\', '')
            filename = filename.replace('.dpt', '')
            filename = filename.replace('-', '')
            filename = filename[0:3]
            filename = filename.replace('_0', '')
            filename = filename.replace('_', '')
            self.filenames.append(filename)
            one_pat = []
            for i in range(len(input_waves)):
                input_df_temp = input_df.drop(input_df[input_df[0] < input_waves[i][0]].index)
                input_df_temp = input_df_temp.drop(input_df_temp[input_df_temp[0] > input_waves[i][1]].index)
                one_pat = np.concatenate((one_pat, input_df_temp[1].values.astype('float')))
                # one_wave = np.concatenate((one_wave, input_df_temp[0].values.astype('float')))
            self.input_matrix.append(one_pat)
        self.one_wave = []
        for i in range(len(input_waves)):
            input_df = pd.read_csv(path_input[0], header=None)
            input_df_temp = input_df.drop(input_df[input_df[0] < input_waves[i][0]].index)
            input_df_temp = input_df_temp.drop(input_df_temp[input_df_temp[0] > input_waves[i][1]].index)
            self.one_wave = np.concatenate((self.one_wave, input_df_temp[0].values.astype('float')))
        print('[PCA]: файлы прочитаны')
        self.all_samples_for_derivative = copy.deepcopy(self.input_matrix)
        self.all_samples_for_derivative.insert(0, self.one_wave)
        if normalization == 'Y' or normalization == 'y':
            for index in range(len(self.input_matrix[0])):
                column = [row[index] for row in self.input_matrix]
                m = np.mean(column)
                s = np.std(column)
                normalization_function = lambda x: (x - m) / s
                normalized_column = normalization_function(column)
                for str_index in range(len(self.input_matrix)):
                    self.input_matrix[str_index][index] = normalized_column[str_index]
            print('[PCA]: нормализация произведена')
        else:
            print('[PCA]: нормализация пропущена')


    def cutting_spectra_and_finding_ratio(self):
        diapason = [[1600, 1650], [1580, 1600], [1550, 1580], [1500, 1510], [1440, 1470]]
        baseline = [1720, 2000]
        # range_for_derivative = [1597, 1701]
        max_min = []
        self.ratio = []
        self.waves = []

        """wfd = self.input_df_list[0].drop(self.input_df_list[0][self.input_df_list[0][0] <= range_for_derivative[0]].index)
        wfd = wfd.drop(wfd[wfd[0] >= range_for_derivative[1]].index)
        wfd = wfd[0].values.astype('float')
        self.all_samples_for_derivative.append(wfd)"""

        for input_df in self.input_df_list:
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
            """rfd = input_df.drop(input_df[input_df[0] <= range_for_derivative[0]].index)
            rfd = rfd.drop(rfd[rfd[0] >= range_for_derivative[1]].index)
            afd = rfd[1].values.astype('float')

            self.all_samples_for_deivative.append(afd)"""

            # находим макс. и мин., вычитаем из них среднее по базовой линии
            for ind in range(5):
                max_min_temp = []
                waves_temp = []
                max_min.append(max_min_temp)
                self.waves.append(waves_temp)
                max_min_temp.append(max(intensities[ind]) - coefficient)
                waves_temp.append(waves_in_diapason[ind][intensities[ind].index(max(intensities[ind]))])
            """max_min_temp.append(min(intensities[1]) - coefficient)
            waves_temp.append(waves_in_diapason[1][intensities[1].index(min(intensities[1]))])
            max_min_temp.append(max(intensities[2]) - coefficient)
            waves_temp.append(waves_in_diapason[2][intensities[2].index(max(intensities[2]))])
            max_min_temp.append(min(intensities[3]) - coefficient)
            waves_temp.append(waves_in_diapason[3][intensities[3].index(min(intensities[3]))])
            max_min_temp.append(max(intensities[4]) - coefficient)
            waves_temp.append(waves_in_diapason[4][intensities[4].index(max(intensities[4]))])"""

            # находим отношения максимумов и минимумов попарно
            ratio_temp = []
            for i in range(len(max_min_temp)):
                for j in range(i + 1, len(max_min_temp)):
                    ratio_temp.append(max_min_temp[i]/max_min_temp[j])
            self.ratio.append(ratio_temp)


    def sorting_ratio_and_waves_by_names(self):
        # сортируем все интенсивности из фаилов по донорам, секретирующим и несекретирующим пациентам
        self.donor = []
        self.waves_d = []
        self.patient = []
        self.waves_p = []
        self.non_secreting = []
        self.waves_n = []
        for i in range(len(self.filenames)):
            if (self.filenames[i][0] == 'D') or (self.filenames[i][0] == 'O') or (self.filenames[i][0] == 'B') or \
                    (self.filenames[i][0] == 'H'):
                self.donor.append(self.ratio[i])
                self.waves_d.append(self.waves[i])
            elif (self.filenames[i][0] == 'P') or (self.filenames[i][0] == 'M'):
                self.patient.append(self.ratio[i])
                self.waves_p.append(self.waves[i])
            elif self.filenames[i][0] == 'N':
                self.non_secreting.append(self.ratio[i])
                self.waves_n.append(self.waves[i])
        # return donor, waves_d, patient, waves_p, non_secreting, waves_n


    def calculate_ratio(self):
        # donor, waves_d, patient, waves_p, non_secreting, waves_n = self.sorting_ratio_and_waves_by_names()
        # находим среднее по каждой группе образцов
        self.normal = [0.27451837331978024, 0.35147226885800564, 0.08820076135136147, 0.5330751953591907,
                       1.2822510795668887, 0.3226583936939652, 1.9167901877339504, 0.2517124655425862,
                       1.4916283337223264, 5.450819881670117]
        self.result_d = [0] * len(self.ratio[0])
        for i in self.donor:
            self.result_d = [x + y for x, y in zip(self.result_d, i)]
        for i in range(len(self.result_d)):
            if len(self.donor) != 0:
                self.result_d[i] = self.result_d[i]/len(self.donor)
                # self.normal.append(1 / self.result_d[i])
                self.result_d[i] = self.result_d[i]*(1/self.result_d[i])

        self.result_waves_d = [0] * len(self.waves[0])
        for i in self.waves_d:
            self.result_waves_d = [x + y for x, y in zip(self.result_waves_d, i)]
        for i in range(len(self.result_waves_d)):
            if len(self.waves_d) != 0:
                self.result_waves_d[i] = self.result_waves_d[i] / len(self.waves_d)

        self.result_p = [0] * len(self.ratio[0])
        for i in self.patient:
            self.result_p = [x + y for x, y in zip(self.result_p, i)]
        for i in range(len(self.result_p)):
            if len(self.patient) != 0:
                self.result_p[i] = (self.result_p[i]/len(self.patient)) * self.normal[i]

        self.result_waves_p = [0] * len(self.waves[0])
        for i in self.waves_p:
            self.result_waves_p = [x + y for x, y in zip(self.result_waves_p, i)]
        for i in range(len(self.result_waves_p)):
            if len(self.waves_p) != 0:
                self.result_waves_p[i] = self.result_waves_p[i] / len(self.waves_p)

        self.result_n = [0] * len(self.ratio[0])
        for i in self.non_secreting:
            self.result_n = [x + y for x, y in zip(self.result_n, i)]
        for i in range(len(self.result_n)):
            if len(self.non_secreting) != 0:
                self.result_n[i] = (self.result_n[i] / len(self.non_secreting)) * self.normal[i]

        self.result_waves_n = [0] * len(self.waves[0])
        for i in self.waves_n:
            self.result_waves_n = [x + y for x, y in zip(self.result_waves_n, i)]
        for i in range(len(self.result_waves_n)):
            if len(self.waves_n) != 0:
                self.result_waves_n[i] = self.result_waves_n[i] / len(self.waves_n)
        # print(self.normal)

    def calculate_and_sort_eigenvalues_and_vectors(self, input_data):
        # вычисляем транспонированную матрицу от входной
        x_matrix = np.array(input_data)
        x_matrix_transpose = x_matrix.transpose()

        # перемножаем, x_matrix * x_matrix_transpose = c_matrix
        # перемножаем, x_matrix_transpose * x_matrix = b_matrix
        c_matrix = x_matrix @ x_matrix_transpose
        b_matrix = x_matrix_transpose @ x_matrix

        # записать c_matrix и b_matrix в файл
        c_matrix_dataframe = pd.DataFrame(c_matrix)
        b_matrix_dataframe = pd.DataFrame(b_matrix)
        curpath = os.path.abspath(os.curdir)
        os.chdir('C:\PCA_with_R')
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

        self.eigenvectors_c = []
        self.eigenvalues_c = []
        self.eigenvectors_b = []
        self.eigenvalues_b = []

        for value in eigenvalues_c_tmp:
            self.eigenvalues_c.append(value[0])
        for value in eigenvalues_b_tmp:
            self.eigenvalues_b.append(value[0])
        for vector in eigenvectors_c_tmp:
            self.eigenvectors_c.append(vector)
        for vector in eigenvectors_b_tmp:
            self.eigenvectors_b.append(vector)
        os.chdir(curpath)


    def calculate_t_and_p_matrix(self):
        u_values, u_vectors, v_values, v_vectors = self.eigenvalues_c, self.eigenvectors_c, self.eigenvalues_b, \
                                                   self.eigenvectors_b
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


    def show_graphic_of_eigenvalues_and_pc(self):
        u_values, u_vectors, v_values, v_vectors = self.eigenvalues_c, self.eigenvectors_c, self.eigenvalues_b, \
                                                   self.eigenvectors_b

        # какие-то числа совпадают, смотрим, где чисел меньше
        if len(u_values) > len(v_values):
            eigenvalues = v_values
        else:
            eigenvalues = u_values
        eigenvalues_plot = np.insert(eigenvalues, 0, np.sum(eigenvalues))

        x = [i for i in range(len(eigenvalues_plot))]
        y = eigenvalues_plot
        plt.plot(x, y, color='black', linestyle='solid', linewidth=2, marker='o', markerfacecolor='red', markersize=8)
        plt.xlabel('номер значения')
        plt.ylabel('значение')
        plt.title('собственные значения')
        print('[PCA]: вывожу график собственных значений...')
        plt.show()

        I = len(self.eigenvectors_b[0])
        J = len(self.eigenvectors_c[0])
        lambda_0 = sum(eigenvalues)
        TRV = []
        for i in range(0, len(eigenvalues)):
            TRV.append((1 / (I * J)) * (lambda_0 - sum(eigenvalues[0:i])))
        ERV = []
        for trv in TRV:
            ERV.append(1 - (trv / ((1 / (I * J)) * lambda_0)))

        x = [i for i in range(len(eigenvalues))]
        y = TRV
        plt.plot(x, y, color='red', linestyle='solid', linewidth=2, marker='o', markerfacecolor='black', markersize=8)
        y = ERV
        plt.plot(x, y, color='green', linestyle='solid', linewidth=2, marker='o', markerfacecolor='black', markersize=8)
        plt.xlabel('номер значения')
        plt.ylabel('TRV, ERV')
        plt.title('главные компоненты\nTRV - красный\nERV - зеленый')
        print('[PCA]: вывожу график главных компонент...')
        plt.show()


    def show_patient_graph(self, Patients):
        error_radial = [0.5, 0.4, 0.001, 0.07, 0.01, 0.001, 0.001, 0.2, 0.03, 0.001]
        for i in range(len(error_radial)):
            error_radial[i] = error_radial[i] * self.normal[i]

        all_secr_numb = []
        all_nesecr_numb = []
        pat_name_secr = []
        pat_name_nesecr = []
        for index in range(len(self.filenames)):
            if (self.filenames[index][0] == 'P') or (self.filenames[index][0] == 'M'):
                all_secr_numb.append(self.filenames[index][1:3])
                pat_name_secr.append(self.filenames[index])
            if self.filenames[index][0] == 'N':
                all_nesecr_numb.append(self.filenames[index][1:3])
                pat_name_nesecr.append(self.filenames[index])

        secr_intensities = []
        nesecr_intensities = []
        secr_waves = []
        nesecr_waves = []
        copy_patient = copy.deepcopy(self.patient)
        copy_non_secreting = copy.deepcopy(self.non_secreting)
        self.copy_result_d = copy.deepcopy(self.result_d)

        for number in range(len(Patients)):
            for name_secr in range(len(all_secr_numb)):
                if int(all_secr_numb[name_secr]) == int(Patients[number]):
                    secr_intensities.append(copy_patient[name_secr])
                    secr_waves.append(self.waves_p[name_secr])
                    break

            if all_nesecr_numb != []:
                for name_nesecr in range(len(all_nesecr_numb)):
                    if int(all_nesecr_numb[name_nesecr]) == int(Patients[number]):
                        nesecr_intensities.append(copy_non_secreting[name_nesecr])
                        nesecr_waves.append(self.waves_n[name_nesecr])
                        break

        for patient in range(len(secr_intensities)):
            for index in range(len(secr_intensities[patient])):
                secr_intensities[patient][index] = secr_intensities[patient][index] * self.normal[index]
        for patient in range(len(nesecr_intensities)):
            for index in range(len(nesecr_intensities[patient])):
                nesecr_intensities[patient][index] = nesecr_intensities[patient][index] * self.normal[index]

        return secr_intensities, nesecr_intensities, error_radial, secr_waves, nesecr_waves


    def write_eigenvalues_and_eigenvectors_in_files(self, research_name, t_pca, p_pca, t_der1, p_der1, t_der2, p_der2):
        t_matrix_pca = t_pca
        p_matrix_pca = p_pca
        t_matrix_der1 = t_der1
        p_matrix_der1 = p_der1
        t_matrix_der2 = t_der2
        p_matrix_der2 = p_der2
        t_pca_dataframe = pd.DataFrame(t_matrix_pca)
        p_pca_dataframe = pd.DataFrame(p_matrix_pca)
        t_der1_dataframe = pd.DataFrame(t_matrix_der1)
        p_der1_dataframe = pd.DataFrame(p_matrix_der1)
        t_der2_dataframe = pd.DataFrame(t_matrix_der2)
        p_der2_dataframe = pd.DataFrame(p_matrix_der2)
        ratio_dataframe = pd.DataFrame(self.ratio)
        waves_dataframe = pd.DataFrame(self.waves)
        """eigenvalues_c_dataframe = pd.DataFrame(self.eigenvalues_c)
        eigenvectors_c_dataframe = pd.DataFrame(self.eigenvectors_c)
        eigenvalues_b_dataframe = pd.DataFrame(self.eigenvalues_b)
        eigenvectors_b_dataframe = pd.DataFrame(self.eigenvectors_b)"""
        filenames_dataframe = pd.DataFrame(self.filenames)
        current_datetime = str(datetime.today().strftime('%Y-%m-%d_%H-%M'))
        if research_name != '':
            files_directory = 'C:\PCA_with_R\output_csv\\{} {}'.format(current_datetime, research_name.replace(':', '-'))
        else:
            files_directory = 'C:\PCA_with_R\output_csv\\{}'.format(current_datetime)
        os.mkdir(files_directory)
        t_pca_dataframe.to_csv(files_directory + '\\t_pca.csv'.format(current_datetime), index=False,
                               header=None)
        p_pca_dataframe.to_csv(files_directory + '\\p_pca.csv'.format(current_datetime), index=False,
                               header=None)
        t_der1_dataframe.to_csv(files_directory + '\\t_der1.csv'.format(current_datetime), index=False,
                               header=None)
        p_der1_dataframe.to_csv(files_directory + '\\p_der1.csv'.format(current_datetime), index=False,
                                header=None)
        t_der2_dataframe.to_csv(files_directory + '\\t_der2.csv'.format(current_datetime), index=False,
                                header=None)
        p_der2_dataframe.to_csv(files_directory + '\\p_der2.csv'.format(current_datetime), index=False,
                                header=None)
        ratio_dataframe.to_csv(files_directory + '\\ratio.csv'.format(current_datetime), index=False,
                                       header=None)
        waves_dataframe.to_csv(files_directory + '\\waves.csv'.format(current_datetime), index=False,
                                       header=None)
        filenames_dataframe.to_csv(files_directory + '\\filenames.csv'.format(current_datetime), index=False,
                                   header=None)
        """eigenvalues_c_dataframe.to_csv(files_directory + '\\eigenvalues_c.csv'.format(current_datetime), index=False,
                                       header=None)
        eigenvectors_c_dataframe.to_csv(files_directory + '\\eigenvectors_c.csv'.format(current_datetime), index=False,
                                        header=None)
        eigenvalues_b_dataframe.to_csv(files_directory + '\\eigenvalues_b.csv'.format(current_datetime), index=False,
                                       header=None)
        eigenvectors_b_dataframe.to_csv(files_directory + '\\eigenvectors_b.csv'.format(current_datetime), index=False,
                                        header=None)"""
        print('[PCA]: исследование {} {} сохранено'.format(current_datetime, research_name))


    def read_eigenvalues_and_eigenvectors_from_files(self, path_csv):
        os.chdir(path_csv)
        self.ratio = []
        self.waves = []
        self.t_matrix_pca = []
        self.p_matrix_pca = []
        self.t_matrix_der1 = []
        self.p_matrix_der1 = []
        self.t_matrix_der2 = []
        self.p_matrix_der2 = []
        self.filenames = []
        """self.eigenvectors_c = []
        self.eigenvalues_c = []
        self.eigenvectors_b = []
        self.eigenvalues_b = []"""
        self.ratio_tmp = pd.read_csv('ratio.csv', header=None).values.tolist()
        self.waves_tmp = pd.read_csv('waves.csv', header=None).values.tolist()
        self.t_pca_tmp = pd.read_csv('t_pca.csv', header=None).values.tolist()
        self.p_pca_tmp = pd.read_csv('p_pca.csv', header=None).values.tolist()
        self.t_der1_tmp = pd.read_csv('t_der1.csv', header=None).values.tolist()
        self.p_der1_tmp = pd.read_csv('p_der1.csv', header=None).values.tolist()
        self.t_der2_tmp = pd.read_csv('t_der2.csv', header=None).values.tolist()
        self.p_der2_tmp = pd.read_csv('p_der2.csv', header=None).values.tolist()
        filenames_tmp = pd.read_csv('filenames.csv', header=None).values.tolist()
        """eigenvalues_c_tmp = pd.read_csv('input_csv\\eigenvalues_c.csv', header=None).values.tolist()
        eigenvectors_c_tmp = pd.read_csv('input_csv\\eigenvectors_c.csv', header=None).values.tolist()
        eigenvalues_b_tmp = pd.read_csv('input_csv\\eigenvalues_b.csv', header=None).values.tolist()
        eigenvectors_b_tmp = pd.read_csv('input_csv\\eigenvectors_b.csv', header=None).values.tolist()"""

        for element in self.ratio_tmp:
            self.ratio.append(element)

        for element in self.waves_tmp:
            self.waves.append(element)

        for element in self.t_pca_tmp:
            self.t_matrix_pca.append(element)
        self.t_matrix_pca = np.array(self.t_matrix_pca)

        for element in self.p_pca_tmp:
            self.p_matrix_pca.append(element)
        self.p_matrix_pca = np.array(self.p_matrix_pca)

        for element in self.t_der1_tmp:
            self.t_matrix_der1.append(element)
        self.t_matrix_der1 = np.array(self.t_matrix_der1)

        for element in self.p_der1_tmp:
            self.p_matrix_der1.append(element)
        self.p_matrix_der1 = np.array(self.p_matrix_der1)

        for element in self.t_der2_tmp:
            self.t_matrix_der2.append(element)
        self.t_matrix_der2 = np.array(self.t_matrix_der2)

        for element in self.p_der2_tmp:
            self.p_matrix_der2.append(element)
        self.p_matrix_der2 = np.array(self.p_matrix_der2)

        for element in filenames_tmp:
            self.filenames.append(element[0])
        """for element in eigenvalues_c_tmp:
            self.eigenvalues_c.append(element[0])
        for element in eigenvalues_b_tmp:
            self.eigenvalues_b.append(element[0])
        for element in eigenvectors_c_tmp:
            self.eigenvectors_c.append(element)
        for element in eigenvectors_b_tmp:
            self.eigenvectors_b.append(element)"""
        print('[PCA]: сохраненное исследование прочитано')


    def derivative_function(self, data_for_derivative):
        all_derivatives = []
        all_derivatives.append(copy.deepcopy(data_for_derivative[0].tolist()))
        for sample in range(len(data_for_derivative)):
            sample_derivative = []
            if sample == 0:
                pass
            else:
                for point in range(len(data_for_derivative[1])):
                    if point == 0:
                        pass
                        """deriv = 0.5 * ((data_for_derivative[sample][point+1] - data_for_derivative[sample][point]) /
                                       (data_for_derivative[0][point+1] - data_for_derivative[0][point]))
                        sample_derivative.append(deriv)"""
                    elif point == len(data_for_derivative[1]) - 1:
                        pass
                        """deriv = 0.5 * ((data_for_derivative[sample][point] - data_for_derivative[sample][point-1]) /
                                       (data_for_derivative[0][point] - data_for_derivative[0][point-1]))
                        sample_derivative.append(deriv)"""
                    elif (point != 0) and (point != len(data_for_derivative[1]) - 1):
                        deriv = 0.5 * (((data_for_derivative[sample][point+1] - data_for_derivative[sample][point]) /
                                        (data_for_derivative[0][point+1] - data_for_derivative[0][point])) +
                                       ((data_for_derivative[sample][point] - data_for_derivative[sample][point-1]) /
                                        (data_for_derivative[0][point] - data_for_derivative[0][point-1])))
                        sample_derivative.append(deriv)
                all_derivatives.append(np.array(sample_derivative))
        all_derivatives[0].pop(0)
        all_derivatives[0].pop(len(all_derivatives[0])-1)
        all_derivatives[0] = np.array(all_derivatives[0])
        return all_derivatives


    def derivative_saving(self, saving_data):
        saving_data_dataframe = pd.DataFrame(saving_data).transpose()
        current_datetime = str(datetime.today().strftime('%Y-%m-%d_%H-%M'))
        files_directory = 'C:\PCA_with_R\output_csv\\{}_deriv'.format(current_datetime)
        os.mkdir(files_directory)
        saving_data_dataframe.to_csv(files_directory + '\\t_der2.csv'.format(current_datetime), index=False,
                                header=None)


    """def gaussian(self, x, A, x0, sig):
        return A*np.exp(-(x-x0)**2/(2*sig**2))


    def multi_gaussian_3(self, x, *pars):
        self.offset = pars[-1]
        self.g1 = self.gaussian(x, pars[0], pars[1], pars[2])
        self.g2 = self.gaussian(x, pars[3], pars[4], pars[5])
        self.g3 = self.gaussian(x, pars[6], pars[7], pars[8])
        return self.g1 + self.g2 + self.g3 + self.offset


    def multi_gaussian_4(self, x, *pars):
        self.offset = pars[-1]
        self.g1 = self.gaussian(x, pars[0], pars[1], pars[2])
        self.g2 = self.gaussian(x, pars[3], pars[4], pars[5])
        self.g3 = self.gaussian(x, pars[6], pars[7], pars[8])
        self.g4 = self.gaussian(x, pars[9], pars[10], pars[11])
        return self.g1 + self.g2 + self.g3 + self.g4 + self.offset


    def multi_gaussian_5(self, x, *pars):
        self.offset = pars[-1]
        self.g1 = self.gaussian(x, pars[0], pars[1], pars[2])
        self.g2 = self.gaussian(x, pars[3], pars[4], pars[5])
        self.g3 = self.gaussian(x, pars[6], pars[7], pars[8])
        self.g4 = self.gaussian(x, pars[9], pars[10], pars[11])
        self.g5 = self.gaussian(x, pars[12], pars[13], pars[14])
        return self.g1 + self.g2 + self.g3 + self.g4 + self.g5 + self.offset"""


    """def fitting(self, derivative, spec_num):

        # Для полосы Амид-I находим прямую, которыую из неё вычтем, чтобы края полосы лежали на у = 0
        minab1 = min(self.all_samples_for_deivative[spec_num][:3])
        minab2 = min(self.all_samples_for_deivative[spec_num][len(self.all_samples_for_deivative[spec_num])-4:])
        for index in range(len(self.all_samples_for_deivative[1])):
            if index < 5:
                left_min_index = np.where(self.all_samples_for_deivative[spec_num] == minab1)
            if index > len(self.all_samples_for_deivative[spec_num])-5:
                right_min_index = np.where(self.all_samples_for_deivative[spec_num] == minab2)
        minwav1 = self.all_samples_for_deivative[0][left_min_index[0]]
        minwav2 = self.all_samples_for_deivative[0][right_min_index[0]]
        for index in range(len(self.all_samples_for_deivative[spec_num])):
            self.all_samples_for_deivative[spec_num][index] = self.all_samples_for_deivative[spec_num][index] - \
                                                       ((((self.all_samples_for_deivative[0][index]
                                                           - minwav1)/(minwav2 - minwav1))*(minab2 - minab1)) + minab1)
        maxab = max(self.all_samples_for_deivative[spec_num])
        # Вычли прямую, теперь находим минимумы производной и их положения, затем выделяем из них локальные
        # wavder = [[1605.1, 1615.2], [1630.2, 1643.2], [1644.1, 1660.1], [1662.5, 1678.4], [1680.3, 1692.4]]
        minder = []
        minderind = []
        wavderfirst = [15, 44, 82, 117, 175]
        wavdersecond = [44, 77, 115, 144, 199]
        locmin = []
        locind = []
        '''for i in range(len(derivative[0])):
            for num in range(len(wavder)):
                if round(derivative[0][i], 1) == wavder[num][0]:
                    wavdersecond.append(i)
                elif round(derivative[0][i], 1) == wavder[num][1]:
                    wavderfirst.append(i)
        print(wavderfirst, wavdersecond)'''
        for ind in range(len(wavderfirst)):
            for valind in range(len(derivative[spec_num])):
                if derivative[spec_num][valind] == min(derivative[spec_num][wavderfirst[ind]:wavdersecond[ind]]):
                    minder.append(min(derivative[spec_num][wavderfirst[ind]:wavdersecond[ind]]))
                    minderind.append(valind)
        print(minder, minderind)
        # нашли все минмумы, ищем локальные
        for i in range(len(minderind)):
            if derivative[spec_num][minderind[i]] < derivative[spec_num][minderind[i]+3] \
                    and derivative[spec_num][minderind[i]] < derivative[spec_num][minderind[i]+3]:
                locmin.append(derivative[spec_num][minderind[i]])
                locind.append(minderind[i])
            else:
                pass

        # находим разложение полосы и строим его
        x, y = self.all_samples_for_deivative[0], self.all_samples_for_deivative[spec_num]
        # предположим примерную форму и положение полос

        if abs(minder[1]) > abs(minder[2]):
            alpha_guess = 0.75 * maxab
            betha_guess = 1.15 * alpha_guess
        else:
            betha_guess = 0.75 * maxab
            alpha_guess = 1.15 * betha_guess
        guess = np.array([1, derivative[0][minderind[0]], 1,
                          betha_guess, derivative[0][minderind[1]], 1,
                          alpha_guess, derivative[0][minderind[2]], 1,
                          0.4 * maxab, derivative[0][minderind[3]], 1,
                          1, derivative[0][minderind[4]], 1, 1])
        err = np.array(
            [np.inf, 8, np.inf, 0.3 * maxab, 2, np.inf, 0.3 * maxab, 2, np.inf, 0.2 * maxab, 2, np.inf, np.inf, 2, np.inf, np.inf])
        bounds = [guess - err, guess + err]
        if len(locmin) == 3:
            guess3 = guess[3:13]
            err3 = err[3:13]
            bounds3 = [guess3 - err3, guess3 + err3]
            popt, pcov = curve_fit(f=self.multi_gaussian_3, xdata=x, ydata=y, p0=guess3, bounds=bounds3, absolute_sigma=True)
        elif len(locmin) == 4:
            if locind[0] == minderind[0]:
                guess4 = guess[:13]
                err4 = err[:13]
                bounds4 = [guess4 - err4, guess4 + err4]
                popt, pcov = curve_fit(f=self.multi_gaussian_4, xdata=x, ydata=y, p0=guess4, bounds=bounds4, absolute_sigma=True)
            elif locind[3] == minderind[4]:
                guess4 = guess[3:]
                err4 = err[3:]
                bounds4 = [guess4 - err4, guess4 + err4]
                popt, pcov = curve_fit(f=self.multi_gaussian_4, xdata=x, ydata=y, p0=guess4, bounds=bounds4, absolute_sigma=True)
        elif len(locmin) == 5:
            popt, pcov = curve_fit(f=self.multi_gaussian_5, xdata=x, ydata=y, p0=guess, bounds=bounds, absolute_sigma=True)
        plt.figure()
        plt.plot(x, y, '-', linewidth=4, label='Data')
        plt.plot(x, self.multi_gaussian_3(x, *popt), 'd--', linewidth=2, label='Fit')
        plt.plot(x, self.g1, 'b--', linewidth=2, label='g1')
        plt.plot(x, self.g2, 'r--', linewidth=2, label='g2')
        plt.plot(x, self.g3, 'g--', linewidth=2, label='g3')
        if len(locmin) == 4:
            plt.plot(x, self.g4, 'y--', linewidth=2, label='g4')
        elif len(locmin) == 5:
            plt.plot(x, self.g4, 'y--', linewidth=2, label='g4')
            plt.plot(x, self.g5, 'o--', linewidth=2, label='g5')
        plt.legend()
        plt.show()"""

