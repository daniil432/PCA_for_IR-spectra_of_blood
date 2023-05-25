import re
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class SpectraReader(object):
    def __init__(self):
        self.spectra_matrix = None

    def read_spectra(self, path_dpt=f'{os.curdir}\\input_dpt'):
        self.spectra_matrix = pd.DataFrame()
        filenames = []
        os.chdir(path_dpt)
        path_input = glob.glob("*.dpt")
        for file_path in path_input:
            input_df = pd.read_csv(file_path, header=None)
            if path_input.index(file_path) == 0:
                self.spectra_matrix['waves'] = input_df[0]
            filename = file_path.replace('input_dpt\\', '')
            filename = re.match('[a-zA-Z-0-9]+', filename).group()
            self.spectra_matrix[filename] = input_df[1]
            filenames.append(filename if isinstance(filename, str) else filename[0])
        os.chdir('../')
        return filenames

    def cut_spectra(self, separate_df, input_range='100000-0'):
        if separate_df is True:
            cutted_matrix = []
        else:
            cutted_matrix = pd.DataFrame(columns=self.spectra_matrix.columns)
        input_range = input_range.replace(' ', '')
        input_range = [list(map(int, input_range.split(',')[i].split('-'))) for i in range(len(input_range.split(',')))]
        for wave in input_range:
            temp = self.spectra_matrix.drop(self.spectra_matrix[self.spectra_matrix['waves'] < min(wave)].index)
            temp = temp.drop(temp[temp['waves'] > max(wave)].index)
            if separate_df is True:
                cutted_matrix.append(temp.reset_index(drop=True))
            else:
                cutted_matrix = pd.concat([cutted_matrix, temp]).reset_index(drop=True)
        return cutted_matrix


class AverageAnal(object):
    def __init__(self, filenames):
        self.filenames = filenames

    def get_waves_ratios(self, df_ranges):
        ratios = []
        prepared = []
        waves = []
        maxmin = []
        # Проверяем, что нужно искать в вырезанных фрагментах - максимумы или минимумы.
        for ind in range(len(df_ranges)):
            # Если для вырезанного фрагмента значения поглощения на краях больше, чем в середине среза - минимум.
            size = df_ranges[ind][self.filenames[0]].size
            if df_ranges[ind][self.filenames[0]].iloc[0] > df_ranges[ind][self.filenames[0]].iloc[size//2]:
                maxmin.append(False)
            else:
                maxmin.append(True)

        for name in self.filenames:
            temp_r = []
            temp_w = []
            for cut in range(len(df_ranges)):
                if maxmin[cut]:
                    temp_r.append(df_ranges[cut][name].max())
                    temp_w.append(df_ranges[cut]['waves'][df_ranges[cut][name].idxmax()])
                else:
                    temp_r.append(df_ranges[cut][name].min())
                    temp_w.append(df_ranges[cut]['waves'][df_ranges[cut][name].idxmin()])
            prepared.append(temp_r)
            waves.append(temp_w)

        order = [1, 4, 3, 6, 5, 7, 2]
        for sample in range(len(prepared)):
            temp = []
            for i in range(len(prepared[sample])):
                for j in range(i + 1, len(prepared[sample])):
                    if order[i] < order[j]:
                        temp.append(prepared[sample][i]/prepared[sample][j])
                    else:
                        temp.append(prepared[sample][j] / prepared[sample][i])
            ratios.append(temp)
        return ratios, waves

    def calc_average(self, ratio, waves):
        ratio_waves = dict()
        for name in range(len(self.filenames)):
            if self.filenames[name][0] in ratio_waves.keys():
                ratio_waves[self.filenames[name][0]][0].append(ratio[name])
                ratio_waves[self.filenames[name][0]][1].append(waves[name])
            else:
                ratio_waves[self.filenames[name][0]] = [[ratio[name]], [waves[name]]]

        # result_dict = {'D': list(list(ratio), list(waves), list(errors_r), list(errors_w)), 'M': list(...), ...}
        result_dict = dict()
        for key in ratio_waves.keys():
            result_dict[key] = [[], [], [], []]
        for key in ratio_waves.keys():
            for component in range(len(ratio_waves[key])):
                result_temp = [0 for _ in range(len(ratio_waves[key][component][0]))]
                for elem in ratio_waves[key][component]:
                    result_temp = [x + y for x, y in zip(result_temp, elem)]
                for val in range(len(result_temp)):
                    if len(result_temp) != 0:
                        result_temp[val] = result_temp[val] / len(ratio_waves[key][component])
                result_dict[key][component] = result_temp
                result_dict[key][component + 2] = create_errors_for_values([list(i) for i in zip(*ratio_waves[key][component])])
        return result_dict

    def normalize_average(self, average, ratio, norm_key='D'):
        for ind in range(len(average[norm_key][0])):
            for key in average.keys():
                if key != norm_key:
                    average[key][2][ind] = average[key][2][ind] * (1/average[norm_key][0][ind])
                    average[key][0][ind] = average[key][0][ind] * (1/average[norm_key][0][ind])
            for sample in range(len(ratio)):
                ratio[sample][ind] = ratio[sample][ind] * (1/average[norm_key][0][ind])
            average[norm_key][2][ind] = average[norm_key][2][ind] * (1 / average[norm_key][0][ind])
            average[norm_key][0][ind] = average[norm_key][0][ind] * (1 / average[norm_key][0][ind])
        return average, ratio


class PcaAnal(object):
    def __init__(self, matrix, drop_first=True):
        self.matrix_orig = matrix
        if drop_first is True:
            self.matrix = matrix.drop(matrix.columns[0], axis=1).T
        else:
            self.matrix = matrix

    def graph_single(self, save):
        fig = plt.figure(dpi=600)
        plt.gca().invert_xaxis()
        data = [self.matrix_orig.loc[:, self.matrix_orig.columns[0]],
                self.matrix_orig.loc[:, self.matrix_orig.columns[7]]]
        plt.plot(data[0], data[1], linewidth=2, color='black', linestyle='solid')
        plt.xlabel('Wavenumber, cm$^{-1}$', fontsize=11)
        plt.ylabel('Absorbance', fontsize=11)
        plt.text(1615, 0.8, 'Amide-I\nStretching vibrations C=O')
        plt.text(1600, 0.45, 'Amide-II bending\nvibrations N-H\nSide chains vibrations')
        plt.text(1525, 0.2, 'Tyr')
        plt.text(1440, 0.49, "Amide-II' bending\nvibrations N-D")
        #fig.show()
        if save is True:
            fig.savefig(f'fig_1_en.tiff')
            fig.savefig(f'fig_1_en.eps')

    def graph_many(self, save, filenames):
        fig = plt.figure(dpi=600)
        # M17, N7
        i1 = 39  # M17
        # i1 = 33  # M3
        # i1 = 33  # M3
        i2 = 52  # N7
        plt.gca().invert_xaxis()
        print(filenames)
        print(filenames[1], filenames[5], filenames[i1-1], filenames[i2-1])
        data = [self.matrix_orig.loc[:, self.matrix_orig.columns[0]], self.matrix_orig.loc[:, self.matrix_orig.columns[4]],
                self.matrix_orig.loc[:, self.matrix_orig.columns[i1]], self.matrix_orig.loc[:, self.matrix_orig.columns[i2]]]
        plt.plot(data[0], data[1], linewidth=2, color='green', linestyle='solid')
        plt.plot(data[0], data[2], linewidth=2, color='red', linestyle='dashed')
        plt.plot(data[0], data[3], linewidth=2, color='blue', linestyle='-.')
        plt.legend(['Здоровые доноры', 'Пациенты с секретирующей ММ', 'Пациенты с не секретирующей ММ'], fancybox=True,
                   framealpha=1, shadow=True)  # Russian ver.
        # plt.legend(['Healthy donors', 'Secretory MM patients', 'Non secretory MM patients'], fancybox=True,
        #            framealpha=1, shadow=True)  # English ver.
        plt.xlabel('Волновое число, см$^{-1}$', fontsize=11)  # Russian ver.
        # plt.xlabel('Wavenumber, cm$^{-1}$', fontsize=11)  # English ver.
        plt.ylabel('Поглощение', fontsize=11)  # Russian ver.
        # plt.ylabel('Absorbance', fontsize=11)  # English ver.
        plt.text(1630, 1.17, 'M$_{I}$')
        plt.text(1440, 0.63, 'M$_{II}$')
        plt.text(1570, 0.38, 'M$_{S}$')
        plt.text(1520, 0.18, 'M$_{T}$')
        plt.text(1600, 0.12, 'N$_{1}$')
        plt.text(1530, 0.01, 'N$_{2}$')
        plt.text(1510, 0.00, 'N$_{3}$')
        #fig.show()
        if save is True:
            fig.savefig(f'fig_2_en.tiff')
            fig.savefig(f'fig_2_en.eps')

    def eigen_graph(self, save=False, name=''):
        eigenvalues = self.eigenvalues
        eigenvalues = np.insert(eigenvalues, 0, np.sum(eigenvalues))
        x = [i for i in range(len(eigenvalues))]
        y = eigenvalues
        plt.plot(x, y, color='black', linestyle='solid', linewidth=2, marker='o', markerfacecolor='red', markersize=8)
        plt.xlabel('Номер собственного значения')
        plt.ylabel('Собственное значение')
        plt.title('График собственных значений')
        plt.show()
        if save is True:
            plt.savefig(f'eigenplot_{name}.png')
        plt.close()

    def normalize(self):
        self.matrix = (self.matrix - self.matrix.mean()) / self.matrix.std()
        return self.matrix

    def performPCA(self):
        c_matrix = self.matrix @ self.matrix.T
        b_matrix = self.matrix.T @ self.matrix
        eigenval_c, eigenvec_c = np.linalg.eig(c_matrix)
        eigenval_b, eigenvec_b = np.linalg.eig(b_matrix)
        eigenval_c, eigenval_b = eigenval_c.real, eigenval_b.real
        eigenvec_c, eigenvec_b = eigenvec_c.real, eigenvec_b.real

        # строим матрицу S, изначально заполняем нулями
        s_matrix = np.zeros((len(eigenval_c), len(eigenval_b)))

        # какие-то числа совпадают, смотрим, где чисел меньше
        if len(eigenval_c) > len(eigenval_b):
            self.eigenvalues = eigenval_b
        else:
            self.eigenvalues = eigenval_c

        # на диагональ матрицы S ставим собственные числа
        for index in range(len(self.eigenvalues)):
            s_matrix[index][index] = (abs(self.eigenvalues[index])) ** (1 / 2)

        # находим матрицы T и P
        t_matrix = eigenvec_c @ s_matrix
        p_matrix = eigenvec_b
        # Способ через метод PCA пакета sklearn
        # pca = PCA(n_components=5)
        # t_matrix = pca.fit_transform(self.matrix)
        # p_matrix = pca.components_.T
        return t_matrix, p_matrix

    def heatmap_pca(self, matrix, slice=-1, save=False, name=''):
        sns.heatmap(matrix[:, :slice])
        plt.show()
        if save is True:
            plt.savefig(f'heatmap_{name}.png')
        plt.close()
        table = pd.DataFrame(matrix[:, :slice])
        sns.pairplot(table)
        plt.show()
        if save is True:
            plt.savefig(f'pairplot_{name}.png')
        plt.close()


def derivative_df(matrix):
    matrix = matrix.T.values.tolist()
    all_derivatives = []
    all_derivatives.append(matrix[0])
    for sample in range(len(matrix)):
        sample_derivative = []
        if sample == 0:
            pass
        else:
            for point in range(len(matrix[1])):
                if point == 0:
                    pass
                elif point == len(matrix[1]) - 1:
                    pass
                elif (point != 0) and (point != len(matrix[1]) - 1):
                    deriv = 0.5 * (((matrix[sample][point + 1] - matrix[sample][point]) /
                                    (matrix[0][point + 1] - matrix[0][point])) +
                                   ((matrix[sample][point] - matrix[sample][point - 1]) /
                                    (matrix[0][point] - matrix[0][point - 1])))
                    sample_derivative.append(deriv)
            all_derivatives.append(sample_derivative)
    all_derivatives[0].pop(0)
    all_derivatives[0].pop(len(all_derivatives[0]) - 1)
    all_derivatives = pd.DataFrame(all_derivatives).T
    return all_derivatives


def create_errors_for_values(matrix):
    res = []
    for vec in matrix:
        err = mean_error(vec)
        res.append(err)
    return res


def mean_error(vector):
    try:
        vector = np.array(vector)
    except:
        pass
    N = len(vector)
    M = np.sum(vector)/N
    D = 0
    for R in vector:
        D += (R-M)**2
    error = np.sqrt(D/N/(N-1))
    return error


if __name__ == "__main__":
    s = SpectraReader()
    filenames = s.read_spectra('input_dpt')
    matrix_for_average = s.cut_spectra('y', '1600-1700, 1580-1620, 1550-1590, 1520-1550, '
                                            '1500-1525, 1497-1512, 1420-1480')
    matrix_for_pca = s.cut_spectra('n', '1700-1350')

    a = AverageAnal(filenames)
    ratio, waves = a.get_waves_ratios(matrix_for_average)
    ratio_waves = a.calc_average(ratio, waves)
    ratio_waves, ratio_norm = a.normalize_average(ratio_waves, ratio)
    ratio_norm = pd.DataFrame(ratio_norm)

    p = PcaAnal(matrix_for_pca)
    p.normalize()
    p.graph_single(save=False)
    t_matrix, p_matrix = p.performPCA()
    #p.heatmap_pca(t_matrix, 10, save=False, name='orig')

    pr = PcaAnal(ratio_norm, drop_first=False)
    tr_matrix, pr_matrix = pr.performPCA()
    pr.heatmap_pca(tr_matrix, 10, save=False, name='ratio_pca')

    #d = derivative_df(matrix_for_pca)
