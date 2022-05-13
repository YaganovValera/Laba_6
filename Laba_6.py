import numpy as np
import random
import time
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

try:
    N = int(input("Введите длину матрицы (положительное, целое число, > 3): "))
    while N <= 3:
        N = int(input("Введите длину матрицы (положительное, целое число, > 3): "))
    K = int(input("Введите число K (целое число): "))
    start = time.monotonic()
    np.set_printoptions(linewidth=1000)
    # Создание и заполнение матрицы A:
    A = np.random.randint(-10.0, 10.0, (N, N))
    print("Матрица А:\n", A)
    # Создание подматриц:
    submatrix_length = N // 2                                                       # Длина подматрицы
    sub_matrix_C = np.array(A[:submatrix_length, submatrix_length+N % 2:N])
    sub_matrix_B = np.array(A[:submatrix_length, :submatrix_length])
    sub_matrix_E = np.array(A[submatrix_length+N % 2:N, submatrix_length+N % 2:N])
    # Создание матрицы F:
    F = A.copy()
    print("\nМатрица F: \n", F)
    # Обработка матрицы С:
    count_number_in_column = np.sum(sub_matrix_C[:, 0:submatrix_length:2] > K)
    multiplication_of_numbers = sub_matrix_C[1::2].prod()
    # Формируем матрицу F:
    if count_number_in_column > multiplication_of_numbers:
        F[:submatrix_length, submatrix_length + N % 2:N] = sub_matrix_B[:submatrix_length, ::-1]
        F[:submatrix_length, :submatrix_length] = sub_matrix_C[:submatrix_length, ::-1]
    else:
        F[:submatrix_length, submatrix_length+N % 2:N] = sub_matrix_E
        F[submatrix_length+N % 2:N, submatrix_length+N % 2:N] = sub_matrix_C
    print("\nОтформатированная матрица F: \n", F)
    # Вычисляем выражение:
    try:
        if np.linalg.det(A) > sum(np.diagonal(F)):
            print("\n Результат выражения A*A^T - K*F^(-1): \n", A*A.transpose() - np.linalg.inv(F)*K)
        else:
            G = np.tri(N)*A
            print("\nРезультат выражения (A^(-1) +G-F^Т)*K:\n", (np.linalg.inv(A)+G-F.transpose())*K)
    except np.linalg.LinAlgError:
        print('Одна из матриц является вырожденной(определитель равен 0), поэтому обратную матрицу найти невозможно.')
    finish = time.monotonic()
    print("\nВремя работы программы:", finish - start, "sec.")


    print("\nМатрица, которая используется при построение графиков: \n", A)
                                                                        # Использование библиотеки matplotlib
    av = [np.mean(abs(A[i, ::])) for i in range(N)]
    av = int(sum(av))                                                   # Сумма средних значений строк (используется при создание третьего графика)
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    x = list(range(1, N+1))
    for j in range(N):
        y = list(A[j, ::])
                                                                        # №1 Программа выводит обычный график
        axs[0, 0].plot(x, y, '.-', label=f"{j} строка.")
        axs[0, 0].set(title="График с использованием функции plot:", xlabel='Номер элемента в строке', ylabel='Значение элемента')
        axs[0, 0].grid()
                                                                        # №2 Программа выводит гистограмму, по которой можно определить max и min n-ый элемент среди всех строк
        axs[0, 1].bar(x, y, 0.4, label=f"{j} строка.")
        axs[0, 1].set(title="График с использованием функции bar:", xlabel='Номер элемента в строке', ylabel='Значение элемента')
        if N <= 10:
            axs[0, 0].legend(loc='lower right')
            axs[0, 1].legend(loc='lower right')
                                                                        # №3 Программа выводит отношение средних значений от каждой строки
    explode = [0]*(N-1)
    explode.append(0.1)
    sizes = [round((np.mean(abs(A[i, ::])) * 100)/av, 1) for i in range(N)]
    axs[1, 0].set_title("График с использованием функции pie:")
    axs[1, 0].pie(sizes, labels=list(range(1, N+1)), explode=explode, autopct='%1.1f%%', shadow=True)
                                                                        # №4 Программа выводит аннотированную тепловую карту
    def heatmap(data, row_labels, col_labels, ax, cbar_kw={}, **kwargs):
        im = ax.imshow(data, **kwargs)
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
        ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)
        return im, cbar
    def annotate_heatmap(im, data = None, textcolors=("black", "white"),threshold=0):
        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()
        kw = dict(horizontalalignment="center", verticalalignment="center")
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(data[i, j] > threshold)])
                text = im.axes.text(j, i, data[i, j], **kw)
                texts.append(text)
        return texts
    im, cbar = heatmap(A, list(range(N)), list(range(N)), ax=axs[1, 1], cmap="YlGn")
    texts = annotate_heatmap(im)
    axs[1, 1].set(title="Создание аннотированных тепловых карт:", xlabel="Номер столбца", ylabel="Номер строки")
    plt.suptitle("Использование библиотеки matplotlib")
    plt.tight_layout()
    plt.show()
                                                    # Использование библеотеки seaborn
    number_row = []
    for i in range(1, N+1):
        number_row += [i]*N
    number_item = list(range(1, N+1))*N
    df = pd.DataFrame({"Значения": A.flatten(), "Номер строки": number_row,
                       "Номер элемента в строке": number_item})
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    plt.subplot(2, 2, 1)
    plt.title("Использование функции lineplot")
    sns.lineplot(x="Номер элемента в строке", y="Значения", hue="Номер строки", data=df, palette="Set2")
    plt.subplot(222)
    plt.title("Использование функции boxplot")
    sns.boxplot(x="Номер строки", y="Значения", palette="Set2", data=df)
    plt.subplot(223)
    plt.title("Использование функции kdeplot")
    sns.kdeplot(data=df, x="Номер элемента в строке", y="Значения", hue="Номер строки", palette="Set2")
    plt.subplot(224)
    plt.title("Использование функции heatmap")
    sns.heatmap(data=A, annot=True, fmt="d", linewidths=.5)
    plt.suptitle("Использование библиотеки seaborn")
    plt.tight_layout()
    plt.show()
except ValueError:
    print("Введены неверные даннные")

