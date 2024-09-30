import math
from matplotlib import pyplot as plt
import numpy

def dF(Yn, A):
    dY = numpy.matmul(Yn, A)
    return dY

def f(X):
    return numpy.array([math.e**X + 2*math.e**(-2*X), math.e**X + math.e**(-2*X), math.e**X - 3*math.e**(-2*X)])

def InFile(file, eps, y, h_num, h_min):
    file.write('Точность epsilon: ' + str(eps))
    file.write('\nМинимальный шаг: ' + str(h_min))
    file.write('\nКоличество шагов: ' + str(h_num))
    file.write('\nРешение, полученное методом Эйлера:\n')
    for i in y:
        file.write(str(i) + '\n')
    file.write('\n-----------------------------------------------\n')

def euler(a, b, A, Y0, h0, eps, file):

    h = h0
    dots = []
    new_dot = b - h
    dots.append(new_dot)
    h_val = [] #значение шага в каждой точке
    h_val.append(h)
    h_num = 1  # количество шагов для заданной точности

    y = []
    y_exact = []

    y.append(Y0)
    y_exact.append(Y0)
    i = 0

    while new_dot > a - h:
        y_exact.append(f(new_dot))
        new_dot -= h
        dots.append(new_dot)
        y_new = y[i] - h * dF(y[i], A)
        while numpy.linalg.norm(y[i] - y_new) > eps:
            h /= 2
            y_new = y[i] - h * dF(y[i], A)
        if numpy.linalg.norm(y[i] - y_new) < eps/64:
            h *= 2
            h_num -= 1
        i += 1
        y.append(y_new)

        h_val.append(h)
        h_num += 1

    h_min = min(h_val) #минимальное значение шага для заданной точности

    #запись полученных данных в файл
    InFile(file, eps, y, h_num, h_min)

    return y, y_exact, h_num, h_val, h_min, dots



if __name__ == '__main__':
    A = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
    h0 = 0.1
    a = 0
    b = 2
    h_min_eps = []
    h_num_eps = []
    epsilon = [0.1, 0.01, 0.001]
    Y0 = numpy.array(f(2))

    nodes = numpy.linspace(a, b, int((b - a) / h0))

    file = open('results.txt', 'w')
    for i in epsilon:
        y, y_exact, h_num, h_val, h_min, dots = euler(a, b, A, Y0, h0, i, file)
        h_min_eps.append(h_min)
        h_num_eps.append(h_num)
        #1. изменение шага по отрезку
        plt.plot(dots, h_val)
        plt.xlabel(r'x')
        plt.ylabel(r'Шаг h')
        plt.suptitle('Изменение шага для eps = ' + str(i))
        plt.semilogy()
        #plt.show()
        plt.savefig('graphics\step_change_'+str(i)+'.png')
        plt.clf()

    file.close

    #построение графиков
    #2. минимальный шаг
    plt.plot(epsilon, h_min_eps)
    plt.xlabel(r'epsilon')
    plt.ylabel(r'Шаг h')
    plt.suptitle('Минимальный шаг')
    plt.semilogy()
    plt.semilogx()
    #plt.show()
    plt.savefig('graphics\step_min.png')
    plt.clf()

    #3. количество шагов
    plt.plot(epsilon, h_num_eps)
    plt.xlabel(r'epsilon')
    plt.ylabel(r'n')
    plt.suptitle('Количество шагов')
    plt.semilogy()
    plt.semilogx()
    #plt.show()
    plt.savefig('graphics\step_num.png')
    plt.clf()

    # 4. точное решение
    plt.plot(dots, y_exact)
    plt.xlabel(r'x')
    plt.ylabel(r'y')
    plt.suptitle('Точное решение')
    plt.semilogy()
    plt.savefig('graphics\exact_sol.png')
    plt.clf()


    # 4. метод Эйлера
    plt.plot(dots, y)
    plt.xlabel(r'x')
    plt.ylabel(r'y')
    plt.suptitle('Метод Эйлера')
    plt.semilogy()
    plt.savefig('graphics\euler_sol.png')
    plt.clf()

