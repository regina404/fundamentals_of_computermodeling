import numpy as np
import math
from scipy.stats import uniform
from scipy.stats import chi2
import matplotlib.pyplot as plt


def gen_uniform_values(n):  # генерування ksi_i
    vals = uniform.rvs(size=n)
    return vals


def gen_x(n):  # генерування x_i
    ksi = gen_uniform_values(n)
    vals = -(1 / lambd) * np.log(ksi)
    return vals


def F_expon(x, lambd):  # Функція експоненційного розподілу
    return 1 - np.exp(- lambd * x)


def diagram(vals, X, n):
    plt.figure(figsize=(4, 2.5))
    plt.hist(vals, bins=50, density=True)
    plt.title(f"{n} спостережень")
    plt.axvline(X, color='k', linewidth=1)  # наносимо вибіркове середнє на графік
    plt.show()


# Chi2 criterion 
def get_equal_intervals(r, h):  # розбиваємо на r інтервалів шириною h
    ai = [0]
    s = 0
    for j in range(r):
        ai.append(s + h)
        s = s + h
    return ai


def get_v(X, xmax, r, h):  # кількість спостережень які потрапляють в кожен інтервал
    v = [0 for i in range(r)]
    for x_i in X:
        if x_i == xmax:
            v[-1] += 1
        else:
            ind = int(x_i / h)
            v[ind] += 1
    return v


def get_p(r, lambd, breakpoints):  # теоретичні ймовірності потрапляння в кожен інтервал
    funcs = [F_expon(point, lambd) for point in breakpoints]
    p = [(funcs[i] - funcs[i - 1]) for i in range(1, r + 1)]
    return p


def adjust_intervals(v, p): # об'єднує сусідні інтервали якщо кількість влучань < 5
    for i in reversed(range(len(v))):
        if v[i] < 5:
            new_v_i = v[i] + v[i - 1]
            v = v[:i - 1] + [new_v_i] + v[i + 1:]
            new_p_i = p[i] + p[i - 1]
            p = p[:i - 1] + [new_p_i] + p[i + 1:]
    new_r = len(v) # нове значення кількості інтервалів
    return v, p, new_r


def get_delta(n, r, v, p):  # значення статистики
    s = 0
    for i in range(r):
        s += (v[i] - n * p[i]) ** 2 / p[i]
    res = s / n
    return res


def chi2_pearson_test(x_vals, n, gamma, lambd_teor):
    r = 50  # кількість інтервалів
    xmax = np.max(x_vals)
    h = xmax / r  # ширина інтервалу
    print("Кількість інтервалів =", r, "Ширина інтервалу =", h)
    breakpoints = get_equal_intervals(r, h)
    v = get_v(x_vals, xmax, r, h)
    p = get_p(r, lambd_teor, breakpoints)

    v, p, r = adjust_intervals(v, p)

    delta = get_delta(n, r, v, p)  # значення статистики
    Z_gamma = chi2.ppf(1 - gamma, df=r - 1)  # теоретичне значення статистики
    print("Z_gamma=", Z_gamma)
    print('Delta=', delta)
    if delta < Z_gamma:
        print(f"Ho ACCEPTED: x_i ~ exp(λ={lambd})")
    else:
        print(f"Ho REJECTED: x_i !~ exp(λ={lambd})")


lambda_values = [1, 5, 10]
gamma = 0.05  # рівень значимості
n = 10000


def task(lambd, n):
    x_vals = gen_x(n)
    X = np.sum(x_vals) / n  # вибіркове середнє
    S = sum((i - X) ** 2 for i in x_vals) / n  # вибіркова дисперсія
    diagram(x_vals, X, n)

    print("λ =", lambd)
    print("Вибіркове середнє:", X)
    print("Вибіркова дисперсія:", S)

    # Перевіряємо на відповідність експоненційному закону розподілу
    chi2_pearson_test(x_vals, n, gamma, lambd_teor=lambd)


for lambd in lambda_values:
    task(lambd, n)
