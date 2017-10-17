import math


def f_prime(x_old):
    return -2 * x_old + 3


def cal():
    x_old = 0
    x_new = 4
    eps = 0.01
    presision = 0.00001

    while abs(x_new - x_old) > presision:
        x_old = x_new
        x_new = x_old + eps * f_prime(x_old)
    return x_new


if __name__ == "__main__":
    print cal()
