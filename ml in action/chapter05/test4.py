# --*-- encoding:utf-8 --*--
'''
求导数为0时，自变量的值
'''


def f_prime(x_old):
    return -2 * x_old + 3


def cal():
    x_old = 0  # 导数为0时，x的初始值
    x_new = 50  # 导数为0时，x的估计值
    eps = 0.4
    presision = 0.00001

    while abs(x_new - x_old) > presision:  # abs 返回绝对值 上一次求导的值减去本次求导的值,如果趋势是0，则认为极值存在
        x_old = x_new
        x_new = x_old + eps * f_prime(x_old)
    return x_new


def mycal():
    x0 = 0
    e = 0.1
    for i in range(12004):
        y = x0 + e * f_prime(x0)
        x0 = y
    return x0


if __name__ == "__main__":
    print(mycal())
