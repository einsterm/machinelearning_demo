import matplotlib.pyplot as plt
import random


def first_digital(x):
    while x >= 10:
        x /= 10
    return x


if __name__ == "__main__":
    n = 1
    frequency = [0] * 9
    for i in range(1, 1000):
        n *= i
        # n = random.randint(1, 100)
        # print n
        m = first_digital(n) - 1
        frequency[m] += 1

    plt.plot(frequency, 'r-', linewidth=2)
    # plt.plot(frequency, 'go', markerize=8)
    plt.grid(True)
    plt.show()
