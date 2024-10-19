import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Calculate Z score
def standardScore(x, m, q):
    return round(float((x - m) / q), 2)

# calculates the probability based on one given Z score 
def calculateProbability(xS, m, q, mode):
    zS = standardScore(xS, m, q)

    x = np.linspace(-5, 5, 1000)
    y = (1 / (np.sqrt(2 * np.pi))) * (np.exp(- 0.5 * (x ** 2)))

    if (zS > 0 or zS < 0) and mode == '<':
        probability = norm.cdf(zS)
        plt.fill_between(x, y, where=(x <= zS), color='grey', alpha=0.4)
    elif (zS < 0 or zS > 0) and mode == '>':
        probability = 1 - norm.cdf(zS)
        plt.fill_between(x, y, where=(x >= zS), color='grey', alpha=0.4) 

    # Plot probability using matplotlib
    plt.plot(x, y, label='Gaussian Curve', color='k')
    plt.xlabel('Z Score')
    plt.ylabel('Probability Density Function')
    plt.title('Gaussian Curve')
    plt.grid(True)
    plt.show()

    print('Z =', zS)
    return probability

# calculates the probability based on two given Z scores
def calculateProbability2(x1, x2, m, q):
    z1 = standardScore(x1, m, q)
    z2 = standardScore(x2, m, q)

    x = np.linspace(-5, 5, 1000)
    y = (1 / (np.sqrt(2 * np.pi))) * (np.exp(- 0.5 * (x ** 2)))

    if z1 == 0 and z2 >= 0:
        probability2 = 0.5 - norm.cdf(z2)
        plt.fill_between(x, y, where=(x >= 0) & (x <= z2), color='grey', alpha=0.4)
    elif z1 <= 0 and z2 == 0:
        probability2 = 0.5 - norm.cdf(z1)
        plt.fill_between(x, y, where=(x <= 0) & (x >= z1), color='grey', alpha=0.4)
    elif z1 != 0 and z2 != 0:
        probability2 = 0.5 - (norm.cdf(z1)) + (norm.cdf(z2) - 0.5)
        plt.fill_between(x, y, where = (x >= z1) & (x <= z2), color = 'grey', alpha = 0.4)

    # Plot probability using matplotlib
    plt.plot(x, y, label = 'Gaussian Curve', color = 'k')
    plt.xlabel('Z Score')
    plt.ylabel('Probability Density Function')
    plt.title('Gaussian Curve')
    plt.grid(True)
    plt.show()

    print('Z1 =', z1)
    print('Z2 =', z2)
    return probability2

# Check random variable x
Check_RandomVarX = input('Input one random var X? (yes/no): ')

if Check_RandomVarX.lower() == 'yes': 
    mode = input('Mode (">" or "<"): ')

    xS = float(input('X: '))
    m = float(input('M: '))
    q = float(input('Q: '))

    probability = abs(calculateProbability(xS, m, q, mode))

    print('Probability =', round(probability, 4))
    print('Percentage =', round((probability * 100), 2), '%')

elif Check_RandomVarX.lower() == 'no':
    x1 = float(input('X1: '))
    x2 = float(input('X2: '))
    m = float(input('M: '))
    q = float(input('Q: '))

    probability2 = abs(calculateProbability2(x1, x2, m, q))

    print('Probability =', round(probability2, 4))
    print('Percentage =', round((probability2 * 100), 2), '%')

# X = Value of random var
# M = Mean
# Q = Standard deviation
# Z = Standard score