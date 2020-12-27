import numpy as np
import matplotlib.pyplot as plt
import pickle


with open('inn_last_fish_pop.pickle', 'rb') as f:
    a = pickle.load(f)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


a = moving_average(a, 100)
plt.plot(a, linewidth=.8)
plt.ylabel('Fish population')
plt.xlabel('Steps in million')
plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000], ['0', '2', '4', '6', '8', '10', '12'])
plt.axvspan(0, 600, facecolor='#990000', alpha=0.2)
plt.axvspan(600, 5100, facecolor='#009900', alpha=0.2)
plt.axvspan(5100, 5900, facecolor='#000099', alpha=0.2)
# plt.title('a)                                 b)                                   c)')
plt.title('a                                   b                                    c ')
plt.show()
