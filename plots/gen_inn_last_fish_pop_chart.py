import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl


mpl.font_manager._rebuild()
# plt.rc('font', family='Open Sans')
plt.rc('font', family='Raleway')


with open('inn_last_fish_pop.pickle', 'rb') as f:
    a = pickle.load(f)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


a = moving_average(a, 100)
plt.plot(a, linewidth=.8, c='#ffffff')
plt.ylabel('Fish population at the end of an episode')
plt.xlabel('Steps in million')
plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000], ['0', '2', '4', '6', '8', '10', '12'])
# plt.axvspan(0, 600, facecolor='#990000', alpha=0.2)
# plt.axvspan(600, 5100, facecolor='#009900', alpha=0.2)
# plt.axvspan(5100, 5900, facecolor='#000099', alpha=0.2)
# plt.axvspan(0, 600, facecolor='#344966', alpha=0.6)
# plt.axvspan(600, 5100, facecolor='#2D7DD2', alpha=0.6)
# plt.axvspan(5100, 5900, facecolor='#009440', alpha=0.7)
plt.axvspan(0, 600, facecolor='#00622c', alpha=1.0)
plt.axvspan(600, 5100, facecolor='#007a35', alpha=1.0)
plt.axvspan(5100, 5900, facecolor='#009440', alpha=1.0)
plt.axvline(600, color='black', linestyle='--' , linewidth=.2)
plt.axvline(5100, color='black', linestyle='--', linewidth=.2)
plt.axhline(2, color='black', linestyle='--', linewidth=.5)
# plt.title('a)                                 b)                                   c)')
# plt.title('a                                   b                                    c ')
plt.title('a                                                 b                                                c ')
plt.xlim(0, 5900)
plt.ylim(0, 12)
plt.show()
