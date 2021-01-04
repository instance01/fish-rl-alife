import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl


mpl.font_manager._rebuild()
# plt.rc('font', family='Open Sans')
plt.rc('font', family='Raleway')


# with open('../pickles/inn_last_fish_pop.pickle', 'rb') as f:
with open('../pickles/t800_5fish_last_fish_pop.pickle', 'rb') as f:
    a = pickle.load(f)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


a = moving_average(a, 100)
print(len(a))

plt.figure(figsize=(4, 3))
plt.plot(a, linewidth=.8, c='#ffffff')
plt.ylabel('Fish population at the end of an episode')
plt.xlabel('Steps in million')
plt.xticks([0, 2000, 4000, 6000, 8000, 10000, 12000, 13980], ['0', '3', '6', '9', '12', '15', '18', '21'])
plt.axvspan(0, 2000, facecolor='#00622c', alpha=1.0)
plt.axvspan(2000, 4800, facecolor='#007a35', alpha=1.0)
plt.axvspan(4800, 13980, facecolor='#009440', alpha=1.0)
plt.axvline(2000, color='black', linestyle='--' , linewidth=.2)
plt.axvline(4800, color='black', linestyle='--', linewidth=.2)
plt.axhline(2, color='black', linestyle='--', linewidth=.5)
plt.title(' a           b                              c                    ')
plt.xlim(0, 13980)
plt.ylim(0, 7)
plt.tight_layout()
plt.show()
