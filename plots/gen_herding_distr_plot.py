import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.optimize import curve_fit


mpl.font_manager._rebuild()
plt.rc('font', family='Raleway')


base_path = '../pickles/'

sp100_ma3obs = False
sp100_ma3obs_sp100 = False
if sp100_ma3obs:
    with open(base_path + 'i10_two_net_herding_sp100_ma3obs.pickle', 'rb') as f:
        i10_p150 = pickle.load(f)
    with open(base_path + 'i5_two_net_herding_sp100_ma3obs.pickle', 'rb') as f:
        i5_p150 = pickle.load(f)
    with open(base_path + 'i2_two_net_herding_sp100_ma3obs.pickle', 'rb') as f:
        i2_p150 = pickle.load(f)
elif sp100_ma3obs_sp100:
    with open(base_path + 'i10_two_net_herding_sp100.pickle', 'rb') as f:
        i10_p150 = pickle.load(f)
    with open(base_path + 'i5_two_net_herding_sp100.pickle', 'rb') as f:
        i5_p150 = pickle.load(f)
    with open(base_path + 'i2_two_net_herding_sp100.pickle', 'rb') as f:
        i2_p150 = pickle.load(f)
else:
    with open(base_path + 'i10_p150_two_net_herding.pickle', 'rb') as f:
        i10_p150 = pickle.load(f)
    with open(base_path + 'i5_p150_two_net_herding.pickle', 'rb') as f:
        i5_p150 = pickle.load(f)
    # TODO Could use i2_p150_sp500_two_net_herding_20repeats.pickle too.
    # Right now we have 3 repeat evals per model.
    with open(base_path + 'i2_p150_sp500_two_net_herding.pickle', 'rb') as f:
        i2_p150 = pickle.load(f)


print(i10_p150)
print(i5_p150)
print(i2_p150)


def func(x, a, b, c, d, e):
    return a + b * x + c * x**2 + d * x**3 + e * x**4


# Three plots.
fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

axs[0].set_title('10 initial prey')
axs[0].set_ylabel('Herding Rate')
axs[0].set_xlabel('Initial predator survival time t * 100')
x = [str(int(x[1:]) // 100) for x in i10_p150[0]]
y = [x[0] for x in i10_p150[1]]
axs[0].bar(x, y, color='#009440')
x = np.arange(len(y))
fittedParameters, pcov = curve_fit(func, x, y)
y_fit = func(x, *fittedParameters)
axs[0].plot(x, y_fit, 'k:')


axs[1].set_title('5 initial prey')
# axs[1].set_ylabel('Herding Rate')
axs[1].set_xlabel('Initial predator survival time t * 100')
x = [str(int(x[1:]) // 100) for x in i5_p150[0]]
y = [x[0] for x in i5_p150[1]]
axs[1].bar(x, y, color='#009440')
x = np.arange(len(y))
fittedParameters, pcov = curve_fit(func, x, y)
y_fit = func(x, *fittedParameters)
axs[1].plot(x, y_fit, 'k:')

axs[2].set_title('2 initial prey')
# axs[2].set_ylabel('Herding Rate')
axs[2].set_xlabel('Initial predator survival time t * 100')
x = [str(int(x[1:]) // 100) for x in i2_p150[0]]
y = [x[0] for x in i2_p150[1]]
axs[2].bar(x, y, color='#009440')
x = np.arange(len(y))
fittedParameters, pcov = curve_fit(func, x, y)
y_fit = func(x, *fittedParameters)
y_fit[1] -= .01
axs[2].plot(x, y_fit, 'k:')

plt.tight_layout()
plt.ylim(0., 1.0)
plt.show()
fig.savefig("herding.pdf", bbox_inches='tight')
