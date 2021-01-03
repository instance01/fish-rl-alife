import pickle
import matplotlib.pyplot as plt


base_path = 'pickles/'

with open(base_path + 'i10_p150_two_net_herding.pickle', 'rb') as f:
    i10_p150 = pickle.load(f)
with open(base_path + 'i10_p75_two_net_herding.pickle', 'rb') as f:
    i10_p75 = pickle.load(f)
with open(base_path + 'i5_p150_two_net_herding.pickle', 'rb') as f:
    i5_p150 = pickle.load(f)
with open(base_path + 'i5_p75_two_net_herding.pickle', 'rb') as f:
    i5_p75 = pickle.load(f)


print(i10_p150)
print(i10_p75)
print(i5_p150)
print(i5_p75)


# All in one scatter plot.
# plt.scatter(i10_p150[0], [x[0] for x in i10_p150[1]], c='g')
# plt.scatter(i10_p75[0], [x[0] for x in i10_p75[1]], c='r')
# plt.scatter(i5_p75[0], [x[0] for x in i5_p75[1]], c='b')
# plt.scatter(i5_p150[0], [x[0] for x in i5_p150[1]], c='c')
# plt.show()

# Two plots.
fig, axs = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

axs[0].set_title('10 initial fishes')
axs[0].set_ylabel('Herding/Coop rate')
axs[0].set_xlabel('Initial shark survival time t')
axs[0].scatter([x[1:] for x in i10_p150[0]], [x[0] for x in i10_p150[1]], c='g', label='p150')
axs[0].scatter([x[1:] for x in i10_p75[0]], [x[0] for x in i10_p75[1]], c='r', label='p75')
axs[0].legend()

axs[1].set_title('5 initial fishes')
axs[1].set_ylabel('Herding/Coop rate')
axs[1].set_xlabel('Initial shark survival time t')
axs[1].scatter([x[1:] for x in i5_p150[0]], [x[0] for x in i5_p150[1]], c='g', label='p150')
axs[1].scatter([x[1:] for x in i5_p75[0]], [x[0] for x in i5_p75[1]], c='r', label='p75')
axs[1].legend()

# plt.tight_layout()
txt = "Each point represents roughly 20-30 different models using the respective parameter configuration. p75/p150 stands for the bonus survival time for each fish eaten (i.e. time given is 75 and 150 respectively)."
fig.text(.02, .04, txt, wrap=True)
# plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
# fig.set_size_inches(11, 6, forward=True)
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.2)
plt.show()
