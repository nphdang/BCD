import numpy as np
import matplotlib.pyplot as plt
# set big font
# import seaborn as sns
# sns.set_context("notebook", font_scale=1.8)
plt.style.use('fivethirtyeight')

dataset = "olivetti" # mnist, cifar10, olivetti
with open("rate_rnd_run_ratio_{}.file".format(dataset), "rb") as f:
    rnd = np.load(f)
with open("rate_local_run_ratio_{}.file".format(dataset), "rb") as f:
    local = np.load(f)
with open("rate_bo_run_ratio_{}.file".format(dataset), "rb") as f:
    bo = np.load(f)

print("dataset: {}".format(dataset))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
markers = ["o", "s", "v", "^", "*", "d", "h", "p", "x", "+"]
marker_size = 4
lw = 3
n_runs = 10
fig, ax = plt.subplots(figsize=(7, 5))
x_range = range(0, 9)
# plot random method
y_range = np.mean(rnd, axis=0)
# sort ascending
y_range.sort()
err_range = np.std(rnd, axis=0) / np.sqrt(n_runs)
ax.plot(x_range, y_range, linewidth=lw, label="Random", color=colors[0], marker=markers[0], markersize=marker_size)
ax.fill_between(x_range, y_range - err_range, y_range + err_range, color=colors[0], alpha=0.2)
# plot local method
y_range = np.mean(local, axis=0)
# sort ascending
y_range.sort()
err_range = np.std(local, axis=0) / np.sqrt(n_runs)
ax.plot(x_range, y_range, linewidth=lw, label="VerIDeep", color=colors[1], marker=markers[1], markersize=marker_size)
ax.fill_between(x_range, y_range - err_range, y_range + err_range, color=colors[1], alpha=0.2)
# plot local method
y_range = np.mean(bo, axis=0)
# sort ascending
y_range.sort()
err_range = np.std(bo, axis=0) / np.sqrt(n_runs)
ax.plot(x_range, y_range, linewidth=lw, label="Our method", color=colors[2], marker=markers[2], markersize=marker_size)
ax.fill_between(x_range, y_range - err_range, y_range + err_range, color=colors[2], alpha=0.2)
ax.set_xlabel("Ratio of weight changes")
ax.set_ylabel("Detection rate")
plt.xticks(x_range, ["0.01", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"])
plt.title(dataset)
plt.legend()
plt.savefig("detection_rate_{}.pdf".format(dataset), bbox_inches="tight")
plt.close()

