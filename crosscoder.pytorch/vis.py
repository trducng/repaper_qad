import matplotlib
matplotlib.use('agg')
import gc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


layer7_data = np.load("/data3/mech/internals/transformer.h.8.npy", mmap_mode="r")
# layer8_data = np.load("/data3/mech/internals/transformer.h.9.npy", mmap_mode="r")

# layer7 = np.load("/data3/mech/internals/layer7.stats.npy")
# layer8 = np.load("/data3/mech/internals/layer8.stats.npy")


def descripive_stats():
    """Get the min, max, quantiles 10, 25, 70, 90, median, mean and std"""
    data = layer7[:1000, 1:, :]
    data = data.reshape(-1, data.shape[-1])
    description = np.empty((9, data.shape[-1]))
    description[0] = np.min(data, axis=0)
    description[1] = np.max(data, axis=0)
    description[2:6] = np.quantile(data, [0.1, 0.25, 0.75, 0.9], axis=0)
    description[6] = np.median(data, axis=0)
    description[7] = np.mean(data, axis=0)
    description[8] = np.std(data, axis=0)

    return description


def violin_plot(data, feat_idx_list, out_path):
    """Violin plot of the data"""
    data = data[:1000, 1:, feat_idx_list]
    data = data.reshape(-1, data.shape[-1])
    axes = sns.violinplot(data=data, formatter=lambda x: str(feat_idx_list[x]))
    plt.savefig(out_path)
    axes.clear()
    plt.close()
    plt.clf()

    # gc.collect()


def pearsonr(data):
    """Correlation matrix of the data"""
    data = data[:1000, 1:, :]
    data = data.reshape(-1, data.shape[-1])
    corr = np.corrcoef(data, rowvar=False)
    sns.heatmap(corr)
    plt.savefig("pearsonr.png")
    return corr


def spearmanr(data):
    """Correlation matrix of the data"""
    data = data[:1000, 1:, :]
    data = data.reshape(-1, data.shape[-1])
    corr = stats.spearmanr(data)
    sns.heatmap(corr.correlation)
    plt.savefig("spearmanr.png")
    return corr


def kendalltau(data):
    """Correlation matrix of the data"""
    data = data[:1000, 1:, :]
    data = data.reshape(-1, data.shape[-1])
    corr = stats.kendalltau(data)
    sns.heatmap(corr.correlation)
    plt.savefig("kendalltau.png")
    return corr


def encoder_norm_vs_activation(data):
    pass

def main():
    for i in range(420, 500, 20):
    # for i in range(340, 420, 20):
        violin_plot(layer7_data, list(range(i, i+20)), f"logs/figs/layer7_{i:03}.png")

if __name__ == "__main__":
    main()
