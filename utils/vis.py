import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(mat,
                          y_names=None,
                          x_names=None,
                          ylabel="",
                          xlabel="",
                          title="",
                          matap="magma_r",
                          sort_mat=True,
                          transpose_axes=False,
                          cell_scale=2,
                          cmap_lims=None,
                          ax=None,
                          mat_mult=100.0,  # for percents
                          ):
    s = mat.shape
    # mat_sum = np.sum(mat, axis=1, keepdims=True)
    # mat_perc = mat / mat_sum.astype(float) * 100
    # In case class names are not passed, create them
    if cmap_lims is None:
        cmap_lims = (0.0, 1.0)

    if y_names is None:
        y_names = np.arange(s[0]).astype(np.str).tolist()
    if x_names is None:
        x_names = np.arange(s[1]).astype(np.str).tolist()

    y_names = np.array(y_names)
    x_names = np.array(x_names)

    mean_val = np.mean(np.nan_to_num(mat, 0.1))

    if sort_mat:
        x_inds = np.argsort(np.nan_to_num(mat, mean_val).mean(axis=0))
        y_inds = np.argsort(np.nan_to_num(mat, mean_val).mean(axis=1))
        mat = mat[y_inds, :]
        mat = mat[:, x_inds]

        y_names = y_names[y_inds]
        x_names = x_names[x_inds]
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(s[1] * cell_scale, s[0] * cell_scale))
    im = ax.imshow(mat, interpolation='nearest', cmap=matap, vmin=cmap_lims[0], vmax=cmap_lims[1])

    # We want to show all ticks...
    ax.set(xticks=np.arange(mat.shape[1]),
           yticks=np.arange(mat.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=x_names, yticklabels=y_names,
           title=title,
           ylabel=ylabel,
           xlabel=xlabel)

    ax.set_xticks(np.arange(mat.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(mat.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = (cmap_lims[1] - cmap_lims[0]) / 2.0 + cmap_lims[0]
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if mat[i, j] > thresh else "black"

            ax.text(j, i, "{}".format(format(mat[i, j] * mat_mult, ".1f")),
                    ha="center", va="center",
                    color=color)

    return fig
