import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from six.moves import range
from sklearn import metrics
from sklearn.metrics import roc_curve


def compute_cos_pairwise_eer(embeddings, labels, max_num_embeddings=5000):
    """Compute pairwise EER using cosine similarity.
    The EER is estimated by interp1d and brentq, so it is not the exact value and may be a little different each time.

    Args:
        embeddings: The embeddings.
        labels: The class labels.
        max_num_embeddings: The max number of embeddings to compute the EER.
    :return: The pairwise EER.
    """
    embeddings /= np.sqrt(np.sum(embeddings**2, axis=1, keepdims=True) + 1e-12)
    num_embeddings = embeddings.shape[0]
    if num_embeddings > max_num_embeddings:
        # Downsample the embeddings and labels
        step = num_embeddings // max_num_embeddings
        embeddings = embeddings[range(0, num_embeddings, step), :]
        labels = labels[range(0, num_embeddings, step)]
        num_embeddings = embeddings.shape[0]

    score_mat = np.dot(embeddings, np.transpose(embeddings))
    scores = np.zeros((num_embeddings * (num_embeddings - 1) // 2))
    keys = np.zeros((num_embeddings * (num_embeddings - 1) // 2))
    index = 0
    for i in range(num_embeddings - 1):
        for j in range(i + 1, num_embeddings):
            scores[index] = score_mat[i, j]
            keys[index] = 1 if labels[i] == labels[j] else 0
            index += 1

    fpr, tpr, thresholds = metrics.roc_curve(keys, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


# https://stackoverflow.com/questions/28339746/equal-error-rate-in-python
def EER(y, y_score):
    """
    :param y: True binary labels in range {0, 1} or {-1, 1}. If labels are not binary, pos_label should be explicitly given
    :param y_score: Target scores, can either be probability estimates of the positive class, confidence values,
    or non-thresholded measure of decisions (as returned by “decision_function” on some classifiers).
    :return: eer, thresh
    """
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh.item()


def compute_frr_far(tar, imp):

    tar_unique, tar_counts = np.unique(tar, return_counts=True)
    imp_unique, imp_counts = np.unique(imp, return_counts=True)
    thresholds = np.unique(np.hstack((tar_unique, imp_unique)))

    pt = np.hstack((tar_counts, np.zeros(len(thresholds) - len(tar_counts), dtype=np.int)))
    pi = np.hstack((np.zeros(len(thresholds) - len(imp_counts), dtype=np.int), imp_counts))

    pt = pt[np.argsort(np.hstack((tar_unique, np.setdiff1d(imp_unique, tar_unique))))]
    pi = pi[np.argsort(np.hstack((np.setdiff1d(tar_unique, imp_unique), imp_unique)))]

    fr = np.zeros(pt.shape[0] + 1, dtype=np.int)
    fa = np.zeros(pi.shape[0] + 1, dtype=np.int)

    for i in range(1, len(pt) + 1):
        fr[i] = fr[i - 1] + pt[i - 1]

    for i in range(len(pt) - 1, -1, -1):
        fa[i] = fa[i + 1] + pi[i]

    frr = fr / max(len(tar), 1)
    far = fa / max(len(imp), 1)

    thresholds = np.hstack((thresholds, thresholds[-1] + 1e-6))
    return thresholds, frr, far
