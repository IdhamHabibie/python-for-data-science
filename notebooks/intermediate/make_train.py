import sys

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def make_dataset(n_samples):
    X, y = make_classification(
        n_samples=n_samples, n_features=2, n_redundant=0,
        n_clusters_per_class=1, scale=[2, 9], random_state=21,
        class_sep=.82, shift=[1, 8]
    )
    train_set = np.concatenate((X.round(3), y[:, None]), axis=1)
    return train_set


if __name__ == "__main__":
    n_sample = int(sys.argv[1])
    data = make_dataset(n_sample)
    df = pd.DataFrame(data, columns=["grade", "score", "label"])
    df = df.astype({"grade": np.float, "score": np.float, "label": np.int})
    df.to_csv("data/scores.csv", index=False)

