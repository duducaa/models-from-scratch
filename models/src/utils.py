import numpy as np

def train_test_split(X: np.ndarray, y: np.ndarray, train_size: np.ndarray, shuffle: bool = False):
    idxs = list(range(X.shape[0]))
    
    if shuffle:
        np.random.shuffle(idxs)
        
    X = X[idxs]
    y = y[idxs]

    limit = int(train_size * X.shape[0])
    return X[:limit], X[limit:], y[:limit], y[limit:]

