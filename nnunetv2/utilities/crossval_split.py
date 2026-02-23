from typing import List

import numpy as np
from sklearn.model_selection import KFold


def generate_crossval_split(train_identifiers: List[str], seed=12345, n_splits=5) -> List[dict[str, List[str]]]:
    splits = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (train_idx, test_idx) in enumerate(kfold.split(train_identifiers)):
        train_keys = np.array(train_identifiers)[train_idx]
        test_keys = np.array(train_identifiers)[test_idx]
        splits.append({})
        splits[-1]['train'] = list(train_keys)
        splits[-1]['val'] = list(test_keys)
    return splits

# HERE I MODIFIED THE SOURCE CODE
#################################################
# def generate_temporal_split(train_identifiers: List[str], n_splits: int = 5) -> List[dict[str, List[str]]]:
#     """
#     Temporal cross-validation split.
    
#     fold_0 -> last 20% as validation
#     fold_1 -> previous 20%
#     ...
#     fold_4 -> first 20%
#     Output format is identical to nnU-Net crossval split.
#     """
#     identifiers = np.array(train_identifiers)
#     n_cases = len(identifiers)

#     fold_size = n_cases // n_splits
#     splits = []

#     for fold in range(n_splits):
#         val_start = n_cases - (fold + 1) * fold_size
#         val_end   = n_cases - fold * fold_size if fold != 0 else n_cases
#         val_idx = np.arange(val_start, val_end)
#         train_idx = np.setdiff1d(np.arange(n_cases), val_idx)
#         splits.append({"train": identifiers[train_idx].tolist(), "val": identifiers[val_idx].tolist()})

#     return splits