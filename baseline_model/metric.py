import numpy as np
from tqdm import tqdm

def TopkAccuracy(true_idcs_list, top_k_indices, shop_idx_val):
    """
    query : 15437, gallery : 45455
    true_idcs_list : query에 상응하는 true shop image의 index들 (train_path의 index, max_length로 (-1) padded) (15437,82) (Tensor)
    top_k_indices : top_k (cos similarity) indices(0~45454) -> gallery의 index (15437, k) (Tensor)
    shop_idx_val : gallery의 true index list (train_path의 index) (45455)
    """

    acc = 0
    for i in tqdm(range(len(true_idcs_list)), desc='calculating topkacc'):
        # true idcs중 하나라도 topk_idx_list에 있으면 +1
        if len(set(true_idcs_list[i]).intersection(np.array(shop_idx_val)[top_k_indices[i]])) > 0:
            acc += 1
    return acc / len(true_idcs_list)

