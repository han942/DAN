import world
import torch
import numpy as np


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    

def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', world.config['test_u_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0] 
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)
            

def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result
    

class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                # add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')
# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([max(min(len(test_data[i]), k), 1) for i in range(len(test_data))]) # changed
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
        
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

# === [추가] 사용자별 NDCG 점수 배열을 반환하는 함수 ===
# =========================================================
def NDCGatK_r_per_user(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain (per user)
    기존 함수와 동일하지만, 최종 합계를 내지 않고 사용자별 점수 배열을 반환합니다.
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k), dtype=np.float32)
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
        
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return ndcg # [수정] 합계(sum)가 아닌 개별 점수 배열을 반환



# Demographic-Parity Metrics
def calculate_demographic_parity_ratio(group_rec_counts, group_user_counts):
    """
    Calculates the demographic parity ratio.
    Args:
        group_rec_counts (dict): {group_id: total_recommendations}
        group_user_counts (dict): {group_id: num_users}
    Returns:
        float: demographic parity ratio
    """
    if not group_user_counts or not any(group_user_counts.values()):
        return 0.0

    avg_exposures = {}
    for group_id, user_count in group_user_counts.items():
        if user_count > 0:
            avg_exposures[group_id] = group_rec_counts.get(group_id, 0) / user_count
        else:
            avg_exposures[group_id] = 0

    if not avg_exposures:
        return 0.0
        
    exposures = list(avg_exposures.values())
    min_exposure = min(exposures)
    max_exposure = max(exposures)

    if max_exposure == 0:
        return 1.0  # 모든 그룹에 추천이 없으면 완벽히 공평
        
    return min_exposure / max_exposure

# uRec, uPrec
def uRecPrecatK_r(sorted_items, test_data, r, k, pscore):
    pred_data = r[:, :k]
    pred_pscore = pscore[sorted_items][:, :k]
    ytrue_pscore_inv_sum = np.zeros((len(test_data), 1))

    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        for j in range(length):
            ytrue_pscore_inv_sum[i] += 1 / pscore[items[j]]

    ur = pred_data / pred_pscore
    precis = (ur / np.sum(1. / pred_pscore, axis=1, keepdims=True)).sum()
    recall = (ur / ytrue_pscore_inv_sum).sum()
    
    return {'urecall': recall, 'uprecision': precis}


def uNDCGatK_r(sorted_items, test_data, r, k, pscore):
    pred_data = r[:, :k]
    pred_pscore = pscore[sorted_items][:, :k]
    ytrue_pscore_inv_sum = np.zeros(len(test_data))
    
    max_r = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        max_r[i, :length] = 1
        
        for j in range(length):
            ytrue_pscore_inv_sum[i] += 1 / pscore[items[j]]
    
    tp = np.log2(np.arange(2, k + 2))
    idcg = np.sum(max_r / tp, axis=1)
    udcg = pred_data / (pred_pscore * tp)
    
    udcg = np.sum(udcg, axis=1) / ytrue_pscore_inv_sum
    idcg[idcg == 0.] = 1.
    undcg = udcg / idcg
    undcg[np.isnan(undcg)] = 0.
    
    return np.sum(undcg)
# ====================end Metrics=============================
# =========================================================
