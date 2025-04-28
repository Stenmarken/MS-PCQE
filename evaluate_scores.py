import json
from scipy import stats
import numpy as np

def read_dict(path):
    with open(path, 'r') as f:
        d = json.load(f)
    return d

def evaluate_scores(scores_dict):
    keys = sorted(list(scores_dict.keys()))
    if len(keys) == 0:
        print("Keys list is empty. Returning")
        return
    true_scores = [i for i in range(len(keys))]
    print(true_scores)
    fst = keys[0]
    file_names = list(scores_dict[fst].keys())
    print(file_names)
    sroccs = [0] * len(file_names)
    for i, f in enumerate(file_names):
        scores = [scores_dict[key][f] for key in keys]
        sroccs[i] = stats.spearmanr(scores, true_scores)
        print(f'scores for {f}: {list(map(lambda x: round(x, 2), scores))}')

    np_sroccs = np.array([r.statistic for r in sroccs])
    p_values = np.array([r.pvalue for r in sroccs])
    print(f'np_sroccs: {np_sroccs}')
    print(f'np.std: {np.std(np_sroccs):.2f}')
    print(f'np.mean: {np.mean(np_sroccs):.2f}')

    print(f'p_values: {list(map(lambda x: float(round(x, 2)), p_values))}')
    combined = stats.combine_pvalues(method='fisher', pvalues=p_values)
    print(f'Combined: {combined:}')



if __name__ == "__main__":
    d = read_dict('MS-PCQE_main/0-20_output.json')
    evaluate_scores(d)