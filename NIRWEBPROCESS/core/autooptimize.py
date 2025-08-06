from .bootstrap import train_plsr, train_plsda

def auto_optimize(X, y, classification=False, max_components=10):
    results = []
    for nc in range(1, max_components + 1):
        if classification:
            _, metrics, _ = train_plsda(X, y, n_components=nc)
        else:
            _, metrics, _ = train_plsr(X, y, n_components=nc)
        metrics["n_components"] = nc
        results.append(metrics)
    key_func = (lambda x: x.get("RMSECV", 1e6)) if not classification else (lambda x: -x.get("accuracy", 0))
    results_sorted = sorted(results, key=key_func)
    return results_sorted
