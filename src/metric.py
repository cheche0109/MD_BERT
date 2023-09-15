import sklearn.metrics as skm
import scipy
import scipy.stats


def compute_metrics(true_scores, predicted_scores, metrics=("mse", "pearsonr", "r2", "spearmanr", "mae")):

    metrics_dict = {}
    # add scores to evaluation dict
    metrics_dict["true"] = true_scores
    metrics_dict["predicted"] = predicted_scores
    
    predicted_scores = [float(p) for p in predicted_scores]
    # compute requested metrics
    for metric in metrics:
        if metric == "mse":
            try: 
                metrics_dict["mse"] = skm.mean_squared_error(true_scores, predicted_scores)
            except:
                metrics_dict["mse"] = -1
        elif metric == "pearsonr":
            try:
                metrics_dict["pearsonr"] = scipy.stats.pearsonr(true_scores, predicted_scores)[0]
            except:
                metrics_dict["pearsonr"] = -1
        elif metric == "spearmanr":
            try:
                metrics_dict["spearmanr"] = scipy.stats.spearmanr(true_scores, predicted_scores)[0]
            except:
                metrics_dict["spearmanr"] = -1
        elif metric == "r2":
            try:
                metrics_dict["r2"] = skm.r2_score(true_scores, predicted_scores)
            except:
                metrics_dict["r2"] = -1
        elif metric == "mae":
            try:
                metrics_dict["mae"] = skm.mean_absolute_error(true_scores, predicted_scores)
            except:
                metrics_dict["mae"] = -1
    return metrics_dict