import pandas as pd
from scipy.stats import ttest_ind

def compare_groups(csv_path, metric, group_col, group1, group2):

    df = pd.read_csv(csv_path)

    g1 = df[df[group_col] == group1][metric]
    g2 = df[df[group_col] == group2][metric]

    stat, pval = ttest_ind(g1, g2)

    return {
        "metric": metric,
        "group1": group1,
        "group2": group2,
        "p_value": pval
    }