import numpy as np


def check_no_adj_tar_rep(df):
    # check indices of each type
    check_rep_no = df[df['stim'] == 'no'].index.values
    check_rep_yes = df[df['stim'] == 'yes'].index.values
    index_dist = df[df['target'] == 0].index.values

    # calculate differences of indices- check if adjacent index
    dif_no = np.diff(check_rep_no)
    dif_yes = np.diff(check_rep_yes)
    dif_dist = np.diff(index_dist)

    # get indices where the difference is zero
    ind_no_1 = check_rep_no[np.where(dif_no == 1)[0]]
    ind_yes_1 = check_rep_yes[np.where(dif_yes == 1)[0]]
    ind_dist_1 = index_dist[np.where(dif_dist == 1)[0]]

    # get indices of "sandwitch" distractor: distractor between two other distractors
    dif_dist_ind_1 = np.diff(ind_dist_1)

    ind_sand = ind_dist_1[np.where(dif_dist_ind_1 == 1)[0]] + 1

    counter_dist = 0
    if 1 in dif_no:
        for change in range(len(ind_no_1)):
            tar, dist = df.iloc[ind_no_1[change]].copy(), df.iloc[ind_sand[counter_dist+2*counter_dist]].copy()
            df.iloc[ind_no_1[change]], df.iloc[ind_sand[counter_dist+2*counter_dist]] = dist, tar
            counter_dist = counter_dist+1

    if 1 in dif_yes:
        for change_2 in range(len(ind_yes_1)):
            tar, dist = df.iloc[ind_yes_1[change_2]].copy(), df.iloc[ind_sand[counter_dist+2*counter_dist]].copy()
            df.iloc[ind_no_1[change_2]], df.iloc[ind_sand[counter_dist+2*counter_dist]] = dist, tar
            counter_dist = counter_dist + 1

    return df
