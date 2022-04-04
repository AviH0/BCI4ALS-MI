import pandas as pd
import random as rd

from check_no_adj_tar_rep import check_no_adj_tar_rep


def create_pseudo_rand(path_stim, num_trials):
        # at the moment, 5 different distractors which are not 'yes' or 'no'.
        rep_opt_target = [1, 2, 3]
        rep_target_num = rd.choice(rep_opt_target)
        df = pd.read_excel(path_stim)

        yes_no = df[(df["stim"] == 'yes') | (df["stim"] == 'no')]
        other_rows = df.iloc[2:8]
        target_repeated = pd.concat([yes_no] * rep_target_num, ignore_index=True)
        other_repeated = pd.concat([other_rows] * rep_target_num, ignore_index=True)
        df = pd.concat([target_repeated, other_repeated], ignore_index=True)
        df_rand_ord = df.sample(frac=1, ignore_index=True)

        df_rand_ord.to_excel('CHECK.xlsx')
        # Now check that the same target does not repeat straightaway
        df_final = check_no_adj_tar_rep(df_rand_ord)


        df_final.to_excel("all_stim_rand_order.xlsx")

        return df_final
k=create_pseudo_rand(r'C:\Users\yarde\Documents\GitHub\BCI4ALS-MI\stim_words.xlsx', 1)
# place "r" before the path string to address special character, such as '\'. Don't forget to put the file name
# at the end of the path + '.xlsx'
