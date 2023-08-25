from deeplift.dinuc_shuffle import dinuc_shuffle, traverse_edges, shuffle_edges, prepare_edges
from collections import Counter
import numpy as np


def calc_hypo_on_data(data):
    def split_list(a_list, number_of_splits):
        step = len(a_list) // number_of_splits + (1 if len(a_list) % number_of_splits else 0)
        return [a_list[i * step:(i + 1) * step] for i in range(number_of_splits)]

    def onehot_dinuc_shuffle(s):
        s = np.squeeze(s)
        argmax_vals = "".join([str(x) for x in np.argmax(s, axis=-1)])
        shuffled_dataset_lst = [x for x in traverse_edges(argmax_vals, shuffle_edges(prepare_edges(argmax_vals)))]
        to_remove = [' ', '\n', '[', ']']
        shuffled_dataset_lst = [i for i in shuffled_dataset_lst if i not in to_remove]
        shuffled_dataset_lst = [eval(i) for i in shuffled_dataset_lst]
        shuffled_dataset_lst = split_list(shuffled_dataset_lst, len(s))
        hypo_shuffled_dataset = np.zeros_like(s)
        for seq_index in range(len(shuffled_dataset_lst)):
            for num_index in range(len(shuffled_dataset_lst[seq_index])):
                if shuffled_dataset_lst[seq_index][num_index] == 0:
                    hypo_shuffled_dataset[seq_index][num_index][0] = 1.
                if shuffled_dataset_lst[seq_index][num_index] == 1:
                    hypo_shuffled_dataset[seq_index][num_index][1] = 1.
                if shuffled_dataset_lst[seq_index][num_index] == 2:
                    hypo_shuffled_dataset[seq_index][num_index][2] = 1.
                if shuffled_dataset_lst[seq_index][num_index] == 3:
                    hypo_shuffled_dataset[seq_index][num_index][3] = 1.
        return hypo_shuffled_dataset

    s = data
    hypo_dataset = onehot_dinuc_shuffle(s)
    hypo_dataset = np.transpose(hypo_dataset, (0, 2, 1))
    hypo_dataset = np.expand_dims(hypo_dataset, axis=2)
    return hypo_dataset
