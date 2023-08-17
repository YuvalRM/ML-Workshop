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
        lst = [x for x in traverse_edges(argmax_vals, shuffle_edges(prepare_edges(argmax_vals)))]
        to_remove = [' ', '\n', '[', ']']
        lst = [i for i in lst if i not in to_remove]
        lst = [eval(i) for i in lst]
        lst = split_list(lst, len(s))
        to_return = np.zeros_like(s)
        for seq_index in range(len(lst)):
            for num_index in range(len(lst[seq_index])):
                if lst[seq_index][num_index] == 0:
                    to_return[seq_index][num_index][0] = 1.
                if lst[seq_index][num_index] == 1:
                    to_return[seq_index][num_index][1] = 1.
                if lst[seq_index][num_index] == 2:
                    to_return[seq_index][num_index][2] = 1.
                if lst[seq_index][num_index] == 3:
                    to_return[seq_index][num_index][3] = 1.
        return to_return

    s = data
    new_s = onehot_dinuc_shuffle(s)
    new_s = np.transpose(new_s, (0, 2, 1))
    new_s = np.expand_dims(new_s, axis=2)
    return new_s
