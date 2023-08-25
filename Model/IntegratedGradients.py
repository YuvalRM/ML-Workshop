from captum.attr import IntegratedGradients
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import deepsea_model
import  numpy as np
data_dir = "data_dir/"
# import hypo_scores # uncomment if you wish to use calc_IG_with_hypo()

model = deepsea_model.get_seqpred_model(load_weights=True)

# data manipulation
test_data, test_labels = np.load(f"{data_dir}test_data.npy"), np.load(f"{data_dir}test_labels.npy")
test_data = np.transpose(test_data, (0, 2, 1))  # (455024, 4, 1000)
test_data = np.expand_dims(test_data, axis=2)  # (455024, 4, 1, 1000)
test_dataset = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_labels).float())


# this function calculates the integrated gradients of single sequence and single feature
def calc_IG(sequence_index):
    # defining model input tensors
    input1 = test_data[sequence_index]
    input1 = np.expand_dims(input1, axis=0)
    input1 = torch.tensor(input1).float()

    # defining baselines for each input tensor
    baseline1 = torch.rand(1, 4, 1, 1000)

    # calculating the integrated gradients
    ig = IntegratedGradients(model)
    attributions, approximation_error = ig.attribute((input1),
                                                     baselines=(baseline1),
                                                     method='gausslegendre',
                                                     return_convergence_delta=True,
                                                     target=[0])

    input_seq, input_scores = create_TF_MoDisco_inputs(input1, attributions)
    weights_lst = [input_scores]
    weight_dict = {'task': weights_lst}
    input_lst = [input_seq]

    return input_lst, weight_dict


# this function calculates the integrated gradients of single sequence and single feature, with hypo scores
# hypo scores are optional input to TF-MoDisco
def calc_IG_with_hypo():
    input_lst, weight_dict = calc_IG(0)  # can be changed to any sequence index

    # defining baselines for each input tensor
    baseline1 = torch.rand(1, 4, 1, 1000)

    # this part  calculates hypo scores
    data_to_hypo = hypo_scores.calc_hypo_on_data(test_data)
    hypo_input_1 = data_to_hypo[0]
    hypo_input_1 = np.expand_dims(hypo_input_1, axis=0)
    hypo_input_1 = torch.tensor(hypo_input_1).float()

    # calculating the integrated gradients
    ig = IntegratedGradients(model)
    hypo_attributions, hypo_approximation_error = ig.attribute((hypo_input_1),
                                                     baselines=(baseline1),
                                                     method='gausslegendre',
                                                     return_convergence_delta=True,
                                                     target=[0])

    hypo_input_seq, hypo_input_scores = create_TF_MoDisco_inputs(hypo_input_1, hypo_attributions)
    hypo_weights_lst = [hypo_input_scores]
    hypo_weight_dict = {'task': hypo_weights_lst}

    return input_lst, weight_dict, hypo_weight_dict


# this function calculates the integrated gradients for one feature, over many samples
def calc_IG_mult_samples():
    input_lst = []
    weights_lst = []
    for i in range(len(test_data)): # can be changed to use any group of sequences
        seq_input_lst, seq_weight_dict = calc_IG(i)
        input_lst.append(seq_input_lst)
        weights_lst.append(seq_weight_dict['task'])
    weight_dict = {'task': weights_lst}

    return input_lst, weight_dict


# this function modifies input and scores to TF-MoDisco form
def create_TF_MoDisco_inputs(input_sequence, attributions):
    # scores to TF-MoDisco form
    three_d_tensor = attributions[0]
    three_d_tensor = torch.transpose(three_d_tensor, 0, 1)
    three_d_tensor = torch.transpose(three_d_tensor, 1, 2)
    two_d_tensor = three_d_tensor[0]

    # input to TF-MoDisco form
    three_d_input = input_sequence.int()[0]
    three_d_input = torch.transpose(three_d_input, 0, 1)
    three_d_input = torch.transpose(three_d_input, 1, 2)
    two_d_input = three_d_input[0]

    return two_d_input.numpy(), two_d_tensor.numpy()