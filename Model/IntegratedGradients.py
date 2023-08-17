from captum.attr import IntegratedGradients
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import deepsea_model
import  numpy as np
data_dir = "data_dir/"

model=deepsea_model.get_seqpred_model(load_weights=True)

test_data, test_labels = np.load(f"{data_dir}test_data.npy"), np.load(f"{data_dir}test_labels.npy")
test_data = np.transpose(test_data, (0, 2, 1))  # (455024, 4, 1000)
test_data = np.expand_dims(test_data, axis=2)  # (455024, 4, 1, 1000)
test_dataset = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_labels).float())


# this function calculate the integrated gradients
def calc_IG():
    # defining model input tensors
    input1 = test_data[0]
    input1 = np.expand_dims(input1, axis=0)
    input1 = torch.tensor(input1).float()

    # print("input1", input1.shape) # 1, 4, 1, 1000

    # defining baselines for each input tensor
    baseline1 = torch.rand(1, 4, 1, 1000)

    # defining and applying integrated gradients on ToyModel and the
    ig = IntegratedGradients(model)
    attributions, approximation_error = ig.attribute((input1),
                                                     baselines=(baseline1),
                                                     method='gausslegendre',
                                                     return_convergence_delta=True,
                                                     target=[0])
    # print(attributions)
    # print("attributions shape", attributions.shape)  # 1, 4, 1, 1000
    # print("three_d_tensor shape", attributions[0].shape)  # 4, 1, 1000

    three_d_tensor = attributions[0]
    three_d_tensor = torch.transpose(three_d_tensor, 0, 1)
    # print("three_d_tensor shape after transpose is", three_d_tensor.shape) # 1, 4, 1000
    three_d_tensor = torch.transpose(three_d_tensor, 1, 2)
    # print(three_d_tensor)
    # print("three_d_tensor shape after second transpose  transpose is", three_d_tensor.shape) # 1, 1000, 4 as wanted

    three_d_input = input1.int()[0]
    three_d_input = torch.transpose(three_d_input, 0, 1)
    # print("three_d_input shape after transpose is", three_d_input.shape) # 1, 4, 1000
    three_d_input = torch.transpose(three_d_input, 1, 2)
    # print(three_d_input)
    # print("three_d_tensor shape after second transpose  transpose is", three_d_input.shape) # 1, 1000, 4 as wanted

    # create list with input data on right dimentions (one-hot vectors)
    two_d_input = three_d_input[0]
    input_lst = []
    input_lst.append(two_d_input.numpy())
    # print("two_d_input shape is", input_lst) # 1, 1000, 4 as wanted

    # create list with attributions data on right dimentions
    two_d_input = three_d_tensor[0]
    weights_lst = []
    weights_lst.append(two_d_input.numpy())
    # print("two_d_input shape is", weights_lst) # 1, 1000, 4 as wanted

    print("visited IG function")

    # creates the required dict
    weight_dict = {'task': weights_lst}

    # return input_lst, weights_lst
    return input_lst, weight_dict


# this function calculate the integrated gradients for one feature, on many samples

def calc_IG_mult_samples():
    input_lst = []
    weights_lst = []
    for i in range(len(test_data)):
        # defining model input tensors
        input1 = test_data[i]
        input1 = np.expand_dims(input1, axis=0)
        input1 = torch.tensor(input1).float()

        # defining baselines for each input tensor
        baseline1 = torch.rand(1, 4, 1, 1000)

        # defining and applying integrated gradients on ToyModel and the
        ig = IntegratedGradients(model)
        attributions, approximation_error = ig.attribute((input1),
                                                         baselines=(baseline1),
                                                         method='gausslegendre',
                                                         return_convergence_delta=True,
                                                         target=[0])

        three_d_tensor = attributions[0]
        three_d_tensor = torch.transpose(three_d_tensor, 0, 1)
        three_d_tensor = torch.transpose(three_d_tensor, 1, 2)

        three_d_input = input1.int()[0]
        three_d_input = torch.transpose(three_d_input, 0, 1)
        three_d_input = torch.transpose(three_d_input, 1, 2)


        # create list with input data on right dimentions (one-hot vectors)
        two_d_input = three_d_input[0]
        input_lst.append(two_d_input.numpy())

        # create list with attributions data on right dimentions
        two_d_input = three_d_tensor[0]
        weights_lst.append(two_d_input.numpy())

        print("visited IG function")

    # creates the required dict
    weight_dict = {'task': weights_lst}

    # return input_lst, weights_lst
    return input_lst, weight_dict
