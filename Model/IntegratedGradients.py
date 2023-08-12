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
# defining model input tensors
input1=test_data[0]
input1=np.expand_dims(input1, axis=0)
input1 = torch.tensor(input1)

# defining baselines for each input tensor
baseline1 = torch.rand(1,4,1,1000)

print(input1.shape)
# defining and applying integrated gradients on ToyModel and the
ig = IntegratedGradients(model)
attributions, approximation_error = ig.attribute((input1),
                                                 baselines=(baseline1),
                                                 method='gausslegendre',
                                                 return_convergence_delta=True)