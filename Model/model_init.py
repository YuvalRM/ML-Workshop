import torch
import numpy as np
import deepsea_model
from torch.utils.data import Dataset, DataLoader, TensorDataset

# hyper parameters
best_vloss = 1000000
batch_size = 100
epochs = 2

torch.cuda.empty_cache()

data_dir = "data_dir/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

print("started loading model\n")

# Load the data from the npy files
train_data, train_labels = np.lib.format.open_memmap(f"{data_dir}train_data.npy"), np.load(
    f"{data_dir}train_labels.npy")
valid_data, valid_labels = np.load(f"{data_dir}valid_data.npy"), np.load(f"{data_dir}valid_labels.npy")
test_data, test_labels = np.load(f"{data_dir}test_data.npy"), np.load(f"{data_dir}test_labels.npy")

print("loaded train")
train_data = np.transpose(train_data, (0, 2, 1))  # (4400000, 4, 1000)
train_data = np.expand_dims(train_data, axis=2)  # (4400000, 4, 1, 1000)

print("loaded valid")
valid_data = np.transpose(valid_data, (0, 2, 1))  # (8000, 4, 1000)
valid_data = np.expand_dims(valid_data, axis=2)  # (8000, 4, 1, 1000)

print("loaded test")
test_data = np.transpose(test_data, (0, 2, 1))  # (455024, 4, 1000)
test_data = np.expand_dims(test_data, axis=2)  # (455024, 4, 1, 1000)

# just for testing on CPU
if not torch.cuda.is_available():
    train_size = round(train_data.shape[0] * 0.1)
    valid_size = round(valid_data.shape[0] * 0.1)
    test_size = round(test_data.shape[0] * 0.1)
    train_data = train_data[:train_size]
    valid_data = valid_data[:valid_size]
    test_data = train_data[:test_size]
    train_labels = train_labels[:train_size]
    valid_labels = valid_labels[:valid_size]
    test_labels = train_labels[:test_size]

# Create TensorDatasets
train_dataset = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_labels).float())
valid_dataset = TensorDataset(torch.from_numpy(valid_data).float(), torch.from_numpy(valid_labels).float())
test_dataset = TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_labels).float())

# Create DataLoaders
training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

loss_fn = torch.nn.BCELoss()

model = deepsea_model.get_seqpred_model(load_weights=True)
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.08, momentum=0.9)

print("finished loading model\n")
