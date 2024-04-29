import torch
import pandas as pd
from icecream import ic
import timeit

from iterativennsimple.SparseLinear import SparseLinear
from iterativennsimple.MaskedLinear import MaskedLinear

# import the linear layer from torch.nn
from torch.nn import Linear

# We manually set the seed to ensure that the results are reproducible
# torch.manual_seed(0)

# Test if cuda is available and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

# Select the number of threads used for parallelization
# This is to make sure that the results are consistent, since you may
# run this on a computer with a different number of cores
num_threads = 1
torch.set_num_threads(num_threads)

### Load and preprocess the training data

# Turn a pandas dataframe into a pytorch tensor
def df_to_tensor(df):
    return torch.tensor(df.values, dtype=torch.float32)

# Read the start data
z_start = pd.read_parquet('tests/MNIST_small_start.parquet')
# Read the target data
z_target = pd.read_parquet('tests/MNIST_small_target.parquet')

# Data preprocessing
z_start_tensor = df_to_tensor(z_start)
z_target_tensor = df_to_tensor(z_target)

ic(z_start_tensor.shape)
ic(z_target_tensor.shape)

# Only use the given number of samples
max_num_samples = 500
num_samples = min(max_num_samples, z_start_tensor.shape[0])
z_start_tensor = z_start_tensor[:num_samples]
z_target_tensor = z_target_tensor[:num_samples]

mask = (z_start_tensor == z_target_tensor).all(axis=0)
x_mask = mask
y_mask = ~mask

num_in_features = int(x_mask.sum()) # 784
num_out_features = int(y_mask.sum()) # 10
num_hidden_features = [num_in_features//2, 2*num_out_features]

ic(num_in_features)
ic(num_hidden_features)
ic(num_out_features)

# Create a dense linear model
class DenseModel(torch.nn.Module):

    def __init__(self):
        super(DenseModel, self).__init__()
    
        self.linear1 = Linear(in_features=num_in_features, out_features=num_hidden_features[0],
                     bias=False,
                     dtype=torch.float32,
                     device=device)
        self.activation = torch.nn.ReLU()
        self.linear2 = Linear(in_features=num_hidden_features[0], out_features=num_hidden_features[1],
                     bias=False,
                     dtype=torch.float32,
                     device=device)
        self.linear3 = Linear(in_features=num_hidden_features[1], out_features=num_out_features,
                     bias=False,
                     dtype=torch.float32,
                     device=device)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x

dense_model = DenseModel()

ic(dense_model)

# Create a sparse linear model

sparsity = 0.2
sparsity_string = 'R='+str(sparsity)

ic(sparsity)

class SparseModel(torch.nn.Module):

    def __init__(self):
        super(SparseModel, self).__init__()
    
        self.linear1 = SparseLinear.from_singleBlock(col_size=num_in_features, row_size=num_hidden_features[0],  
                                    block_type=sparsity_string, initialization_type='G=0.0,0.001',
                                    optimized_implementation=True)
        self.activation = torch.nn.ReLU()
        self.linear2 = SparseLinear.from_singleBlock(col_size=num_hidden_features[0], row_size=num_hidden_features[1],  
                                    block_type=sparsity_string, initialization_type='G=0.0,0.001',
                                    optimized_implementation=True)
        self.linear3 = Linear(in_features=num_hidden_features[1], out_features=num_out_features,
                     bias=False,
                     dtype=torch.float32,
                     device=device)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x

sparse_model = SparseModel()

ic(sparse_model)

# Create a low-rank linear model
linear1_hidden_rank = max(1,round(sparse_model.linear1.number_of_trainable_parameters() / (num_in_features + num_hidden_features[0])))
linear2_hidden_rank = max(1,round(sparse_model.linear2.number_of_trainable_parameters() / (num_hidden_features[0] + num_hidden_features[1])))

class LowRankModel(torch.nn.Module):

    def __init__(self):
        super(LowRankModel, self).__init__()
    
        self.linear1a = Linear(in_features=num_in_features, out_features=linear1_hidden_rank,
                     bias=False,
                     dtype=torch.float32,
                     device=device)
        self.linear1b = Linear(in_features=linear1_hidden_rank, out_features=num_hidden_features[0],
                     bias=False,
                     dtype=torch.float32,
                     device=device)
        self.activation = torch.nn.ReLU()
        self.linear2a = Linear(in_features=num_hidden_features[0], out_features=linear2_hidden_rank,
                     bias=False,
                     dtype=torch.float32,
                     device=device)
        self.linear2b = Linear(in_features=linear2_hidden_rank, out_features=num_hidden_features[1],
                     bias=False,
                     dtype=torch.float32,
                     device=device)
        self.linear3 = Linear(in_features=num_hidden_features[1], out_features=num_out_features,
                     bias=False,
                     dtype=torch.float32,
                     device=device)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1a(x)
        x = self.linear1b(x)
        x = self.activation(x)
        x = self.linear2a(x)
        x = self.linear2b(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x

low_rank_model = LowRankModel()

ic(low_rank_model)

# Run the forward pass for both models

x = torch.randn((num_samples, num_in_features),
                dtype=torch.float32,
                device=device)

dense_output = dense_model(x)
sparse_output = sparse_model(x)
low_rank_output = low_rank_model(x)

ic(dense_output.shape)
ic(dense_output[0,:])
ic(sparse_output.shape)
ic(sparse_output[0,:])
ic(low_rank_output.shape)
ic(low_rank_output[0,:])

dense_number_of_parameters = sum(param.numel() for param in dense_model.parameters() if param.requires_grad)
sparse_number_of_parameters = sum(param.numel() for param in sparse_model.parameters() if param.requires_grad)
low_rank_number_of_parameters = sum(param.numel() for param in low_rank_model.parameters() if param.requires_grad)

ic(dense_number_of_parameters)
ic(sparse_number_of_parameters)
ic(low_rank_number_of_parameters)

### Train the dense model
print("Dense Model:")

# A dataloader which returns a batch of start and target data
class Data(torch.utils.data.Dataset):
    def __init__(self, z_start, z_target):
        self.z_start = z_start
        self.z_target = z_target
    def __len__(self):
        return len(self.z_start)
    def __getitem__(self, idx):
        return self.z_start[idx], self.z_target[idx]
    
train_data = Data(z_start_tensor, z_target_tensor)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(dense_model.parameters(), lr=0.001)

max_epochs = 100
last_loss = 10**9

# Train the model
for epoch in range(max_epochs):
    for batch_idx, (start, target) in enumerate(train_loader):
        optimizer.zero_grad()

        loss = 0.0
        out = dense_model(start[:,0:num_in_features])

        loss = criterion(out, target[:, num_in_features:num_in_features+num_out_features])
        loss.backward()

        optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}')
        # assert loss.item() < last_loss
        last_loss = loss.item()

### Train the low-rank model
print("Low Rank Model:")

# A dataloader which returns a batch of start and target data
class Data(torch.utils.data.Dataset):
    def __init__(self, z_start, z_target):
        self.z_start = z_start
        self.z_target = z_target
    def __len__(self):
        return len(self.z_start)
    def __getitem__(self, idx):
        return self.z_start[idx], self.z_target[idx]
    
train_data = Data(z_start_tensor, z_target_tensor)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(low_rank_model.parameters(), lr=0.001)

max_epochs = 100
last_loss = 10**9

# Train the model
for epoch in range(max_epochs):
    for batch_idx, (start, target) in enumerate(train_loader):
        optimizer.zero_grad()

        loss = 0.0
        out = low_rank_model(start[:,0:num_in_features])

        loss = criterion(out, target[:, num_in_features:num_in_features+num_out_features])
        loss.backward()

        optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}')
        # assert loss.item() < last_loss
        last_loss = loss.item()

### Train the sparse model
print("Sparse Model:")

# A dataloader which returns a batch of start and target data
class Data(torch.utils.data.Dataset):
    def __init__(self, z_start, z_target):
        self.z_start = z_start
        self.z_target = z_target
    def __len__(self):
        return len(self.z_start)
    def __getitem__(self, idx):
        return self.z_start[idx], self.z_target[idx]
    
train_data = Data(z_start_tensor, z_target_tensor)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(sparse_model.parameters(), lr=0.001)

max_epochs = 100
last_loss = 10**9

# Train the model
for epoch in range(max_epochs):
    for batch_idx, (start, target) in enumerate(train_loader):
        optimizer.zero_grad()

        loss = 0.0
        out = sparse_model(start[:,0:num_in_features])

        loss = criterion(out, target[:, num_in_features:num_in_features+num_out_features])
        loss.backward()

        optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}')
        # assert loss.item() < last_loss
        last_loss = loss.item()