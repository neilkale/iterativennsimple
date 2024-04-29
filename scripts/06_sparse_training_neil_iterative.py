####### IMPORTS ###############################################################################################################
import torch
import pandas as pd
import numpy as np

# Data processing imports
from sklearn.model_selection import train_test_split

# Model imports
from iterativennsimple.Sequential2D import Sequential2D, Identity
from iterativennsimple.Sequential1D import Sequential1D
from iterativennsimple.SparseLinear import SparseLinear
from torch.nn import Linear

# System imports
import os
import sys

# Debugging and unit testing imports
from icecream import ic
from torchinfo import summary
from datetime import datetime
from time import time
import wandb

# Model evaluation imports
from torchmetrics.classification import MulticlassAccuracy

####### SET UP GPU ###############################################################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_threads = 1
torch.set_num_threads(num_threads)

####### GLOBAL VARIABLES ###############################################################################################################

SPARSITY = 0.2
HIDDEN_SIZES = [20, 20]
MAX_NUM_SAMPLES = np.Inf
VAL_PCT, TEST_PCT = 0.15, 0.15

LEARNING_RATE = 0.001
MAX_EPOCHS = 10000
MAX_TIME = 20
BATCH_SIZE = 100
LOG_EVERY_N_EPOCHS = 10
CRITERION = torch.nn.MSELoss()

VERBOSE = 1

####### MAIN ###############################################################################################################

def main():
    # This is based on notebooks/4-rcp-MLP.ipynb
    
    ##### READ COMMAND LINE ARGUMENTS #####
    global SPARSITY
    if len(sys.argv) > 1: SPARSITY = float(sys.argv[1])
    if VERBOSE: ic(SPARSITY)

    ##### SET UP THE MODEL #####
    # Read the data
    features_list, data = read_data(HIDDEN_SIZES, MAX_NUM_SAMPLES)
    if VERBOSE: ic(features_list)

    # Split the data into train/test/validation sets
    train_data, val_data, test_data = split_data(data, val_pct = VAL_PCT, test_pct = TEST_PCT)
    if VERBOSE: ic(len(train_data))
    if VERBOSE: ic(len(val_data))
    if VERBOSE: ic(len(test_data))

    models = [SparseIterativeMLP, SparseIterativeNN, SparseMLP, DenseIterativeMLP, DenseIterativeNN, DenseMLP, LowRankMLP, LowRankIterativeNN, LowRankIterativeMLP]

    # Create and train each model.
    for model in models:
        model = model(features_list, SPARSITY)
        initialize_wandb(model)
        train_model(model, train_data, features_list, is_iterative=model.is_iterative, validation_data = val_data)
        test_loss, test_accuracy = get_loss_and_accuracy(model, test_data, features_list, is_iterative=model.is_iterative)
        wandb.log({'test_loss':test_loss, 'test_accuracy':test_accuracy})
        wandb.finish()
        print()
    
####### MODEL CLASSES ###############################################################################################################
            
##### SPARSE MLP as INN #####
    # This model is a basic feedforward neural network with SparseLinear blocks at each step, except for a single
    # dense layer at the end. However, the model is implemented as an iterative neural network.
class SparseIterativeMLP(torch.nn.Module):

    def __init__(self, features_list, sparsity=1):
        super(SparseIterativeMLP, self).__init__()
        self.model_name = 'SparseIterativeMLP'
        self.is_iterative = True

        num_in_features, num_hidden_features, num_out_features = features_list[0], features_list[1:-1], features_list[-1]

        I = Identity(in_features=num_in_features, out_features=num_in_features)
        linear = torch.nn.ModuleList()
        prev_size = num_in_features
        for hidden_size in num_hidden_features:
            layer = SparseLinear.from_singleBlock(col_size=prev_size, row_size=hidden_size,  
                                    block_type='R='+str(sparsity), initialization_type='G=0.0,0.001',
                                    optimized_implementation=True)
            layer = Sequential1D(layer, torch.nn.ReLU(), 
                                      in_features=prev_size, out_features=hidden_size)
            linear.append(layer)
            prev_size = hidden_size
        
        final_linear = Linear(in_features=prev_size, out_features=num_out_features,
                     bias=False,
                     dtype=torch.float32,
                     device=device)
        final_linear = Sequential1D(final_linear, torch.nn.Softmax(dim=1), 
                                      in_features=prev_size, out_features=num_out_features)
        
        blocks = np.empty((len(num_hidden_features)+2,
                           len(num_hidden_features)+2),
                           dtype = object)
        blocks[0,0] = I
        for i in range(len(linear)):
            blocks[i+1,i] = linear[i]
        blocks[-1,-2] = final_linear

        layer_size_list = [num_in_features] + num_hidden_features + [num_out_features]
        self.map = Sequential2D(
            in_features_list=layer_size_list,
            out_features_list=layer_size_list,
            blocks=transpose_blocks(blocks)
        )
   
    def forward(self, x):
        return self.map(x)

##### SPARSE INN #####
    # This model is a 'classic' INN where all blocks are trainable SparseLinears, except for an identity to preserve the inputs
    # and a single dense layer at the end. The final row has Softmax nonlinearity, the rest is ReLU.
class SparseIterativeNN(torch.nn.Module):

    def __init__(self, features_list, sparsity=1):
        super(SparseIterativeNN, self).__init__()
        self.model_name = 'SparseIterativeNN'
        self.is_iterative = True

        num_in_features = features_list[0]
        num_out_features = features_list[-1]

        blocks = np.empty((len(features_list),
                           len(features_list)),
                           dtype = object)

        I = Identity(in_features=num_in_features, out_features=num_in_features)
        blocks[0,0] = I

        for i in range(len(blocks)):
            for j in range(len(blocks[0])):
                if (i == 0): pass # Skip the first row
                else: # Set every other block to a SparseLinear
                    cols = features_list[j]
                    rows = features_list[i]
                    layer = SparseLinear.from_singleBlock(col_size=cols, row_size=rows,  
                                    block_type='R='+str(sparsity), initialization_type='G=0.0,0.001',
                                    optimized_implementation=True)
                    if (i == len(blocks)): # No non-linearity in the final row.
                        layer = Sequential1D(layer, torch.nn.Softmax(dim=1),
                                            in_features=cols, out_features=rows)
                    else: 
                        layer = Sequential1D(layer, torch.nn.ReLU(), 
                                            in_features=cols, out_features=rows)
                    blocks[i][j] = layer
        
        final_linear = Linear(in_features=features_list[-2], out_features=num_out_features,
                     bias=False,
                     dtype=torch.float32,
                     device=device)
        final_linear = Sequential1D(final_linear, torch.nn.Softmax(dim=1), 
                                      in_features=features_list[-2], out_features=num_out_features)
        blocks[-1,-2] = final_linear

        self.map = Sequential2D(
            in_features_list=features_list,
            out_features_list=features_list,
            blocks=transpose_blocks(blocks)
        )
   
    def forward(self, x):
        return self.map(x)

##### SPARSE MLP #####
    # This model is a basic feedforward neural network with SparseLinear blocks at each step, except for a single
    # dense layer at the end. This model is NOT an iterative neural network.
class SparseMLP(torch.nn.Module):

    def __init__(self, features_list, sparsity=1):
        super(SparseMLP, self).__init__()
        self.model_name = 'SparseMLP'
        self.is_iterative = False


        num_in_features, num_hidden_features, num_out_features = features_list[0], features_list[1:-1], features_list[-1]

        self.linear = torch.nn.ModuleList()
        prev_size = num_in_features
        for hidden_size in num_hidden_features:
            layer = SparseLinear.from_singleBlock(col_size=prev_size, row_size=hidden_size,  
                                    block_type='R='+str(sparsity), initialization_type='G=0.0,0.001',
                                    optimized_implementation=True)
            self.linear.append(layer)
            prev_size = hidden_size
        
        self.activation = torch.nn.ReLU()
        self.final_linear = Linear(in_features=prev_size, out_features=num_out_features,
                     bias=False,
                     dtype=torch.float32,
                     device=device)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        for linear in self.linear:
            x = linear(x)
            x = self.activation(x)
        x = self.final_linear(x)
        x = self.softmax(x)
        return x

##### DENSE MLP #####
    # This model is a basic feedforward neural network with torch.nn.Linear blocks at each step. This model is NOT an iterative neural network.
class DenseMLP(torch.nn.Module):

    def __init__(self, features_list, sparsity=1):
        super(DenseMLP, self).__init__()
        self.model_name = 'DenseMLP'
        self.is_iterative = False


        num_in_features, num_hidden_features, num_out_features = features_list[0], features_list[1:-1], features_list[-1]

        self.linear = torch.nn.ModuleList()
        prev_size = num_in_features
        for hidden_size in num_hidden_features:
            layer = Linear(in_features=prev_size, out_features=hidden_size,
                     bias=False,
                     dtype=torch.float32,
                     device=device)
            self.linear.append(layer)
            prev_size = hidden_size
        
        self.activation = torch.nn.ReLU()
        self.final_linear = Linear(in_features=prev_size, out_features=num_out_features,
                     bias=False,
                     dtype=torch.float32,
                     device=device)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        for linear in self.linear:
            x = linear(x)
            x = self.activation(x)
        x = self.final_linear(x)
        x = self.softmax(x)
        return x

##### DENSE INN #####
    # This model is a 'classic' INN where all blocks are trainable torch.nn.Linear blocks, except for an identity to preserve the inputs.
    # The final row has Softmax nonlinearity, the rest is ReLU.
class DenseIterativeNN(torch.nn.Module):

    def __init__(self, features_list, sparsity=1):
        super(DenseIterativeNN, self).__init__()
        self.model_name = 'DenseIterativeNN'
        self.is_iterative = True

        num_in_features = features_list[0]
        num_out_features = features_list[-1]

        blocks = np.empty((len(features_list),
                           len(features_list)),
                           dtype = object)

        I = Identity(in_features=num_in_features, out_features=num_in_features)
        blocks[0,0] = I

        for i in range(len(blocks)):
            for j in range(len(blocks[0])):
                if (i == 0): pass # Skip the first row
                else: # Set every other block to a SparseLinear
                    cols = features_list[j]
                    rows = features_list[i]
                    layer = Linear(in_features=cols, out_features=rows,
                                    bias=False,
                                    dtype=torch.float32,
                                    device=device)
                    if (i == len(blocks)): # No non-linearity in the final row.
                        layer = Sequential1D(layer, torch.nn.Softmax(dim=1),
                                            in_features=cols, out_features=rows)
                    else: 
                        layer = Sequential1D(layer, torch.nn.ReLU(), 
                                            in_features=cols, out_features=rows)
                    blocks[i][j] = layer
        
        final_linear = Linear(in_features=features_list[-2], out_features=num_out_features,
                     bias=False,
                     dtype=torch.float32,
                     device=device)
        final_linear = Sequential1D(final_linear, torch.nn.Softmax(dim=1), 
                                      in_features=features_list[-2], out_features=num_out_features)
        blocks[-1,-2] = final_linear

        self.map = Sequential2D(
            in_features_list=features_list,
            out_features_list=features_list,
            blocks=transpose_blocks(blocks)
        )
   
    def forward(self, x):
        return self.map(x)

##### DENSE MLP as INN #####
    # This model is a basic feedforward neural network with SparseLinear blocks at each step, except for a single
    # dense layer at the end. However, the model is implemented as an iterative neural network.
class DenseIterativeMLP(torch.nn.Module):

    def __init__(self, features_list, sparsity=1):
        super(DenseIterativeMLP, self).__init__()
        self.model_name = 'DenseIterativeMLP'
        self.is_iterative = True

        num_in_features, num_hidden_features, num_out_features = features_list[0], features_list[1:-1], features_list[-1]

        I = Identity(in_features=num_in_features, out_features=num_in_features)
        linear = torch.nn.ModuleList()
        prev_size = num_in_features
        for hidden_size in num_hidden_features:
            layer = Linear(in_features=prev_size, out_features=hidden_size,
                                    bias=False,
                                    dtype=torch.float32,
                                    device=device)
            layer = Sequential1D(layer, torch.nn.ReLU(), 
                                      in_features=prev_size, out_features=hidden_size)
            linear.append(layer)
            prev_size = hidden_size
        
        final_linear = Linear(in_features=prev_size, out_features=num_out_features,
                     bias=False,
                     dtype=torch.float32,
                     device=device)
        final_linear = Sequential1D(final_linear, torch.nn.Softmax(dim=1), 
                                      in_features=prev_size, out_features=num_out_features)
        
        blocks = np.empty((len(num_hidden_features)+2,
                           len(num_hidden_features)+2),
                           dtype = object)
        blocks[0,0] = I
        for i in range(len(linear)):
            blocks[i+1,i] = linear[i]
        blocks[-1,-2] = final_linear

        layer_size_list = [num_in_features] + num_hidden_features + [num_out_features]
        self.map = Sequential2D(
            in_features_list=layer_size_list,
            out_features_list=layer_size_list,
            blocks=transpose_blocks(blocks)
        )
   
    def forward(self, x):
        return self.map(x)

##### LOW RANK MLP #####
    # This model is a basic feedforward neural network with torch.nn.Linear blocks at each step. This model is NOT an iterative neural network.
class LowRankMLP(torch.nn.Module):

    def __init__(self, features_list, sparsity=1):
        super(LowRankMLP, self).__init__()
        self.model_name = 'LowRankMLP'
        self.is_iterative = False


        num_in_features, num_hidden_features, num_out_features = features_list[0], features_list[1:-1], features_list[-1]

        self.linear = torch.nn.ModuleList()
        prev_size = num_in_features
        for hidden_size in num_hidden_features:
            hidden_rank = max(1,round(prev_size*hidden_size*sparsity / (prev_size + hidden_size)))
            layer = LowRankLayer(prev_size, hidden_rank, hidden_size)
            self.linear.append(layer)
            prev_size = hidden_size
        
        self.activation = torch.nn.ReLU()
        self.final_linear = Linear(in_features=prev_size, out_features=num_out_features,
                     bias=False,
                     dtype=torch.float32,
                     device=device)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        for linear in self.linear:
            x = linear(x)
            x = self.activation(x)
        x = self.final_linear(x)
        x = self.softmax(x)
        return x

##### LOW RANK INN #####
    # This model is a 'classic' INN where all blocks are trainable torch.nn.Linear blocks, except for an identity to preserve the inputs.
    # The final row has Softmax nonlinearity, the rest is ReLU.
class LowRankIterativeNN(torch.nn.Module):

    def __init__(self, features_list, sparsity=1):
        super(LowRankIterativeNN, self).__init__()
        self.model_name = 'LowRankIterativeNN'
        self.is_iterative = True

        num_in_features = features_list[0]
        num_out_features = features_list[-1]

        blocks = np.empty((len(features_list),
                           len(features_list)),
                           dtype = object)

        I = Identity(in_features=num_in_features, out_features=num_in_features)
        blocks[0,0] = I

        for i in range(len(blocks)):
            for j in range(len(blocks[0])):
                if (i == 0): pass # Skip the first row
                else: # Set every other block to a SparseLinear
                    cols = features_list[j]
                    rows = features_list[i]
                    hidden_rank = max(1,round(cols*rows*sparsity / (cols + rows)))
                    layer = LowRankLayer(cols, hidden_rank, rows)
                    if (i == len(blocks)): # No non-linearity in the final row.
                        layer = Sequential1D(layer, torch.nn.Softmax(dim=1),
                                            in_features=cols, out_features=rows)
                    else: 
                        layer = Sequential1D(layer, torch.nn.ReLU(), 
                                            in_features=cols, out_features=rows)
                    blocks[i][j] = layer
        
        final_linear = Linear(in_features=features_list[-2], out_features=num_out_features,
                     bias=False,
                     dtype=torch.float32,
                     device=device)
        final_linear = Sequential1D(final_linear, torch.nn.Softmax(dim=1), 
                                      in_features=features_list[-2], out_features=num_out_features)
        blocks[-1,-2] = final_linear

        self.map = Sequential2D(
            in_features_list=features_list,
            out_features_list=features_list,
            blocks=transpose_blocks(blocks)
        )
   
    def forward(self, x):
        return self.map(x)

##### DENSE MLP as INN #####
    # This model is a basic feedforward neural network with SparseLinear blocks at each step, except for a single
    # dense layer at the end. However, the model is implemented as an iterative neural network.
class LowRankIterativeMLP(torch.nn.Module):

    def __init__(self, features_list, sparsity=1):
        super(LowRankIterativeMLP, self).__init__()
        self.model_name = 'LowRankIterativeMLP'
        self.is_iterative = True

        num_in_features, num_hidden_features, num_out_features = features_list[0], features_list[1:-1], features_list[-1]

        I = Identity(in_features=num_in_features, out_features=num_in_features)
        linear = torch.nn.ModuleList()
        prev_size = num_in_features
        for hidden_size in num_hidden_features:
            hidden_rank = max(1,round(prev_size*hidden_size*sparsity / (prev_size + hidden_size)))
            layer = LowRankLayer(prev_size, hidden_rank, hidden_size)
            layer = Sequential1D(layer, torch.nn.ReLU(), 
                                      in_features=prev_size, out_features=hidden_size)
            linear.append(layer)
            prev_size = hidden_size
        
        final_linear = Linear(in_features=prev_size, out_features=num_out_features,
                     bias=False,
                     dtype=torch.float32,
                     device=device)
        final_linear = Sequential1D(final_linear, torch.nn.Softmax(dim=1), 
                                      in_features=prev_size, out_features=num_out_features)
        
        blocks = np.empty((len(num_hidden_features)+2,
                           len(num_hidden_features)+2),
                           dtype = object)
        blocks[0,0] = I
        for i in range(len(linear)):
            blocks[i+1,i] = linear[i]
        blocks[-1,-2] = final_linear

        layer_size_list = [num_in_features] + num_hidden_features + [num_out_features]
        self.map = Sequential2D(
            in_features_list=layer_size_list,
            out_features_list=layer_size_list,
            blocks=transpose_blocks(blocks)
        )
   
    def forward(self, x):
        return self.map(x)

####### TRAINING LOOPS ###############################################################################################################
    
def train_model(map, train_data, features_list, is_iterative: bool, validation_data = None):
    # Read the training data into a data loader.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # Define the loss function and optimizer
    criterion = CRITERION
    optimizer = torch.optim.Adam(map.parameters(), lr=LEARNING_RATE)
    iterations = len(features_list)-1
    
    # Set up the loss tracking
    times, train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], [], []
    epoch, elapsed_time, start_time = -1, 0, time()

    # Train the model until epoch limit or time limit reached.
    while epoch < MAX_EPOCHS and elapsed_time < MAX_TIME:

        # Track the loss and accuracy over all minibatches to average at the end.
        batch_train_losses, batch_train_accuracies = [],[]

        for batch_idx, (start, target) in enumerate(train_loader):
            optimizer.zero_grad()

            # Compute the output vector and backpropagate the resulting gradients.
            mapped = start
            if is_iterative:
                for i in range(iterations):
                    mapped = map(mapped)
                yhat, y = mapped[:, -features_list[-1]:], target[:, -features_list[-1]:]
            else:
                mapped = map(mapped[:, 0:features_list[0]])
                yhat, y = mapped, target[:, -features_list[-1]:]
            loss = criterion(yhat, y)
            loss.backward()
            batch_train_losses.append(loss.item())

            # Compute and save the training accuracies for each batch.
            metric = MulticlassAccuracy(num_classes=features_list[-1])
            accuracy = metric(torch.max(yhat, 1)[1],torch.max(y, 1)[1])
            batch_train_accuracies.append(accuracy)

            optimizer.step()

        # Compute the average accuracy and the loss per input.
        train_accuracy = sum(batch_train_accuracies)/len(batch_train_accuracies)
        train_loss = sum(batch_train_losses)/len(train_data)

        if epoch % LOG_EVERY_N_EPOCHS == 0:
            if VERBOSE: print(f'Epoch {epoch}, Loss {train_loss:.3e}, Accuracy {train_accuracy:.3f}')

            # Compute and save the validation accuracy and validation loss per input.
            if validation_data:
                val_loss, val_accuracy = get_loss_and_accuracy(map, validation_data, features_list, is_iterative)

            save_to_wandb(train_loss, train_accuracy, val_loss, val_accuracy, epoch, elapsed_time)

        # Update the epoch number and elapsed time
        epoch, elapsed_time = epoch+1, time()-start_time

    wandb.log({'num_epochs':epoch , 'time':elapsed_time})

####### HELPER FUNCTIONS ###############################################################################################################

def transpose_blocks(blocks):
    return [[blocks[j][i] for j in range(len(blocks))] for i in range(len(blocks[0]))]

# Turn a pandas dataframe into a pytorch tensor
def df_to_tensor(df):
    return torch.tensor(df.values, dtype=torch.float32)

# a dataloader which returns a batch of start and target data
class Data(torch.utils.data.Dataset):
    def __init__(self, z_start, z_target):
        self.z_start = z_start
        self.z_target = z_target
    def __len__(self):
        return len(self.z_start)
    def __getitem__(self, idx):
        return self.z_start[idx], self.z_target[idx]

# Read the data from a Parquet file into Pytorch tensor.
def read_data(hidden_sizes, max_num_samples = np.Inf):
    # Read the start data
    z_start = pd.read_parquet('tests/MNIST_small_start.parquet')
    # Read the target data
    z_target = pd.read_parquet('tests/MNIST_small_target.parquet')

    # Data preprocessing
    z_start_tensor = df_to_tensor(z_start)
    z_target_tensor = df_to_tensor(z_target)

    # Only use the given number of samples
    num_samples = min(max_num_samples, z_start_tensor.shape[0])
    z_start_tensor = z_start_tensor[:num_samples]
    z_target_tensor = z_target_tensor[:num_samples]

    # Find masks to access input/output data
    mask = (z_start_tensor == z_target_tensor).all(axis=0)
    x_mask = mask
    y_mask = ~mask

    # Set hidden dimensions
    input_size = int(x_mask.sum())
    output_size = int(y_mask.sum())
    hidden_sizes = hidden_sizes

    h_idx = torch.arange(input_size, input_size+sum(hidden_sizes))
    y_idx = torch.arange(input_size+sum(hidden_sizes), input_size+sum(hidden_sizes)+output_size)

    zh_start_tensor = torch.cat((z_start_tensor[:, x_mask],
                                torch.zeros(z_start_tensor.shape[0], len(h_idx)), 
                                z_start_tensor[:, y_mask]), dim=1)
    zh_target_tensor = torch.cat((z_target_tensor[:, x_mask], 
                                torch.zeros(z_target_tensor.shape[0], len(h_idx)), 
                                z_target_tensor[:, y_mask]), dim=1)
    
    # Compile the tensors into a dataset and aggregate the features into a list.
    features_list  = [input_size] + hidden_sizes + [output_size]
    data = Data(zh_start_tensor, zh_target_tensor)

    return features_list, data

# Split the PyTorch Dataset into train/test/validation Datasets.
def split_data(data: Data, val_pct = 0, test_pct = 0):
    # If no split requested
    if (val_pct == 0 and test_pct == 0):
        return data, None, None
    
    # Split off the validation set
    z_start_tensor, z_target_tensor = data.z_start, data.z_target
    z_start_tensor_train, z_start_tensor_val, z_target_tensor_train, z_target_tensor_val = train_test_split(
        z_start_tensor, z_target_tensor, test_size = val_pct + test_pct)
    train_data = Data(z_start_tensor_train, z_target_tensor_train)
    val_data = Data(z_start_tensor_val, z_target_tensor_val)

    # If both test and validation are requested
    if (val_pct != 0 and test_pct != 0):
        test_pct_out_of_val = test_pct/(test_pct+val_pct)
        z_start_tensor_val, z_start_tensor_test, z_target_tensor_val, z_target_tensor_test = train_test_split(z_start_tensor_val, z_target_tensor_val, test_size = test_pct_out_of_val)
        test_data = Data(z_start_tensor_test, z_target_tensor_test)
        val_data = Data(z_start_tensor_val, z_target_tensor_val)
    else: test_data = None

    return train_data, val_data, test_data

# Create a low rank layer for use in the models
class LowRankLayer(torch.nn.Module):
    def __init__(self, prev_size, hidden_rank, hidden_size):
        super(LowRankLayer, self).__init__()

        self.layer_in = Linear(in_features=prev_size, out_features=hidden_rank,
                     bias=False,
                     dtype=torch.float32,
                     device=device)
        self.layer_out = Linear(in_features=hidden_rank, out_features=hidden_size,
                     bias=False,
                     dtype=torch.float32,
                     device=device)

    def forward(self, x):
        return self.layer_out(self.layer_in(x))

# Initialize the Weights and Biases logging
def initialize_wandb(model: torch.nn.Module):
    wandb.init(
        project = "Iterative MNIST",
        entity  = "neil-kale",
        name = generate_run_name(model),
        config  = {
            "model_type": model.model_name,
            "num_parameters": summary(model, verbose=VERBOSE).trainable_params,
            "is_iterative": model.is_iterative,

            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
            "max_time": MAX_TIME,
            "batch_size": BATCH_SIZE,
            "sparsity": SPARSITY,
            "log_every_n_epochs": LOG_EVERY_N_EPOCHS,
            "hidden_sizes": HIDDEN_SIZES
        }
    )

# Generate a unique descriptive name for a model.
def generate_run_name(model: torch.nn.Module) -> str:
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f"experiment_{model.model_name}_{current_time}"
    return run_name

# Get model loss and accuracy on a dataset without computing gradients.
def get_loss_and_accuracy(map, data, features_list, is_iterative: bool):
    data_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    criterion = CRITERION
    iterations = len(features_list)-1

    losses, accuracies = [], []

    # Compute the output vector with no gradients, 
    # and evaluate the loss and accuracy for each batch.
    with torch.no_grad():
        for batch_idx, (start, target) in enumerate(data_loader):
            mapped = start
            if is_iterative:
                for i in range(iterations):
                    mapped = map(mapped)
                yhat, y = mapped[:, -features_list[-1]:], target[:, -features_list[-1]:]
            else:
                mapped = map(mapped[:, 0:features_list[0]])
                yhat, y = mapped, target[:, -features_list[-1]:]
            loss = criterion(yhat, y)
            losses.append(loss.item())
            
            metric = MulticlassAccuracy(num_classes=features_list[-1])
            accuracy = metric(torch.max(yhat, 1)[1],torch.max(y, 1)[1])
            accuracies.append(accuracy)

    # Compute the average accuracy and the loss per input. 
    accuracy = sum(accuracies)/len(accuracies)
    loss = sum(losses)/len(data)

    return loss, accuracy

# Save the training and validation losses/accuracies to Weights and Biases.
def save_to_wandb(train_loss, train_accuracy, val_loss, val_accuracy, epoch, elapsed_time):
    #Define elapsed time as a custom W and B metric.
    wandb.define_metric("time")
    wandb.define_metric("val_loss_per_time", step_metric = 'time')
    wandb.define_metric("train_loss_per_time", step_metric = 'time')
    wandb.define_metric("val_accuracy_per_time", step_metric = 'time')
    wandb.define_metric("train_accuracy_per_time", step_metric = 'time')
    
    # Compile the statistics into a dictionary
    log_dict = {
                "time": elapsed_time,
                "train_loss_per_epoch"      : train_loss,
                "val_loss_per_epoch"        : val_loss,
                "train_loss_per_time"       : train_loss,
                "val_loss_per_time"         : val_loss,
                "train_accuracy_per_epoch"  : train_accuracy,
                "val_accuracy_per_epoch"    : val_accuracy,
                "train_accuracy_per_time"   : train_accuracy,
                "val_accuracy_per_time"     : val_accuracy                
    }

    # Log all the statistics in one log call.
    wandb.log(log_dict)

####### RUNNER CODE ###############################################################################################################

if __name__ == "__main__":
    main()