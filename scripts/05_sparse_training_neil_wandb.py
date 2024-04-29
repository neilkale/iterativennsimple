import torch
import pandas as pd
from icecream import ic
import timeit
import time
from datetime import datetime
import matplotlib.pyplot as plt
import wandb
import sys

from torchinfo import summary
from sklearn.model_selection import train_test_split

from iterativennsimple.SparseLinear import SparseLinear
from iterativennsimple.MaskedLinear import MaskedLinear

# import the linear layer from torch.nn
from torch.nn import Linear

# We manually set the seed to ensure that the results are reproducible
torch.manual_seed(0)

# Test if cuda is available and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

# Select the number of threads used for parallelization
# This is to make sure that the results are consistent, since you may
# run this on a computer with a different number of cores
num_threads = 1
torch.set_num_threads(num_threads)

# Base WandB configuration
base_config = {
    "project": "Sparse MNIST",
    "entity": "neil-kale",
    "config": {
        "learning_rate": 0.001,
        "max_epochs": 10,
        "max_time": 5,
        "batch_size": 100,
        "sparsity": 0.2,
        "log_every_n_epochs": 1
    }
}

# Generate a dynamic run name
def generate_run_name(model_type: str) -> str:
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f"experiment_{model_type}_{current_time}"
    return run_name

def init_wandb_run(model_name, **config_overrides):
    config = dict(base_config)  # Start with the base configuration
    config["name"] = model_name  # Set the run name to the model name
    config["config"].update(config_overrides)  # Update with model-specific configs
    
    wandb.init(**config)
    return wandb.config  # Return the config object for further use

### Load and preprocess the training data

# Turn a pandas dataframe into a pytorch tensor
def df_to_tensor(df):
    return torch.tensor(df.values, dtype=torch.float32)

def read_data(MAX_NUM_SAMPLES = float('inf')):
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
    max_num_samples = MAX_NUM_SAMPLES
    num_samples = min(max_num_samples, z_start_tensor.shape[0])
    z_start_tensor = z_start_tensor[:num_samples]
    z_target_tensor = z_target_tensor[:num_samples]

    mask = (z_start_tensor == z_target_tensor).all(axis=0)
    x_mask, y_mask = mask, ~mask
    num_in_features = int(x_mask.sum()) # 784
    num_out_features = int(y_mask.sum()) # 10

    return z_start_tensor, z_target_tensor, num_in_features, num_out_features

def main():

    SPARSITY, MAX_EPOCHS, MAX_TIME, BATCH_SIZE, LR, LOG_EVERY_N_EPOCHS = sys.argv[1:]
    SPARSITY, MAX_EPOCHS, MAX_TIME, BATCH_SIZE, LR, LOG_EVERY_N_EPOCHS = float(SPARSITY), int(MAX_EPOCHS), int(MAX_TIME), int(BATCH_SIZE), float(LR), int(LOG_EVERY_N_EPOCHS)
    base_config['config']['sparsity'] = SPARSITY
    base_config['config']['max_epochs'] = MAX_EPOCHS
    base_config['config']['max_time'] = MAX_TIME
    base_config['config']['batch_size'] = BATCH_SIZE
    base_config['config']['learning_rate'] = LR
    base_config['config']['log_every_n_epochs'] = LOG_EVERY_N_EPOCHS

    z_start_tensor, z_target_tensor, num_in_features, num_out_features = read_data()

    # Split the data into 70% train, 15% test, 15% validation.
    z_start_tensor_train, z_start_tensor_test, z_target_tensor_train, z_target_tensor_test = train_test_split(z_start_tensor, z_target_tensor, test_size = 0.3)
    z_start_tensor_test, z_start_tensor_val, z_target_tensor_test, z_target_tensor_val = train_test_split(z_start_tensor_test, z_target_tensor_test, test_size = 0.5)

    train_data = Data(z_start_tensor_train, z_target_tensor_train)
    test_data = Data(z_start_tensor_test, z_target_tensor_test)
    validation_data = Data(z_start_tensor_val, z_target_tensor_val)

    num_hidden_features = [num_in_features//2, 2*num_out_features]

    dense_model = DenseModel(num_in_features, num_hidden_features, num_out_features)
    # ic(dense_model)
    dense_summary = summary(dense_model, verbose=0)
    print()

    sparse_model = SparseModel(num_in_features, num_hidden_features, num_out_features, SPARSITY)
    # ic(sparse_model)
    sparse_summary = summary(sparse_model, verbose=0)
    print()

    low_rank_model = LowRankModel(num_in_features, num_hidden_features, num_out_features, SPARSITY)
    # ic(low_rank_model)
    low_rank_summary = summary(low_rank_model, verbose=0)
    print()

    dense_model_hyperparams = init_wandb_run(generate_run_name('dense'), model_type='dense')
    print("Dense Model:",end='\n')
    dense_train_losses, dense_val_losses, dense_times = train_model_with_stats_and_validation_wandb(dense_model, train_data, validation_data, num_in_features, num_out_features, BATCH_SIZE = BATCH_SIZE,
                                                                                              MAX_EPOCHS = MAX_EPOCHS, MAX_TIME = MAX_TIME, LR = LR, LOG_EVERY_N_EPOCHS=LOG_EVERY_N_EPOCHS)
    print()
    dense_test_loss = test_model(dense_model, test_data, num_in_features, num_out_features)
    wandb.log({'test_loss':dense_test_loss})
    wandb.log({'num_parameters':dense_summary.trainable_params})
    wandb.finish()

    sparse_model_hyperparams = init_wandb_run(generate_run_name('sparse'), model_type='sparse')
    print("Sparse Model:",end='\n')
    sparse_train_losses, sparse_val_losses, sparse_times = train_model_with_stats_and_validation_wandb(sparse_model, train_data, validation_data, num_in_features, num_out_features, BATCH_SIZE = BATCH_SIZE,
                                                                                              MAX_EPOCHS = MAX_EPOCHS, MAX_TIME = MAX_TIME, LR = LR, LOG_EVERY_N_EPOCHS=LOG_EVERY_N_EPOCHS)
    print()
    sparse_test_loss = test_model(sparse_model, test_data, num_in_features, num_out_features)
    wandb.log({'test_loss':sparse_test_loss})
    wandb.log({'num_parameters':sparse_summary.trainable_params})
    wandb.finish()

    low_rank_model_hyperparams = init_wandb_run(generate_run_name('low_rank'), model_type='low_rank')
    print("Low Rank Model:",end='\n')
    low_rank_train_losses, low_rank_val_losses, low_rank_times = train_model_with_stats_and_validation_wandb(low_rank_model, train_data, validation_data, num_in_features, num_out_features, BATCH_SIZE = BATCH_SIZE,
                                                                                              MAX_EPOCHS = MAX_EPOCHS, MAX_TIME = MAX_TIME, LR = LR, LOG_EVERY_N_EPOCHS=LOG_EVERY_N_EPOCHS)
    print()
    low_rank_test_loss = test_model(low_rank_model, test_data, num_in_features, num_out_features)
    wandb.log({'test_loss':low_rank_test_loss})
    wandb.log({'num_parameters':low_rank_summary.trainable_params})
    wandb.finish()

    # start = time.time()
    # print("---- Test MSE Loss ----")
    # dense_test_loss = test_model(dense_model, test_data, num_in_features, num_out_features)
    # print(f'   Dense Model: {dense_test_loss:.2e}')
    # sparse_test_loss = test_model(sparse_model, test_data, num_in_features, num_out_features)
    # print(f'  Sparse Model: {sparse_test_loss:.2e}')
    # low_rank_test_loss = test_model(low_rank_model, test_data, num_in_features, num_out_features)
    # print(f'Low Rank Model: {low_rank_test_loss:.2e}')
    # print()
    # print(f'Computing test loss took {time.time() - start:.4f}  seconds')
    # print()

    # df = pd.DataFrame([dense_train_losses, dense_val_losses, dense_times, sparse_train_losses, sparse_val_losses, sparse_times, low_rank_train_losses, low_rank_val_losses, low_rank_times], 
    #                   index = ['dense_train_losses', 'dense_val_losses', 'dense_times', 'sparse_train_losses', 'sparse_val_losses', 'sparse_times', 'low_rank_train_losses', 'low_rank_val_losses', 'low_rank_times']).T
    
    # print(df.head())
    # print(df.shape)
    # print()

    # df.to_csv('sparse_training_data.csv',index = False)

    # fig = plt.figure()
    # fig.suptitle('MNIST Results at sparsity=' + str(SPARSITY))


    # ax1 = fig.add_subplot(221)
    # ax1.plot(df['dense_times'],df['dense_train_losses'], c='b', marker = 's', label = 'Dense')
    # ax1.plot(df['sparse_times'],df['sparse_train_losses'], c='r', marker = 'o', label = 'Sparse')
    # ax1.plot(df['low_rank_times'],df['low_rank_train_losses'], c='g', marker = '^', label = 'Low Rank')
    # # ax1.legend(loc = 'upper left')
    # ax1.title.set_text('Train Loss vs Time')
    # plt.yscale('log')

    # ax2 = fig.add_subplot(222, sharey=ax1)
    # epochs = range(len(df))
    # ax2.plot(epochs,df['dense_train_losses'], c='b', marker = 's', label = 'Dense')
    # ax2.plot(epochs,df['sparse_train_losses'], c='r', marker = 'o', label = 'Sparse')
    # ax2.plot(epochs,df['low_rank_train_losses'], c='g', marker = '^', label = 'Low Rank')
    # # ax2.legend(loc = 'upper left')
    # ax2.title.set_text('Train Loss vs Epoch')
    # plt.yscale('log')

    # ax3 = fig.add_subplot(223)
    # ax3.plot(df['dense_times'],df['dense_val_losses'], c='b', marker = 's', label = 'Dense')
    # ax3.plot(df['sparse_times'],df['sparse_val_losses'], c='r', marker = 'o', label = 'Sparse')
    # ax3.plot(df['low_rank_times'],df['low_rank_val_losses'], c='g', marker = '^', label = 'Low Rank')
    # # ax3.legend(loc = 'upper left')
    # ax3.title.set_text('Validation Loss vs Time')
    # plt.yscale('log')

    # ax4 = fig.add_subplot(224, sharey=ax3)
    # ax4.plot(epochs,df['dense_val_losses'], c='b', marker = 's', label = 'Dense')
    # ax4.plot(epochs,df['sparse_val_losses'], c='r', marker = 'o', label = 'Sparse')
    # ax4.plot(epochs,df['low_rank_val_losses'], c='g', marker = '^', label = 'Low Rank')
    # # ax4.legend(loc = 'upper left')
    # ax4.title.set_text('Validation Loss vs Epoch')
    # plt.yscale('log')

    # handles, labels = ax1.get_legend_handles_labels()
    # labels = [f'Dense (n_params = {dense_summary.trainable_params})', f'Sparse (n_params = {sparse_summary.trainable_params})', f'Low Rank (n_params = {low_rank_summary.trainable_params})']
    # fig.legend(handles, labels, loc='lower center', ncol=3)
    # plt.show()

# Create a dense linear model
class DenseModel(torch.nn.Module):

    def __init__(self, num_in_features, num_hidden_features, num_out_features):
        super(DenseModel, self).__init__()

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

# Create a sparse linear model
class SparseModel(torch.nn.Module):

    def __init__(self, num_in_features, num_hidden_features, num_out_features, sparsity=1):
        super(SparseModel, self).__init__()

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

# Create a low rank layer for use in a model
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

# Create a low rank linear model
class LowRankModel(torch.nn.Module):

    def __init__(self, num_in_features, num_hidden_features, num_out_features, sparsity=1):
        super(LowRankModel, self).__init__()

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

# A dataloader which returns a batch of start and target data
class Data(torch.utils.data.Dataset):
    def __init__(self, z_start, z_target):
        self.z_start = z_start
        self.z_target = z_target
    def __len__(self):
        return len(self.z_start)
    def __getitem__(self, idx):
        return self.z_start[idx], self.z_target[idx]

# def train_model(model, train_data, num_in_features, num_out_features, MAX_EPOCHS = 100, LR = 0.001):
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)

#     criterion = torch.nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     max_epochs = MAX_EPOCHS
#     last_loss = 10**9

#     # Train the model
#     for epoch in range(max_epochs):
#         for batch_idx, (start, target) in enumerate(train_loader):
#             optimizer.zero_grad()

#             loss = 0.0
#             out = model(start[:,0:num_in_features])

#             loss = criterion(out, target[:, num_in_features:num_in_features+num_out_features])
#             loss.backward()

#             optimizer.step()
#         if epoch % 10 == 0:
#             print(f'Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}')
#             # assert loss.item() < last_loss
#             last_loss = loss.item()

# def train_model_with_stats(model, train_data, num_in_features, num_out_features, MAX_EPOCHS = 100, LR = 0.001, MAX_TIME = 20, LOG_EVERY_N_EPOCHS = 1):
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)

#     criterion = torch.nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     max_epochs = MAX_EPOCHS
#     last_loss = 10**9

#     # Collect info
#     times = []
#     losses = []

#     start_time = time.time()

#     # Train the model
#     epoch = 0
#     while (epoch < max_epochs) and (time.time() - start_time < MAX_TIME):

#         total_epoch_loss = 0

#         for batch_idx, (start, target) in enumerate(train_loader):
#             optimizer.zero_grad()

#             loss = 0.0
#             out = model(start[:,0:num_in_features])

#             loss = criterion(out, target[:, num_in_features:num_in_features+num_out_features])
#             loss.backward()
#             total_epoch_loss += loss.item()

#             optimizer.step()

#         if epoch % 10 == 0:
#             print(f'Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}')
#             # assert loss.item() < last_loss
#             last_loss = loss.item()
        
#         if epoch % LOG_EVERY_N_EPOCHS == 0:
#             train_time = time.time()-start_time
#             losses.append(total_epoch_loss)
#             times.append(train_time)
        
#         epoch += 1
    
#     return losses, times

# def train_model_with_stats_and_validation(model, train_data, validation_data, num_in_features, num_out_features, 
#                                           MAX_EPOCHS = 100, LR = 0.001, MAX_TIME = 20, LOG_EVERY_N_EPOCHS = 1,
#                                           BATCH_SIZE = 100):
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

#     criterion = torch.nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     max_epochs = MAX_EPOCHS
#     last_loss = 10**9

#     # Collect info
#     times = []
#     train_losses = []
#     val_losses = []

#     start_time = time.time()

#     # Train the model
#     epoch = 0
#     while (epoch < max_epochs) and (time.time() - start_time < MAX_TIME):

#         total_epoch_loss = 0

#         for batch_idx, (start, target) in enumerate(train_loader):
#             optimizer.zero_grad()

#             loss = 0.0
#             out = model(start[:,0:num_in_features])

#             loss = criterion(out, target[:, num_in_features:num_in_features+num_out_features])
#             loss.backward()
#             total_epoch_loss += loss.item()

#             optimizer.step()

#         if epoch % 10 == 0:
#             print(f'Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}')
#             # assert loss.item() < last_loss
#             last_loss = loss.item()
        
#         if epoch % LOG_EVERY_N_EPOCHS == 0:
#             train_time = time.time()-start_time
#             train_losses.append(total_epoch_loss)
#             times.append(train_time)
#             val_loss = test_model(model, validation_data, num_in_features, num_out_features)
#             val_losses.append(val_loss)
        
#         epoch += 1
    
#     return train_losses, val_losses, times

def train_model_with_stats_and_validation_wandb(model, train_data, validation_data, num_in_features, num_out_features, 
                                          MAX_EPOCHS = 100, LR = 0.001, MAX_TIME = 20, LOG_EVERY_N_EPOCHS = 1,
                                          BATCH_SIZE = 100):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    max_epochs = MAX_EPOCHS
    last_loss = 10**9

    # Collect info
    times = []
    train_losses = []
    val_losses = []

    start_time = time.time()

    # Train the model
    epoch = 0
    while (epoch < max_epochs) and (time.time() - start_time < MAX_TIME):

        total_epoch_loss = 0

        for batch_idx, (start, target) in enumerate(train_loader):
            optimizer.zero_grad()

            loss = 0.0
            out = model(start[:,0:num_in_features])

            loss = criterion(out, target[:, num_in_features:num_in_features+num_out_features])
            loss.backward()
            total_epoch_loss += loss.item()

            optimizer.step()
        total_epoch_loss /= len(train_data)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}')
            # assert loss.item() < last_loss
            last_loss = loss.item()
        
        if epoch % LOG_EVERY_N_EPOCHS == 0:
            train_time = time.time()-start_time
            train_losses.append(total_epoch_loss)
            times.append(train_time)
            val_loss = test_model(model, validation_data, num_in_features, num_out_features)
            val_loss /= len(validation_data)
            val_losses.append(val_loss)
            print(f"Logging Epoch: {epoch}, Train Time: {train_time}, Loss: {total_epoch_loss}")

            wandb.define_metric("time")
            wandb.define_metric("val_loss_vs_time", step_metric = 'time')
            wandb.define_metric("train_loss_vs_time", step_metric = 'time')

            log_dict = {
                "time": train_time,
                "train_loss_per_epoch" : total_epoch_loss,
                "val_loss_per_epoch" : val_loss,
                "train_loss_vs_time" : total_epoch_loss,
                "val_loss_vs_time" : val_loss
            }

            wandb.log(log_dict)
            # wandb.log({"train_loss_vs_time": total_epoch_loss}, step=int(train_time*1000))
            # wandb.log({"val_loss_vs_time": val_loss}, step=int(train_time*1000))

            # wandb.log({"train_loss_per_epoch": total_epoch_loss}, step=epoch)
            # wandb.log({"val_loss_per_epoch": val_loss}, step=epoch)
        
        epoch += 1
    
    return train_losses, val_losses, times

def test_model(model, test_data, num_in_features, num_out_features):
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True)
    criterion = torch.nn.MSELoss()
    
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (start, target) in enumerate(test_loader):
            out = model(start[:,0:num_in_features])
            test_loss += criterion(out, target[:, num_in_features:num_in_features+num_out_features]).item()
    
    return test_loss/len(test_data)

if __name__ == "__main__":

    main()