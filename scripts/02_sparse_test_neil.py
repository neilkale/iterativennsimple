import torch
from icecream import ic
import timeit

import pandas as pd
import numpy as np

from iterativennsimple.SparseLinear import SparseLinear
from iterativennsimple.MaskedLinear import MaskedLinear

# import the linear layer from torch.nn
from torch.nn import Linear

def run_test(N_IN, N_OUT, SPARSITY):
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

    num_samples = 5
    num_in_features = N_IN
    num_out_features = N_OUT


    # Create a sparse linear layer
    sparse_model = SparseLinear.from_singleBlock(row_size=num_out_features,
                                                col_size=num_in_features,
                                                block_type='R='+str(SPARSITY),
                                                initialization_type='G=0.0,1.0',
                                                optimized_implementation=True,
                                                transpose=True,
                                                bias=False,
                                                dtype=torch.float32,
                                                device=device)

    # # Print the number of trainable parameters and the min and max values
    # ic(num_out_features * num_in_features)
    # ic(sparse_model.number_of_trainable_parameters())
    # ic(sparse_model.number_of_trainable_parameters()/(num_out_features * num_in_features))

    # ic(sparse_model.sparse_trainable_indices.shape)
    # ic(torch.abs(sparse_model.sparse_trainable_values).max())
    # ic(torch.abs(sparse_model.sparse_trainable_values).min())

    # ic(sparse_model.sparse_trainable_indices[0,:].max())
    # ic(sparse_model.sparse_trainable_indices[0,:].min())
    # ic(sparse_model.sparse_trainable_indices[1,:].max())
    # ic(sparse_model.sparse_trainable_indices[1,:].min())

    sparse_weights = torch.sparse_coo_tensor(sparse_model.sparse_trainable_indices, 
                                            sparse_model.sparse_trainable_values, 
                                            size=(num_out_features, num_in_features),
                                            dtype=torch.float32,
                                            device=device)

    # ic(sparse_weights._indices().shape[1])
    # ic(torch.abs(sparse_weights._values()).max())
    # ic(torch.abs(sparse_weights._values()).min())

    dense_weights = sparse_weights.to_dense().to(device)

    dense_model = Linear(in_features=num_in_features, out_features=num_out_features,
                        bias=False,
                        dtype=torch.float32,
                        device=device)

    x = torch.randn((num_samples, num_in_features),
                    dtype=torch.float32,
                    device=device)

    # Create a linear layer with the same weights as the sparse model
    # ic(dense_model.weight.shape)
    with torch.no_grad():
        dense_model.weight.data = dense_weights
    # ic(dense_model.weight.shape)

    # # Run the forward pass for both models

    # raw_dense_output = x @ (dense_weights.T)
    # raw_sparse_output = x @ (sparse_weights.T)
    # dense_output = dense_model(x)
    # sparse_output = sparse_model(x)

    # ic(raw_dense_output.shape)
    # ic(raw_sparse_output.shape)
    # ic(dense_output.shape)
    # ic(sparse_output.shape)

    # ic(raw_dense_output[:2,:2])
    # ic(raw_sparse_output[:2,:2])
    # ic(dense_output[:2,:2])
    # ic(sparse_output[:2,:2])
        
    # Create a low-rank layer with the same *number of weights* as the sparse model
    # [N_IN x h] @ [h x N_OUT] @ x

    low_rank_hidden_rank = sparse_model.number_of_trainable_parameters() / (num_in_features + num_out_features)
    low_rank_hidden_rank_rounded = round(low_rank_hidden_rank*10)

    # ic(low_rank_hidden_rank)
    # ic(low_rank_hidden_rank_rounded)

    low_rank_possible = True
    try:
        if (low_rank_hidden_rank_rounded == 0): 
            raise Exception()
    except Exception as err:
        print('The number of features and sparsity must be large enough for a non-zero hidden rank that preserves a similar number of weights to the sparse model.')
        low_rank_possible = False
        low_rank_hidden_rank_rounded = 1
        # raise err

    low_rank_model_in = Linear(in_features=num_in_features, out_features=low_rank_hidden_rank_rounded,
                        bias=False,
                        dtype=torch.float32,
                        device=device)
    low_rank_model_out = Linear(in_features=low_rank_hidden_rank_rounded, out_features=num_out_features,
                        bias=False,
                        dtype=torch.float32,
                        device=device)
    low_rank_model = lambda x: low_rank_model_out(low_rank_model_in(x))
    low_rank_weights = [low_rank_model_in.weight.data, low_rank_model_out.weight.data]

    # ic(low_rank_weights[0].shape)
    # ic(low_rank_weights[1].shape)

    raw_low_rank_output = (x @ (low_rank_weights[0].T)) @ low_rank_weights[1].T
    low_rank_output = low_rank_model(x)

    # ic(raw_low_rank_output.shape)
    # ic(low_rank_output.shape)

    # Time the models

    def dt_func():
        with torch.no_grad():
            dense_model(x)

    def st_func():
        with torch.no_grad():
            sparse_model(x)

    def lrt_func():
        with torch.no_grad():
            low_rank_model(x)

    raw_dense_time = timeit.timeit(lambda: x @ (dense_weights.T), globals=globals(), number=10)
    raw_sparse_time = timeit.timeit(lambda: x @ (sparse_weights.T), globals=globals(), number=10)
    raw_low_rank_time = timeit.timeit(lambda: (x @ (low_rank_weights[0].T)) @ low_rank_weights[1].T, globals=globals(), number=10)
    dense_time = timeit.timeit(lambda: dense_model(x), globals=globals(), number=10)
    sparse_time = timeit.timeit(lambda: sparse_model(x), globals=globals(), number=10)
    low_rank_time = timeit.timeit(lambda: low_rank_model(x), globals=globals(), number=10)
    no_grad_dense_time = timeit.timeit(dt_func, globals=globals(), number=10)
    no_grad_sparse_time = timeit.timeit(st_func, globals=globals(), number=10)
    no_grad_low_rank_time = timeit.timeit(lrt_func, globals=globals(), number=10)

    # ic(raw_dense_time)
    # ic(raw_sparse_time)
    # ic(raw_low_rank_time)
    # ic(dense_time)
    # ic(sparse_time)
    # ic(low_rank_time)
    # ic(no_grad_dense_time)
    # ic(no_grad_sparse_time)
    # ic(no_grad_low_rank_time)

    if (low_rank_possible):
        return [N_IN, N_OUT, SPARSITY, raw_dense_time, raw_sparse_time, raw_low_rank_time, dense_time, sparse_time, low_rank_time, no_grad_dense_time, no_grad_sparse_time, no_grad_low_rank_time]
    else:
        return [N_IN, N_OUT, SPARSITY, raw_dense_time, raw_sparse_time, float('NaN'), dense_time, sparse_time, float('NaN'), no_grad_dense_time, no_grad_sparse_time, float('NaN')]

NUM_ITERS_PER_SETTING = 5
SPARSITIES = [0.01, 0.001, 0.0001, 0.00001]
N_IN = [1000, 2000, 5000, 10000, 20000]
N_OUT = [1000, 2000, 5000, 10000, 20000]

def main():
    df = pd.DataFrame(columns = ['n_in','n_out','sparsity','raw_dense_time','raw_sparse_time','raw_low_rank_time','dense_time','sparse_time','low_rank_time','no_grad_dense_time','no_grad_sparse_time','no_grad_low_rank_time'])
    

    count = 0
    for sparsity in SPARSITIES:
        for n_in in N_IN:
            for n_out in N_OUT:
                # Run each setting several times and average the results.
                data = np.zeros(len(df.columns))
                for i in range(NUM_ITERS_PER_SETTING):
                    data += run_test(n_in, n_out, sparsity)
                data /= NUM_ITERS_PER_SETTING
                df.loc[len(df)] = data

                count += 1
                print(f'Progress: {count}/{len(SPARSITIES)*len(N_IN)*len(N_OUT)}')

                df.to_csv('sparse_test_data.csv')

                pass
        pass
    pass

if __name__ == "__main__":
    main()

