# %%
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

import numpy as np

import HydroErr

import dataloader

import itertools

# %%
DEVICE = "cpu"

SEQ_LENGTH = 365 * 2
TARGET_SEQ_LENGTH = 365
BASE_LENGTH = SEQ_LENGTH - TARGET_SEQ_LENGTH

FORCING_DIM = 3

N_CATCHMENTS = 1321

# training hyperparameters
memory_saving = False

# %%
dtest = dataloader.Forcing_Data(
    "./data/data_test_CAMELS_DE.csv",
    record_length=4018,
    storge_device=DEVICE,
    seq_length=SEQ_LENGTH,
    target_seq_length=TARGET_SEQ_LENGTH,
    base_length=BASE_LENGTH,
)

# %%
embedding = torch.load("data/lstm_embedding_test.pt", map_location=torch.device("cpu"))
decoder = torch.load("data/lstm_decoder_test.pt", map_location=torch.device("cpu"))

embedding.eval()
decoder.eval()

# dimension of embedding
catchment_embeddings = [x.data for x in embedding.parameters()][0]
LATENT_dim = catchment_embeddings.shape[1]

# %%
def mse_loss_with_nans(input, target):
    # Adapted from https://stackoverflow.com/a/59851632/3361298

    # Missing data are nans
    mask = torch.isnan(target)

    out = (input[~mask] - target[~mask]) ** 2
    loss = out.mean()

    return loss

# %%
def extract_catchment_embedding(catchment_index):
    # This function extracts the embedding of a catchment indexed by `catchment_index`
    with torch.no_grad():
        code = embedding(torch.tensor([catchment_index], device = DEVICE).to(dtype=torch.int64))

    return code


def extract_catchment_data(catchment_index):
    # This function extract the forcing and discharge data of catchment indexed by `catchment_index`
    x, y = dtest.get_catchment_val_batch(catchment_index)

    return x.detach().clone(), y.detach().clone()

# %%
def predict_code_discharge(code, x):
    # This function predicts discharge hydrograph given `code` and input forcing `x`
    return decoder.decode(code.expand(x.shape[0], -1), x)

# %%
def extract_and_predict_catchment_discharge_and_x_grad_sign(catchment_index):
    # This function extracts catchment embedding then predicts discharge for catchment indexed by `catchment_index`
    # It then compute the grad.sign of x
    code = extract_catchment_embedding(catchment_index)
    x, y = extract_catchment_data(catchment_index)

    # grad of x is reqoured
    x.requires_grad = True

    # prediction
    preds = predict_code_discharge(code, x)

    # compute the gradient
    loss = mse_loss_with_nans(preds, y)
    loss.backward()
    loss = loss.detach().cpu().numpy()

    grid_sign = x.grad.sign()

    return x, y, preds, grid_sign, loss

# %%
def generate_catchment_adv_example(
    catchment_index,
    epsilon=0.1,
    P_reasonable=True,
    P_change_when_it_rains=False,
    P_unchange=False,
    random_mutation=False,
    return_preds=False,
):
    # This function generates adv examples
    # P_unchanged is to choose whether to modify P
    # P_change_when_it_rains is to change P when it rains
    # P_reasonable make sure that there is no negative P in x_adv
    # if random_mutation, the grid_sign is randomly assigned rather than caculated

    x, y, preds, grid_sign, loss = (
        extract_and_predict_catchment_discharge_and_x_grad_sign(catchment_index)
    )

    if random_mutation:
        grid_sign = torch.randint(0, 2, size=grid_sign.shape, dtype=grid_sign.dtype)
        grid_sign = grid_sign * 2 - 1

    x_adv = x.clone().detach() + epsilon * grid_sign
    x = x.clone().detach()

    # restore changes during dry days
    if P_change_when_it_rains:
        x_adv[:, :, 0] = torch.where(x[:, :, 0] == 0, 0, x_adv[:, :, 0])

    # change negative P to 0
    if P_reasonable:
        x_adv[:, :, 0] = torch.where(x_adv[:, :, 0] < 0, 0, x_adv[:, :, 0])

    # keep P unchanged
    if P_unchange:
        x_adv[:, :, 0] = x[:, :, 0]

    if return_preds:
        return x_adv, x, y, preds, loss
    else:
        return x_adv

# %%
def evaluate_catchment_adv_robustness(
    catchment_index,
    epsilon=0.1,
    P_reasonable=True,
    P_change_when_it_rains=False,
    P_unchange=False,
    random_mutation=False,
):

    x_adv, x, y, preds, loss = generate_catchment_adv_example(
        catchment_index=catchment_index,
        epsilon=epsilon,
        P_reasonable=P_reasonable,
        P_change_when_it_rains=P_change_when_it_rains,
        P_unchange=P_unchange,
        random_mutation=random_mutation,
        return_preds=True,
    )

    with torch.no_grad():
        code = embedding(torch.tensor([catchment_index]).to(dtype=torch.int64))
        preds_adv = predict_code_discharge(code, x_adv)

        loss_adv = mse_loss_with_nans(preds_adv, y)
        loss_adv = loss_adv.detach().cpu().numpy()

    # compute NSE and KGE
    y_reshape = y.reshape(-1).detach().to("cpu").numpy()
    preds_reshape = preds.reshape(-1).detach().to("cpu").numpy()
    preds_adv_reshape = preds_adv.reshape(-1).detach().to("cpu").numpy()

    kge = HydroErr.kge_2009(observed_array=y_reshape, simulated_array=preds_reshape)
    kge_adv = HydroErr.kge_2009(
        observed_array=y_reshape, simulated_array=preds_adv_reshape
    )

    nse = HydroErr.nse(observed_array=y_reshape, simulated_array=preds_reshape)
    nse_adv = HydroErr.nse(observed_array=y_reshape, simulated_array=preds_adv_reshape)

    return (
        kge,
        kge_adv,
        nse,
        nse_adv,
        loss,
        loss_adv,
        x,
        x_adv,
        preds_reshape,
        preds_adv_reshape,
    )

# %% [markdown]
# ## Evaluation of P_reasonable_values

# %%
catchment_indices = list(range(0, 1321)) # 1321
epsilon_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1]
P_reasonable = [True, False]

# Generate all unique combinations of input variables
combinations = list(
    itertools.product(catchment_indices, epsilon_values, P_reasonable)
)

# Create a DataFrame from the combinations
df = pd.DataFrame(combinations, columns=["catchment_index", "epsilon", "P_reasonable"])

# Define the output variable column names
output_columns = [
    "kge",
    "kge_adv",
    "nse",
    "nse_adv",
    "loss",
    "loss_adv",
    "P_change_perc",
    "Q_change_perc",
]

# Add the output columns to the DataFrame and initialize with None or NaN
for col in output_columns:
    df[col] = None 

# %%
for i in range(df.shape[0]):
    
    (
        kge,
        kge_adv,
        nse,
        nse_adv,
        loss,
        loss_adv,
        x,
        x_adv,
        preds_reshape,
        preds_adv_reshape,
    ) = evaluate_catchment_adv_robustness(
        catchment_index=df['catchment_index'][i],
        epsilon=df['epsilon'][i],
        P_reasonable=df['P_reasonable'][i],
        P_change_when_it_rains=False,
        P_unchange=False,
        random_mutation=False,
    )
    
    # compute the changes in 
    P_change_perc = (torch.nansum(x_adv[:, :, 0])-torch.nansum(x[:, :, 0]))/torch.nansum(x[:, :, 0])*100  
    P_change_perc =P_change_perc.detach().cpu().item()

    Q_change_perc = (np.nansum(preds_adv_reshape)-np.nansum(preds_reshape))/np.nansum(preds_reshape)*100

    df.iloc[i, 3:12] = kge,	kge_adv, nse,nse_adv,loss,loss_adv, P_change_perc, Q_change_perc

    print(f'{i}/{df.shape[0]} completed')

# %%
df.to_csv('data/results/P_reasonable.csv', sep=',', index=False)

# %% [markdown]
# ## Evaluation of P_change_when_it_rains

# %%
catchment_indices = list(range(0, 1321)) # 1321
epsilon_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1]
P_change_when_it_rains = [True]

# Generate all unique combinations of input variables
combinations = list(
    itertools.product(catchment_indices, epsilon_values, P_change_when_it_rains)
)

# Create a DataFrame from the combinations
df = pd.DataFrame(combinations, columns=["catchment_index", "epsilon", "P_change_when_it_rains"])

# Define the output variable column names
output_columns = [
    "kge",
    "kge_adv",
    "nse",
    "nse_adv",
    "loss",
    "loss_adv",
    "P_change_perc",
    "Q_change_perc",
]

# Add the output columns to the DataFrame and initialize with None or NaN
for col in output_columns:
    df[col] = None

# %%
for i in range(df.shape[0]):
    
    (
        kge,
        kge_adv,
        nse,
        nse_adv,
        loss,
        loss_adv,
        x,
        x_adv,
        preds_reshape,
        preds_adv_reshape,
    ) = evaluate_catchment_adv_robustness(
        catchment_index=df['catchment_index'][i],
        epsilon=df['epsilon'][i],
        P_reasonable=True,
        P_change_when_it_rains=df['P_change_when_it_rains'][i],
        P_unchange=False,
        random_mutation=False,
    )
    
    # compute the changes in 
    P_change_perc = (torch.nansum(x_adv[:, :, 0])-torch.nansum(x[:, :, 0]))/torch.nansum(x[:, :, 0])*100  
    P_change_perc =P_change_perc.detach().cpu().item()

    Q_change_perc = (np.nansum(preds_adv_reshape)-np.nansum(preds_reshape))/np.nansum(preds_reshape)*100

    df.iloc[i, 3:12] = kge,	kge_adv, nse,nse_adv,loss,loss_adv, P_change_perc, Q_change_perc

    print(f'{i}/{df.shape[0]} completed')

# %%
df.to_csv('data/results/P_change_when_it_rains.csv', sep=',', index=False)

# %% [markdown]
# ## Evaluation of P_unchange
# 

# %%
catchment_indices = list(range(0, 1321)) # 1321
epsilon_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1]
P_unchange = [True]

# Generate all unique combinations of input variables
combinations = list(
    itertools.product(catchment_indices, epsilon_values, P_unchange)
)

# Create a DataFrame from the combinations
df = pd.DataFrame(combinations, columns=["catchment_index", "epsilon", "P_unchange"])

# Define the output variable column names
output_columns = [
    "kge",
    "kge_adv",
    "nse",
    "nse_adv",
    "loss",
    "loss_adv",
    "P_change_perc",
    "Q_change_perc",
]

# Add the output columns to the DataFrame and initialize with None or NaN
for col in output_columns:
    df[col] = None

# %%
for i in range(df.shape[0]):
    
    (
        kge,
        kge_adv,
        nse,
        nse_adv,
        loss,
        loss_adv,
        x,
        x_adv,
        preds_reshape,
        preds_adv_reshape,
    ) = evaluate_catchment_adv_robustness(
        catchment_index=df['catchment_index'][i],
        epsilon=df['epsilon'][i],
        P_reasonable=False,
        P_change_when_it_rains=False,
        P_unchange=df['P_unchange'][i],
        random_mutation=False,
    )
    
    # compute the changes in 
    P_change_perc = (torch.nansum(x_adv[:, :, 0])-torch.nansum(x[:, :, 0]))/torch.nansum(x[:, :, 0])*100  
    P_change_perc =P_change_perc.detach().cpu().item()

    Q_change_perc = (np.nansum(preds_adv_reshape)-np.nansum(preds_reshape))/np.nansum(preds_reshape)*100

    df.iloc[i, 3:12] = kge,	kge_adv, nse,nse_adv,loss,loss_adv, P_change_perc, Q_change_perc

    print(f'{i}/{df.shape[0]} completed')

# %%
df.to_csv('data/results/P_unchange.csv', sep=',', index=False)

# %% [markdown]
# ## Evaluation of random_mutation

# %%
catchment_indices = list(range(0, 1321)) # 1321
epsilon_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1]
random_mutation = [True]
repeats = list(range(0, 10)) # repeat 10 times

# Generate all unique combinations of input variables
combinations = list(
    itertools.product(catchment_indices, epsilon_values, random_mutation, repeats)
)

# Create a DataFrame from the combinations
df = pd.DataFrame(combinations, columns=["catchment_index", "epsilon", "random_mutation", "repeats"])

# Define the output variable column names
output_columns = [
    "kge",
    "kge_adv",
    "nse",
    "nse_adv",
    "loss",
    "loss_adv",
    "P_change_perc",
    "Q_change_perc",
]

# Add the output columns to the DataFrame and initialize with None or NaN
for col in output_columns:
    df[col] = None

# %%
for i in range(df.shape[0]):
    
    (
        kge,
        kge_adv,
        nse,
        nse_adv,
        loss,
        loss_adv,
        x,
        x_adv,
        preds_reshape,
        preds_adv_reshape,
    ) = evaluate_catchment_adv_robustness(
        catchment_index=df['catchment_index'][i],
        epsilon=df['epsilon'][i],
        P_reasonable=True,
        P_change_when_it_rains=False,
        P_unchange=False,
        random_mutation=df['random_mutation'][i],
    )
    
    # compute the changes in 
    P_change_perc = (torch.nansum(x_adv[:, :, 0])-torch.nansum(x[:, :, 0]))/torch.nansum(x[:, :, 0])*100  
    P_change_perc =P_change_perc.detach().cpu().item()

    Q_change_perc = (np.nansum(preds_adv_reshape)-np.nansum(preds_reshape))/np.nansum(preds_reshape)*100

    df.iloc[i, 4:13] = kge,	kge_adv, nse,nse_adv,loss,loss_adv, P_change_perc, Q_change_perc
    
    print(f'{i}/{df.shape[0]} completed')

# %%
df.to_csv('data/results/random_mutation.csv', sep=',', index=False)


