import torch
import numpy as np

# %% Define the loss function
def quantile_loss(y_hat, y, quantiles):
    """
    y_hat: (batch_size, num_fcst, patch_len, num_quantiles)
    y: (batch_size, num_fcst, patch_len)
    quantiles: list[float]
    """
    loss = torch.tensor(0.0).to(y.device)
    for i, q in enumerate(quantiles):
        error = y - y_hat[:, :, :, i]
        loss += torch.max((q - 1) * error, q * error).mean()
    return loss / len(quantiles)


def rmse_loss(y_hat, y):
    """
    y_hat: (batch_size, num_fcst, patch_len, *)
    y: (batch_size, num_fcst, patch_len)
    """
    # todo: deal with stride != patch_len, num_fcst = y_hat.shape[1]
    if len(y_hat.shape) == 4:
        y_hat = y_hat[:, :, :, 0]
    error = y - y_hat
    loss = torch.sqrt(torch.mean(error ** 2))
    return loss


def mae_loss(y_hat, y):
    """
    y: (batch_size, horizon)
    y_hat: (batch_size, horizon)
    """
    if len(y_hat.shape) == 4:
        y_hat = y_hat[:, :, :, 0]
    return torch.mean(torch.abs(y_hat - y))

def get_loss(args):
    if args.loss == 'rmse':
        return rmse_loss
    elif args.loss == 'mae':
        return mae_loss
    elif args.loss == 'gaussian_nll':
        loss = torch.nn.GaussianNLLLoss()
        return lambda y_hat, y: loss(y_hat[:, :, :, 0], y, y_hat[:, :, :, 1]**2)
    else:
        raise ValueError('loss not supported')



# def evaluate(model, test_loader, device='cpu'):
#     """Evaluate the model on the test set.
#     Returns
#     -------
#     test_mae, test_rmse
#     """
#     normal_mae = 0
#     normal_mse = 0
#     abnormal_mae = 0
#     abnormal_mse = 0
#
#     model.eval()
#     len_data = len(test_loader.dataset)
#     len_abnormal = 0
#     len_normal = 0
#     with torch.no_grad():
#         for x, y, is_abnormal in test_loader:
#             x = [xx.to(device) for xx in x]
#             y_hat = model.forecast(x)
#             num_target = y_hat.shape[1]
#             y = y[:, -num_target:, :].to(device)
#
#             if (~is_abnormal).any():
#                 bs_len_normal = (~is_abnormal).sum().item()
#                 len_normal += bs_len_normal
#                 normal_mse += rmse_loss(y_hat[~is_abnormal], y[~is_abnormal]).item() ** 2 * bs_len_normal / len_data
#                 normal_mae += mae_loss(y_hat[~is_abnormal], y[~is_abnormal]).item() * bs_len_normal / len_data
#             if is_abnormal.any():
#                 bs_len_abnormal = is_abnormal.sum().item()
#                 len_abnormal += bs_len_abnormal
#                 abnormal_mse += rmse_loss(y_hat[is_abnormal], y[is_abnormal]).item() ** 2 * bs_len_abnormal / len_data
#                 abnormal_mae += mae_loss(y_hat[is_abnormal], y[is_abnormal]).item() * bs_len_abnormal / len_data
#
#     total_rmse = (normal_mse + abnormal_mse) ** 0.5
#     total_mae = normal_mae + abnormal_mae
#
#     normal_rmse = (normal_mse * (len_data / len_normal)) ** 0.5
#     normal_mae = normal_mae * (len_data / len_normal)
#
#     abnormal_rmse = (abnormal_mse * (len_data / len_abnormal)) ** 0.5
#     abnormal_mae = abnormal_mae * (len_data / len_abnormal)
#
#     return total_rmse, total_mae, normal_rmse, normal_mae, abnormal_rmse, abnormal_mae
#

