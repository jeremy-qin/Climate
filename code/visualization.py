# Standard library
from datetime import datetime
import os
import random

# Third party
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


def interpolate_input(x: torch.Tensor, y: torch.Tensor):
    # interpolate input to match output size
    out_h, out_w = 64, 128
    x = torch.nn.functional.interpolate(x, (out_h, out_w), mode="bilinear")
    return x


def visualize(model_module, data_module, data, data_label, samples=1, var = "2m_temperature", save_dir=None):
    """Visualizes model bias.
    :param model_module: A ClimateLearn model.
    :type model_module: LightningModule
    :param data_module: A ClimateLearn dataset.
    :type data_module: LightningDataModule
    :param split: "train", "val", or "test".
    :type split: str, optional
    :param samples: The exact days or the number of days to visualize. If provided as
        exact days, this should be a list of datetime strings, each formatted as
        "YYYY-mm-dd:HH". If provided as the number of days, it must be an int n. In
        this case, n days are randomly sampled from the given split.
    :type samples: List[str]|int, optional
    :param save_dir: The directory to save the visualization to. Defaults to `None`,
        meaning the visualization is not saved.
    :type save_dir: str, optional
    """
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # dataset.setup()
    task_dataset = data

    if type(samples) == int:
        idxs = random.sample(range(0, len(task_dataset)), samples)
    elif type(samples) == list:
        idxs = [
            np.searchsorted(
                task_dataset.time, np.datetime64(datetime.strptime(dt, "%Y-%m-%d:%H"))
            )
            for dt in samples
        ]
    else:
        raise Exception(
            "Invalid type for samples; Allowed int or list[datetime.datetime or np.datetime64]"
        )

    fig, axes = plt.subplots(len(idxs), 4, figsize=(30, 3 * len(idxs)), squeeze=False)

    for index, idx in enumerate(idxs):
        x, y = data[0], data_label[0] # 1, 1, 32, 64
        print(y.shape)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = interpolate_input(x, y)
        pred = model_module.forward(x.unsqueeze(0))  # 1, 1, 32, 64
        print(pred.shape)
        inv_normalize = model_module.denormalization
        init_condition, gt = inv_normalize(x), inv_normalize(y)
        init_condition = np.flip(init_condition.detach().cpu().squeeze().numpy(), 0)
        pred = inv_normalize(pred)
        pred = np.flip(pred.detach().cpu().squeeze().numpy(), 0)
        gt = np.flip(gt.detach().cpu().squeeze().numpy(), 0)
        print(gt.shape)
        bias = pred - gt

        for i, np_array in enumerate([init_condition[1], gt, pred, bias]):
            ax = axes[index][i]
            im = ax.imshow(np_array)
            im.set_cmap(cmap=plt.cm.coolwarm_r)
            fig.colorbar(im, ax=ax)

        if var == "2m_temperature":
          axes[index][0].set_title("Low resolution data [Kelvin]")
          axes[index][1].set_title("High resolution data [Kelvin]")
          axes[index][2].set_title("Downscaled [Kelvin]")
          axes[index][3].set_title("Bias [Kelvin]")
        elif var == "total_cloud_cover":
          axes[index][0].set_title("Low resolution data")
          axes[index][1].set_title("High resolution data")
          axes[index][2].set_title("Downscaled")
          axes[index][3].set_title("Bias")

    fig.tight_layout()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "visualize.png"))
    else:
        plt.show()


# def visualize_mean_bias(model_module, data_module, save_dir=None):
#     """Visualizes mean model bias on the test set.
#     :param model_module: A ClimateLearn model.
#     :type model_module: LightningModule
#     :param data_module: A ClimateLearn dataset.
#     :type data_module: LightningDataModule
#     :param save_dir: The directory to save the visualization to. Defaults to `None`,
#         meaning the visualization is not saved.
#     :type save_dir: str, optional
#     """
#     if save_dir is not None:
#         os.makedirs(save_dir, exist_ok=True)

#     loader = data_module.test_dataloader()

#     all_mean_bias = []
#     for batch in tqdm(loader):
#         x, y, _, _ = batch  # B, 1, 32, 64
#         x = x.to(model_module.device)
#         y = y.to(model_module.device)
#         if len(x.shape) == 5:
#             x = x.squeeze(1)
#         x = interpolate_input(x, y)
#         pred = model_module.forward(x)  # B, 1, 32, 64

#         inv_normalize = model_module.denormalization
#         init_condition, gt = inv_normalize(x), inv_normalize(y)
#         init_condition = np.flip(init_condition.detach().cpu().numpy(), 2)
#         pred = inv_normalize(pred)
#         pred = np.flip(pred.detach().cpu().numpy(), 2)
#         gt = np.flip(gt.detach().cpu().numpy(), 2)
#         bias = pred - gt  # B, 1, 32, 64
#         mean_bias = np.mean(bias, axis=0)
#         all_mean_bias.append(mean_bias)

#     all_mean_bias = np.stack(all_mean_bias, axis=0)
#     mean_bias = np.mean(all_mean_bias, axis=0)

#     fig, axes = plt.subplots(1, 1, figsize=(12, 4), squeeze=False)
#     ax = axes[0, 0]

#     im = ax.imshow(mean_bias.squeeze())
#     im.set_cmap(cmap=plt.cm.coolwarm_r)
#     fig.colorbar(im, ax=ax)
#     ax.set_title("Mean bias [Kelvin]")

#     fig.tight_layout()

#     if save_dir is not None:
#         plt.savefig(os.path.join(save_dir, "visualize_mean_bias.png"))
#     else:
#         plt.show()
