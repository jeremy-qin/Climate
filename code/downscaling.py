default_params = {
    "dataset": "ERA5",
    "task": "downscaling",
    "root_dir": "data",
    "in_vars": ["2m_temperature"],
    "out_vars": ["2m_temperature"],
    "train_start_year": 1979,
    "val_start_year": 2015,
    "test_start_year": 2017,
    "end_year": 2018,
    "root_highres_dir": "highres_data",
    "batch_size_prep": 1,
    "batch_size": 128,
    "num_workers": 12,
    "pred_range": 3,
    "subsample": 6,
    "in_channels": 2,
    "out_channels" : 1,
    "n_blocks": 5,
    "learning_rate": 1e-4,
    "weights_decay": 1e-5,
    "warmup_epochs": 1,
    "max_epochs": 10,
    "optimizer": "AdamW",
    "model": "resnet",
    "root_landcover" : "landcover",
    "accelerator": "gpu",
    "seed": 0,
    "precision": 16,
    "epochs": 5,
    "baseline": True
}

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from visualization import visualize

class CustomDataset(Dataset):
    def __init__(self, train_images, train_labels, transform=None):
        self.train_images = torch.tensor(train_images.astype("float32"))
        self.train_labels = torch.tensor(train_labels.astype("float32"))
        self.transform = transform
        
    def __len__(self):
        return len(self.train_images)
    
    def __getitem__(self, idx):
        image = self.train_images[idx]
        label = self.train_labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    

def defaults(dictionary, dictionary_defaults):
    for key, value in dictionary_defaults.items():
        if key not in dictionary:
            dictionary[key] = value
        else:
            if isinstance(value, dict) and isinstance(dictionary[key], dict):
                dictionary[key] = defaults(dictionary[key], value)
            elif isinstance(value, dict) or isinstance(dictionary[key], dict):
                raise ValueError("Given dictionaries have incompatible structure")
    return dictionary

def landcover_prep(root):
    import cv2
    import numpy as np

    img = cv2.imread(root)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray, (64, 32))

    img_resized_1 = img_resized[:, :32]
    img_resized_2 = img_resized[:, 32:64]
    landcover = np.concatenate((img_resized_2, img_resized_1), axis=1)
    return landcover

def merge_datasets(data_module, img):
    import numpy as np

    img_resized = img

    new_train_data = []
    for batch in data_module.train_dataloader():
        new = np.stack((batch[0][0].squeeze(), img_resized))
        new_train_data.append(new)
    new_train_data = np.array(new_train_data)

    new_train_label = []
    for batch in data_module.train_dataloader():
        new = batch[1][0].numpy()
        new_train_label.append(new)
    new_train_label = np.array(new_train_label)

    new_val_data = []
    for batch in data_module.val_dataloader():
        new = np.stack((batch[0][0].squeeze(), img_resized))
        new_val_data.append(new)
    new_val_data = np.array(new_val_data)

    new_val_label = []
    for batch in data_module.val_dataloader():
        new = batch[1][0].numpy()
        new_val_label.append(new)
    new_val_label = np.array(new_val_label)

    new_test_data = []
    for batch in data_module.test_dataloader():
        new = np.stack((batch[0][0].squeeze(), img_resized))
        new_test_data.append(new)
    new_test_data = np.array(new_test_data)

    new_test_label = []
    for batch in data_module.test_dataloader():
        new = batch[1][0].numpy()
        new_test_label.append(new)
    new_test_label = np.array(new_test_label)

    return new_train_data, new_train_label, new_val_data, new_val_label, new_test_data, new_test_label


def experiment(params):
    from climate_learn.utils.data import load_dataset, view
    from climate_learn.utils.datetime import Year, Days, Hours
    from climate_learn.data import DataModule
    from climate_learn.models import load_model
    from torch.optim import AdamW
    import torch
    import torch.nn as nn
    from climate_learn.models import set_climatology
    from climate_learn.training import Trainer, WandbLogger
    import wandb

    print("Start of experiment")

    dataset = params["dataset"]
    task = params["task"]
    root_dir = params["root_dir"]
    in_vars = params["in_vars"]
    out_vars = params["out_vars"]
    train_start_year = params["train_start_year"]
    val_start_year = params["val_start_year"]
    test_start_year = params["test_start_year"]
    end_year = params["end_year"]
    root_highres_dir = params["root_highres_dir"]
    batch_size_prep = params["batch_size_prep"]
    batch_size = params["batch_size"]
    num_workers = params["num_workers"]
    pred_range = params["pred_range"]
    subsample = params["subsample"]
    in_channels = params["in_channels"]
    out_channels = params["out_channels"]
    n_blocks = params["n_blocks"]
    learning_rate = params["learning_rate"]
    weights_decay = params["weights_decay"]
    warmup_epochs = params["warmup_epochs"]
    max_epochs = params["max_epochs"]
    optimizer = params["optimizer"]
    model_type = params["model"]
    root_landcover = params["root_landcover"]
    accelerator = params["accelerator"]
    seed = params["seed"]
    precision = params["precision"]
    epochs = params["epochs"]
    baseline = params["baseline"]
    var = out_vars[0]


    # data_module = DataModule(
    #     dataset = dataset,
    #     task = task,
    #     root_dir = root_dir,
    #     in_vars = in_vars,
    #     out_vars = out_vars,
    #     train_start_year = Year(train_start_year),
    #     val_start_year = Year(val_start_year),
    #     test_start_year = Year(test_start_year),
    #     end_year = Year(end_year),
    #     root_highres_dir = root_highres_dir,
    #     batch_size = batch_size_prep,
    #     num_workers = num_workers,
    #     pred_range = Days(pred_range),
    #     subsample = Hours(subsample)
    # )

    model_kwargs = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "n_blocks": n_blocks
    }

    optim_kwargs = {
        "lr": learning_rate,
        "weight_decay": weights_decay,
        "warmup_epochs": warmup_epochs,
        "max_epochs": max_epochs,
        "optimizer" : AdamW,
    }

    def collate_fn(batch):
        r"""Collate function for DataLoaders.
        :param batch: A batch of data samples.
        :type batch: List[Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]]
        :return: A tuple of `input`, `output`, `variables`, and `out_variables`.
        :rtype: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]
        """
        inp = torch.stack([batch[i][0] for i in range(len(batch))])
        out = torch.stack([batch[i][1] for i in range(len(batch))])
        variables = in_vars
        out_variables = out_vars
        return inp, out, variables, out_variables

    model_module = load_model(name = model_type, task = task, model_kwargs=model_kwargs, optim_kwargs=optim_kwargs)

    if baseline != True:
        data_module = DataModule(
        dataset = dataset,
        task = task,
        root_dir = root_dir,
        in_vars = in_vars,
        out_vars = out_vars,
        train_start_year = Year(train_start_year),
        val_start_year = Year(val_start_year),
        test_start_year = Year(test_start_year),
        end_year = Year(end_year),
        root_highres_dir = root_highres_dir,
        batch_size = batch_size_prep,
        num_workers = num_workers,
        pred_range = Days(pred_range),
        subsample = Hours(subsample)
        )
        print("Landcover Preparation")
        landcover_img = landcover_prep(root_landcover)
        
        train_data, train_label, val_data, val_label, test_data, test_label = merge_datasets(data_module=data_module, img=landcover_img)

        train_dataset = CustomDataset(train_data, train_label)
        val_dataset = CustomDataset(val_data, val_label)
        test_dataset = CustomDataset(test_data, test_label)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn = collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn = collate_fn)

        class MyDataModule(DataModule):
            def test_dataset(self):
                return test_data

            def train_dataloader(self):
                return train_dataloader
            
            def val_dataloader(self):
                return val_dataloader

            def test_dataloader(self):
                return test_dataloader
            
        data_module = MyDataModule(
            dataset = dataset,
            task = task,
            root_dir = root_dir,
            in_vars = in_vars,
            out_vars = out_vars,
            train_start_year = Year(train_start_year),
            val_start_year = Year(val_start_year),
            test_start_year = Year(test_start_year),
            end_year = Year(end_year),
            root_highres_dir = root_highres_dir,
            batch_size = batch_size,
            num_workers = num_workers,
            pred_range = Days(pred_range),
            subsample = Hours(subsample)
        )
    else:
        data_module = DataModule(
        dataset = dataset,
        task = task,
        root_dir = root_dir,
        in_vars = in_vars,
        out_vars = out_vars,
        train_start_year = Year(train_start_year),
        val_start_year = Year(val_start_year),
        test_start_year = Year(test_start_year),
        end_year = Year(end_year),
        root_highres_dir = root_highres_dir,
        batch_size = batch_size,
        num_workers = num_workers,
        pred_range = Days(pred_range),
        subsample = Hours(subsample)
    )
    
    set_climatology(model_module, data_module)

    trainer = Trainer(
        seed = seed,
        accelerator = accelerator,
        precision = precision,
        max_epochs = epochs,
        logger = WandbLogger(project="downscaling", name="Downscaling")
    )
    print("Training")

    trainer.fit(model_module, data_module)

    trainer.test(model_module, data_module)
    
    print("End of Experiment")



if __name__ == "__main__":
    import json
    import argparse
    import wandb

    from datetime import datetime

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--params", "-p", type=str, help="JSON params file")
    parser.add_argument("--direct", "-d", type=str, help="JSON state string")
    
    arguments = parser.parse_args()
    
    if arguments.direct is not None:
        params = json.loads(arguments.direct)
    elif arguments.params is not None:
        with open(arguments.params) as file:
            params = json.load(file)
    else:
        params = {}

    params = defaults(params, default_params)
    # log_name = params["dataset"] + "-" + current_time
    # wandb.init(project="CNN-Magic", name = log_name, entity="qinjerem", config=params)

    experiment(params)

    # wandb.finish()

