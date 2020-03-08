import numpy as np
import os
import pandas as pd
import pydicom

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

HERE = os.path.dirname(__file__)


class PnmBaseDataset(Dataset):
    def __init__(self, df, base_path, transform=None):
        self.df = df if isinstance(df, pd.DataFrame) else pd.read_csv(df)
        self.transform = transform
        self.base_path = base_path

    def __len__(self):
        return len(self.df)


class PnmDetectionDataset(PnmBaseDataset):
    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        study_path = os.path.join(self.base_path, f"{item['patientId']}.dcm")
        image = self.transform(pydicom.read_file(study_path).pixel_array)

        x = item["x"] / 1024
        y = item["y"] / 1024
        width = item["width"] / 1024
        height = item["height"] / 1024
        label = item["Target"]

        # convert x,y to box center coordinates (YoLo bounding box notations)
        x_center = x + width / 2
        y_center = y + height / 2
        # format target vector
        target = np.zeros((5, 1))
        target[0] = label
        target[1] = x_center
        target[2] = y_center
        target[3] = width
        target[4] = height
        target = transforms.ToTensor()(target.astype(np.float32)).view(-1)

        return image, target


class PnmClassificationDataset(PnmBaseDataset):
    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        study_path = os.path.join(self.base_path, f"{item['patientId']}.dcm")
        image = self.transform(pydicom.read_file(study_path).pixel_array)

        target = np.zeros(1, dtype=np.float32)
        target[0] = item["Target"]

        return image, target


TRANSFORMS = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ]
)


def get_dataloaders(dataset_type="detection", normal_fraction=0.4, batch_size=8):
    """
    Get r
    :param dataset_type: Dataset type. Can be either "detection" or classification
    :param normal_fraction: Portion of the normal cases to include in our dataset. Using 0.4 created a rather balanced
    dataset.
    :return: dict. {"train": <train dataloader>, "valid": <validation dataloader>}
    """
    valid_data_types = ["detection", "classification"]
    assert dataset_type in valid_data_types, f"Invalid dataset type. Must be one of {valid_data_types}"
    np.random.seed(1)
    data_base_path = os.path.join(HERE, os.path.pardir, "data")
    labels_df = pd.read_csv(os.path.join(data_base_path, "stage_2_train_labels.csv"))
    labels_df.fillna(0, inplace=True)

    normals = labels_df.loc[labels_df["Target"] == 0]
    abnormals = labels_df.loc[labels_df["Target"] == 1]
    labels_df = pd.concat([normals.iloc[:int(len(normals) * normal_fraction)], abnormals])

    shuf = labels_df.reindex(np.random.permutation(labels_df.index))
    train_df = shuf.iloc[:int(0.9 * len(shuf))]
    valid_df = shuf.iloc[int(0.9 * len(shuf)):]

    DatasetClass = PnmClassificationDataset if dataset_type == "classification" else PnmDetectionDataset
    datasets = {
        "train": DatasetClass(train_df, os.path.join(data_base_path, "stage_2_train_images"), TRANSFORMS),
        "valid": DatasetClass(valid_df, os.path.join(data_base_path, "stage_2_train_images"), TRANSFORMS)
    }

    dataloaders = {
        key: DataLoader(datasets[key], batch_size=batch_size, shuffle=(key == "train"), num_workers=batch_size)
        for key in datasets
    }
    return dataloaders
