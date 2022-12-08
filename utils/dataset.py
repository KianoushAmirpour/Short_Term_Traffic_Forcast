import re
import h5py
import glob
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import GaussianBlur


class GenerateTrainFiles:
    """This class generates training, validation and static data.
    """
    def __init__(self, args):
        self.args = args
        self.CITIES = self.args.cities
        self.files = self._get_the_files()  # 2019_data, 2020_data, static_data
        self.train_files = self._generate_train_val_files()  # paths to training data
        self.valid_files = self._generate_train_val_files(test=True)  # paths to validation data

    def _load_files(self, path: str):
        """This function returns the paths of files

        Args:
            path (str): paths to file

        Returns:
            list of paths to files, np.array
        """
        path_to_file = glob.glob("../data" + path)
        if len(path_to_file) > 1:  # paths to dynamic data
            return path_to_file  # just return a list of paths to dynamic files
        else:
            for path in path_to_file:  # paths to static data
                with h5py.File(path, "r") as f:
                    data = f.get("array")
                    if not isinstance(data, np.ndarray):
                        data = np.array(data)
                        return data  # return a np.array (9, 495, 436)

    def _get_the_files(self):
        static_filter = "_static.h5"
        dynamic_filter = "_8ch.h5"
        list_of_dynamic_data_2019 = []
        list_of_dynamic_data_2020 = []
        dict_of_static_data = {}
        for city in self.CITIES:
            path_to_static_data = f"/{city}/*{static_filter}"
            dict_of_static_data[city] = self._load_files(path_to_static_data)
            path_to_dynamic_data = f"/{city}/training/*{dynamic_filter}"
            dynamic_files = self._load_files(path_to_dynamic_data)
            list_of_dynamic_data_2019.extend([file for file in dynamic_files if file.split("\\")[1].startswith("2019")])
            list_of_dynamic_data_2020.extend([file for file in dynamic_files if file.split("\\")[1].startswith("2020")])
        return list_of_dynamic_data_2019, list_of_dynamic_data_2020, dict_of_static_data

    def _create_dataframe(self):
        raw_files = []
        dynamic_data_2019, dynamic_data_2020 = self.files[0], self.files[1]
        raw_files = dynamic_data_2019 + dynamic_data_2020
        info_of_files = []
        for file in raw_files:
            file_info = {}
            file_info['file'] = str(file)
            city = re.search(r"([A-Z]+)", file).group(1)
            date = re.search(r"([0-9]+-[0-9]+-[0-9]+)", file).group(1)
            file_info['date'] = date
            file_info['city'] = city
            info_of_files.append(file_info)
        df = pd.DataFrame(info_of_files)
        df["date"] = pd.to_datetime(df["date"])
        df['dayofweek'] = df["date"].dt.dayofweek
        df['month'] = df["date"].dt.month
        df['year'] = df["date"].dt.year
        return df

    def _generate_train_val_files(self, test=False):

        if test:
            n_sample = 1
            year_to_chooes = self.args.val_year[0]
        else:
            n_sample = 2  # 4(moredata)
            year_to_chooes = self.args.train_year[0]

        self.df = self._create_dataframe()
        chosen_files = []
        df_year = self.df[self.df["year"] == year_to_chooes]
        for city in self.df.city.unique():  # cities
            df_city = df_year[df_year["city"] == city]
            for month in range(1, self.df.month.nunique()+1):  # months
                df_month = df_city[df_city["month"] == month]
                for weekday in range(self.df.dayofweek.nunique()):  # days
                    df_day = df_month[df_month["dayofweek"] == weekday].sample(n_sample)
                    files = [Path(f) for f in df_day.file.values]
                    chosen_files.extend(files)
        random.shuffle(chosen_files)
        return chosen_files


class TrainDataset(Dataset):
    def __init__(self, dynamic_data: list, static_data: dict):
        self.dynamic_files = dynamic_data
        self.static_data = static_data

    def __len__(self):
        return len(self.dynamic_files) * 241

    def __getitem__(self, index):
        if index > self.__len__():
            raise IndexError("Index out of bounds")

        file_index = index // 241
        start_index = index % 241

        file_name = self.dynamic_files[file_index]
        city = re.search(r"([_A-Z_]+)", str(file_name)).group(1)

        data = self._load_the_data(file_name)
        data = data[start_index: start_index + 25]  # (25, 495, 436, 8)
        # data = data / 255.0

        input_dynamic_data = data[:12]  # (12, 495, 436, 8)
        target = data[[12, 13, 14, 17, 20, 23]]  # (6, 495, 436, 8)
        input_static_data = self.static_data[city]  # (9, 495, 436)

        return input_dynamic_data, target, input_static_data

    def _load_the_data(self, file):
        with h5py.File(file, "r") as f:
            data = f.get("array")
            if not isinstance(data, np.ndarray):
                data = np.array(data)
                return data


def train_local_sampler(dataset):
    n_samples = 241  # 241 sample for each file
    n = len(dataset)  # num_files * 241
    raw_indices = np.array(range(n))
    ndays = len(dataset) // n_samples
    rest_days = n - ndays * n_samples
    dataset_indices = []
    for n in range(ndays):  # choosing 240 indecis of each file and shuffle them
        a = n * n_samples
        b = (n+1) * n_samples
        arr = raw_indices[a:b]
        np.random.shuffle(arr)
        dataset_indices.append(arr)
    if rest_days > 0:
        arr = raw_indices[ndays * n_samples:]
        dataset_indices.append(arr)
    return np.concatenate(dataset_indices)


def val_local_sampler(dataset, num_sample=1):
    n = len(dataset)
    raw_indices = np.array(range(n))  # (n * 241)
    dataset_indices = []
    for i in range(n):  # loops over days
        first_filter = raw_indices[i * 241: (i+1) * 241]
        for j in range(0, 228, 12):  # loops over 20 hours in each day from 00:00 AM to 20:00 PM
            current_indeces = first_filter[j: (j+12)]  # choose each hour
            if len(current_indeces) == 0:
                break
            chosen_idx = np.random.randint(0, 12, num_sample)  # choose one random index for each hour
            dataset_indices.append(current_indeces[chosen_idx])
    random.shuffle(dataset_indices)
    return np.concatenate(dataset_indices)


def collate_fn(batch, blur=False):
    input_dynamic_batch, target_batch, input_static_batch = zip(*batch)

    input_dynamic_batch = np.stack(input_dynamic_batch, axis=0)
    input_dynamic_batch = np.moveaxis(input_dynamic_batch, source=4, destination=2)  # (bs, 12, 495, 436, 8) --> (bs, 12, 8, 495, 436)
    input_dynamic_batch = input_dynamic_batch.reshape(-1, 96, 495, 436)  # (bs, 12, 8, 495, 436) --> (bs, 96, 495, 436)
    input_dynamic_batch = torch.from_numpy(input_dynamic_batch).float()
    input_dynamic_batch = F.pad(input_dynamic_batch, pad=(6, 6, 1, 0))  # (bs, 96, 496, 448)

    target_batch = np.stack(target_batch, axis=0)
    target_batch = np.moveaxis(target_batch, source=4, destination=2)  # (bs, 6, 495, 436, 8) --> (bs, 6, 8, 495, 436)
    target_batch = target_batch.reshape(-1, 48, 495, 436)  # (bs, 6, 8, 495, 436) --> (bs, 48, 495, 436)
    target_batch = torch.from_numpy(target_batch).float()

    input_static_batch = np.stack(input_static_batch, axis=0)  # (9, 495, 436)
    input_static_batch = torch.from_numpy(input_static_batch)
    input_static_batch = F.pad(input_static_batch, pad=(6, 6, 1, 0))  # (9, 496, 448)

    input_batch = torch.cat([input_dynamic_batch, input_static_batch], dim=1)  # (bs, 105, 496, 448)

    if blur:
        transform = GaussianBlur((5, 5), sigma=1.2)
        input_batch = transform(input_batch)
        target_batch = transform(target_batch)

    return input_batch, target_batch
