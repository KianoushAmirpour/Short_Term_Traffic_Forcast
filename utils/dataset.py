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


class GenerateTrainFiles:
    def __init__(self, args, is_test):
        self.args = args
        self.test = is_test
        self.CITIES = self.args.cities
        self.files = self._get_the_files()  # 2019_data, 2020_data, static_data
        self.dynamic_train_files = self._generate_train_val_files()  # train_files, valid_files, for test it just returns valid_files

    def _load_files(self, file: str):
        path_to_file = glob.glob("../data" + file)
        if len(path_to_file) > 1:
            return path_to_file  # just return a list of dynamic files
        else:
            for path in path_to_file:
                with h5py.File(path, "r") as f:
                    data = f.get("array")
                    if not isinstance(data, np.ndarray):
                        data = np.array(data)
                        return data

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
        if (len(self.args.years) == 1) and (self.args.years[0] == "2019"):
            raw_files.extend(dynamic_data_2019)
        elif (len(self.args.years) == 1) and (self.args.years[0] == "2020"):
            raw_files.extend(dynamic_data_2020)
        else:
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
        df['year'] = df["date"].dt.year
        df['month'] = df["date"].dt.month
        df['yeartype'] = df['city'] + df['year'].astype(str)
        # df = df.sample(frac=1).reset_index(drop=True)
        return df

    def _generate_train_val_files(self):
        self.df = self._create_dataframe()
        train_files = []
        valid_files = []
        for year in self.df.year.unique():  # years
            df_year = self.df[self.df["year"] == year]
            for city in self.df.city.unique():  # cities
                df_city = df_year[df_year["city"] == city]
                for month in range(1, self.df.month.nunique()+1):  # months
                    df_month = df_city[df_city["month"] == month]
                    for weekday in range(self.df.dayofweek.nunique()):  # days
                        df_day = df_month[df_month["dayofweek"] == weekday].sample(3)  # 2 samples for training and one for validation
                        files = [Path(f) for f in df_day.file.values]
                        train_files.extend(files[0:2])
                        valid_files.extend(files[2:3])
        random.shuffle(train_files)
        random.shuffle(valid_files)
        if self.test:
            return valid_files
        return train_files, valid_files


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
        data = data[start_index: start_index + 25]
        # data = data / 255.0

        input_dynamic_data = data[:12]
        target = data[[12, 13, 14, 17, 20, 23]]
        input_static_data = self.static_data[city]

        return input_dynamic_data, target, input_static_data

    def _load_the_data(self, file):
        with h5py.File(file, "r") as f:
            data = f.get("array")
            if not isinstance(data, np.ndarray):
                data = np.array(data)
                return data


def train_local_sampler(dataset):
    n_samples = 241
    n = len(dataset)
    raw_indices = np.array(range(n))
    ndays = len(dataset) // n_samples
    rest_days = n - ndays * n_samples
    dataset_indices = []
    for n in range(ndays):  # shuffling 240 indecis of each file
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
    # assert n == (15 * 241)
    raw_indices = np.array(range(n))  # n * 241
    dataset_indices = []
    for i in range(n):  # loops over unique days
        first_filter = raw_indices[i * 241: (i+1) * 241]
        for j in range(0, 228, 12):  # loops over 20 hours in each day
            current_indeces = first_filter[j: (j+12)]  # hour
            if len(current_indeces) == 0:
                break
            chosen_idx = np.random.randint(0, 12, num_sample)  # choose one random index for each unique hour
            dataset_indices.append(current_indeces[chosen_idx])
    random.shuffle(dataset_indices)
    return np.concatenate(dataset_indices)


def collate_fn(batch):
    input_dynamic_batch, target_batch, input_static_batch = zip(*batch)

    input_dynamic_batch = np.stack(input_dynamic_batch, axis=0)
    input_dynamic_batch = np.moveaxis(input_dynamic_batch, source=4, destination=2)  # (-1, 12, 495, 436, 8) --> (-1, 12, 8, 495, 436)
    input_dynamic_batch = input_dynamic_batch.reshape(-1, 96, 495, 436)  # (-1, 12, 8, 495, 436) --> (-1, 96, 495, 436)
    input_dynamic_batch = torch.from_numpy(input_dynamic_batch).float()
    input_dynamic_batch = F.pad(input_dynamic_batch, pad=(6, 6, 1, 0))  # (-1, 96, 496, 448)

    target_batch = np.stack(target_batch, axis=0)
    target_batch = np.moveaxis(target_batch, source=4, destination=2)
    target_batch = target_batch.reshape(-1, 48, 495, 436)
    target_batch = torch.from_numpy(target_batch).float()

    input_static_batch = np.stack(input_static_batch, axis=0)  # (9, 495, 436)
    input_static_batch = torch.from_numpy(input_static_batch)
    input_static_batch = F.pad(input_static_batch, pad=(6, 6, 1, 0))  # (9, 496, 448)

    input_batch = torch.cat([input_dynamic_batch, input_static_batch], dim=1)  # (-1, 105, 496, 448)

    return input_batch, target_batch
