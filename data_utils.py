import torch
from torchvision import transforms
from skimage import io
import os
import csv
from PIL import Image
from tqdm import tqdm
import itertools
from collections import Counter
import numpy as np

class Dataset(torch.utils.data.Dataset):
  def __init__(self, root_dir, data_list, data_labels, transforms=None):
        self.data = data_list
        self.labels = data_labels
        self.transforms = transforms
        self.root_dir = root_dir

  def __len__(self):
        return len(self.data)

  def __getitem__(self, index):
        id = self.data[index]
        X = torch.load(os.path.join(self.root_dir, "images", id + ".pt"))
        if(self.transforms is not None):
            X = self.transforms(X)
        y = self.labels[index]
        return X, y

# reads from train file and test file and removes missing files
# divide test set into valid/test
def partition_test_train_with_valid(root_dir, train_path, test_path):
    train_set = set([x.split(".")[0] for x in open(train_path).read().strip().split("\n")])
    test_set = set([x.split(".")[0] for x in open(test_path).read().strip().split("\n")])
    available_files = set([x.split(".")[0] for x in os.listdir(os.path.join(root_dir, "images"))])
    train_set = train_set.intersection(available_files)
    test_set = test_set.intersection(available_files)
    return list(train_set), list(test_set)[:int(len(test_set)/2)], list(test_set)[int(len(test_set)/2):]

def parse_data(data_dir):
    train_x, test_x, valid_x = partition_test_train_with_valid(data_dir, os.path.join(data_dir, "train_val_list.txt"), os.path.join(data_dir, "test_list.txt"))
    train_y = []
    test_y = []
    valid_y = []
    for f_name in train_x:
        label_name = os.path.join(data_dir, "labels", f_name + ".pt")
        y = torch.load(label_name)
        y = torch.tensor([0]) if y[7] == 1.0 else torch.tensor([1])
        train_y.append(y.item())
    for f_name in test_x:
        label_name = os.path.join(data_dir, "labels", f_name + ".pt")
        y = torch.load(label_name)
        y = torch.tensor([0]) if y[7] == 1.0 else torch.tensor([1])
        test_y.append(y.item())
    for f_name in valid_x:
        label_name = os.path.join(data_dir, "labels", f_name + ".pt")
        y = torch.load(label_name)
        y = torch.tensor([0]) if y[7] == 1.0 else torch.tensor([1])
        valid_y.append(y.item())
    return train_x, train_y, test_x, test_y, valid_x, valid_y

def create_sampler(labels):
    _, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    sample_weights = weights[labels]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler

def create_loader(data_dir, transform, params):
    train_x, train_y, test_x, test_y, valid_x, valid_y = parse_data(data_dir)
    train_dataset = Dataset(data_dir, train_x, train_y, transforms=transform)
    train_indices = list(range(len(train_dataset)))
    valid_dataset = Dataset(data_dir, valid_x, valid_y, transforms=transform)
    valid_indices = list(range(len(valid_dataset)))
    np.random.shuffle(train_indices)
    np.random.shuffle(valid_indices)
    train_sampler, val_sampler = create_sampler(train_y), create_sampler(valid_y)
    trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, train_indices), sampler=train_sampler, **params)
    valloader = torch.utils.data.DataLoader(torch.utils.data.Subset(valid_dataset, valid_indices), sampler=val_sampler, **params)
    return train_dataset, valid_dataset, trainloader, valloader

def generate_class_counts(data_dir, *lists):
    counts = []
    for list in lists:
        d = Counter()
        for f_name in tqdm(list):
            label_name = os.path.join(data_dir, "labels", f_name + ".pt")
            y = torch.load(label_name)
            y = torch.tensor([0]) if y[7] == 1.0 else torch.tensor([1])
            d[y.item()] += 1
        counts.append(d)
    return counts

# converts data from images/csv to torch tensors
def convert_data_to_tensors():
    tfms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    csv_labels = {}
    unique_classes = set()
    with open("data/Data_Entry_2017_v2020.csv") as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            multilabels = [x.strip() for x in row[1].split("|")]
            csv_labels[row[0]] = multilabels
            unique_classes.update(multilabels)
    image_files = [x for x in os.listdir("data/images") if '.png' in x]

    classes_to_idx = {'Nodule': 0, 'Atelectasis': 1, 'Consolidation': 2, 'Pneumonia': 3, 'Hernia': 4, 'Effusion': 5, 'Edema': 6, 'No Finding': 7, 'Fibrosis': 8, 'Infiltration': 9, 'Pleural_Thickening': 10, 'Mass': 11, 'Cardiomegaly': 12, 'Emphysema': 13, 'Pneumothorax': 14}
    idx_to_classes = {v: k for k, v in classes_to_idx.items()}
    num_classes = len(classes_to_idx)

    with open(os.path.join("data", "classes.txt"), "w") as file:
        for x in range(num_classes):
            file.write(idx_to_classes[x] + "\n")

    for x in tqdm(range(len(image_files))):
        img_name = os.path.join("data", "images", image_files[x])
        id = image_files[x].split(".")[0]
        img = Image.open(img_name).convert('L')
        img_tensor = tfms(img)
        label_tensor = torch.zeros(num_classes)
        labels_list = csv_labels[image_files[x]]
        for each in labels_list:
            label_tensor[classes_to_idx[each]] = 1.0
        torch.save(img_tensor, os.path.join("data", "images", id + ".pt"))
        torch.save(label_tensor, os.path.join("data", "labels", id + ".pt"))
        os.remove(img_name)

def augment_data(data, labels, mode="undersampled"):
    d = Counter(labels)
    print(d)
