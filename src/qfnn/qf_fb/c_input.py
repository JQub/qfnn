import torch
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
from .c_mnist import *


def modify_target_ori(target, interest_num):
    for j in range(len(target)):
        for idx in range(len(interest_num)):
            if target[j] == interest_num[idx]:
                target[j] = idx
                break

    new_target = torch.zeros(target.shape[0], len(interest_num))

    for i in range(target.shape[0]):
        one_shot = torch.zeros(len(interest_num))
        one_shot[target[i].item()] = 1
        new_target[i] = one_shot.clone()

    return target, new_target


def modify_target(target, interest_num):
    new_target = torch.zeros(target.shape[0], len(interest_num))

    for i in range(target.shape[0]):
        one_shot = torch.zeros(len(interest_num))
        one_shot[target[i].item()] = 1
        new_target[i] = one_shot.clone()
    return target, new_target


def select_num(dataset, interest_num):
    labels = dataset.targets  # get labels
    labels = labels.numpy()
    idx = {}
    for num in interest_num:
        idx[num] = np.where(labels == num)

    fin_idx = idx[interest_num[0]]
    for i in range(1, len(interest_num)):
        fin_idx = (np.concatenate((fin_idx[0], idx[interest_num[i]][0])),)

    fin_idx = fin_idx[0]

    dataset.targets = labels[fin_idx]
    dataset.data = dataset.data[fin_idx]

    # print(dataset.targets.shape)

    dataset.targets, _ = modify_target_ori(dataset.targets, interest_num)
    # print(dataset.targets.shape)

    return dataset


class ToQuantumData_Batch(object):
    def __call__(self, tensor):
        data = tensor
        input_vec = data.view(-1)
        vec_len = input_vec.size()[0]
        input_matrix = torch.zeros(vec_len, vec_len)
        input_matrix[0] = input_vec
        input_matrix = input_matrix.transpose(0, 1)
        u, s, v = np.linalg.svd(input_matrix)
        output_matrix = torch.tensor(np.dot(u, v))
        output_data = output_matrix[:, 0].view(data.shape)
        return output_data


class ToQuantumData(object):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size

    def __call__(self, tensor):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = tensor.to(device)
        input_vec = data.view(-1)
        vec_len = input_vec.size()[0]
        input_matrix = torch.zeros(vec_len, vec_len)
        input_matrix[0] = input_vec
        input_matrix = input_matrix.transpose(0, 1)
        u, s, v = np.linalg.svd(input_matrix)
        output_matrix = torch.tensor(np.dot(u, v))
        # print(output_matrix)
        output_data = output_matrix[:, 0].view(1, self.img_size, self.img_size)
        return output_data


def load_data(interest_num, datapath, isppd, img_size, batch_size, inference_batch_size, is_to_q=True, num_workers=0):
    if isppd:
        train_data = qfMNIST(root=datapath, img_size=img_size, train=True)
        test_data = qfMNIST(root=datapath, img_size=img_size, train=False)

    else:
        # convert data to torch.FloatTensor
        if is_to_q:
            transform = transforms.Compose(
                [transforms.Resize((img_size, img_size)), transforms.ToTensor(), ToQuantumData(img_size)])
            transform_inference = transforms.Compose(
                [transforms.Resize((img_size, img_size)), transforms.ToTensor(), ToQuantumData(img_size)])
        else:
            transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
            transform_inference = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])

        # choose the training and test datasets
        train_data = datasets.MNIST(root=datapath, train=True,
                                    download=True, transform=transform)
        test_data = datasets.MNIST(root=datapath, train=False,
                                   download=True, transform=transform_inference)

    train_data = select_num(train_data, interest_num)
    test_data = select_num(test_data, interest_num)

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               num_workers=num_workers, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=inference_batch_size,
                                              num_workers=num_workers, shuffle=True, drop_last=True)

    return train_loader, test_loader


def to_quantum_matrix(tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = tensor.to(device)
    input_vec = data.view(-1)
    vec_len = input_vec.size()[0]
    input_matrix = torch.zeros(vec_len, vec_len)
    input_matrix[0] = input_vec
    input_matrix = np.float64(input_matrix.transpose(0, 1))
    u, s, v = np.linalg.svd(input_matrix)
    output_matrix = torch.tensor(np.dot(u, v), dtype=torch.float64)
    return output_matrix