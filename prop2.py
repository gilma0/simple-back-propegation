import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader, Dataset
import torch.utils.data.sampler as sampler


class MyModel(nn.Module):
    def __init__(self, in_features: int = 32, out_features: int = 1):
        super().__init__()
        self.layer1 = nn.Linear(in_features, 8)
        self.layer2 = nn.Linear(8, 4)
        self.layer3 = nn.Linear(4, 2)
        self.layer4 = nn.Linear(2, 1)

    def forward(self, X):
        y = self.layer1(X)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        return y


class OurDataset(Dataset):
    def __init__(self, X, y):
        self.X = self._pre_process(X)
        self.y = y
        if torch.cuda.is_available():
            print(f'Moving tensors to cuda!')
            self.X.to(device='cuda')
            self.y.to(device='cuda')

    def _pre_process(self, X):
        means = X.mean(0)
        max, _ = X.max(0)
        min, _ = X.min(0)
        return (X - means) / (max - min)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(model, criterion, optimizer, dl_train, num_epochs: int=10):
    for epoch in range(num_epochs):
        epoch_loss = []
        for x, y in dl_train:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred.squeeze(), y.to(dtype=torch.float))
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        print(f'Loss for epoch {epoch+1}: {torch.tensor(epoch_loss).mean()}')


def eval_model(model, criterion, dl_test):
    total_loss = []
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for x, y in dl_test:
            y_pred = model(x)
            loss = criterion(y_pred.squeeze(), y.to(dtype=torch.float))
            total_loss.append(loss.item())
            y_pred[y_pred > 0] = 1
            y_pred[y_pred <= 0] = -1
            correct = (y_pred.squeeze() == y).sum().item()
            total_correct += correct
            total_samples += y.shape[0]
            #print(f'{correct} out of {y.shape[0]}')
    print(f'Total correct: {total_correct} out of {total_samples} which is {100 * total_correct/total_samples}')


def create_train_validation_loaders(dataset: Dataset, validation_ratio,
                                    batch_size=100, num_workers=2):
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :param batch_size: Batch size the loaders will return from each set.
    :param num_workers: Number of workers to pass to dataloader init.
    :return: A tuple of train and validation DataLoader instances.
    """
    if not(0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    random_indices = torch.randperm(len(dataset))
    split = int(len(dataset) * validation_ratio)
    dl_train = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(random_indices[split:]),
        num_workers=num_workers
    )
    dl_valid = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(random_indices[:split]),
        num_workers=num_workers
    )

    return dl_train, dl_valid


def get_data_loaders(filename, batch_size: int=10):
    labels = []
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            d = line.split(",")[1:]
            labels.append(-1 if d[0] == "N" else 1)
            features = []
            for feature in d[2:]:
                if feature in ['?', '?\n']:
                    feature = 0
                features.append(float(feature))
            data.append(torch.tensor(features))
    X = torch.stack(data, 0)
    y = torch.tensor(labels)

    ds = OurDataset(X, y)
    return create_train_validation_loaders(ds, 0.33, batch_size)


if __name__ == '__main__':
    data_file = 'C:\\Users\\Gil-PC\\Downloads\\wpbc.data'
    dl_train, dl_test = get_data_loaders(data_file)

    model = MyModel(32, 1)
    lr = 0.001
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)

    start = time.time()

    train_model(model, criterion, optimizer, dl_train=dl_train, num_epochs=10)
    print(f'success in test: ')
    eval_model(model, criterion, dl_test)
    print(f'success in training')
    eval_model(model, criterion, dl_train)


    end = time.time()
    print(f'time it took: {end - start} seconds')
