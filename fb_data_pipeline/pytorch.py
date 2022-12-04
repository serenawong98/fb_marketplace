from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
import os
import datetime

class FbDataset(Dataset):

    def __init__(self, dataframe):
        super().__init__()
        self.data = dataframe

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        features = torch.tensor(data_row[0])
        labels = torch.tensor(data_row[1])
        return(features, labels)

    def __len__(self):
        return len(self.data)


class CNN(torch.nn.Module):
    def __init__(self, colourchannel, flattened_input):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(colourchannel, 8, 7),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 7),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(flattened_input, 10),
            torch.nn.Softmax()
        )


    def forward(self, X):
        return self.layers(X)


class Resnet50(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet50.fc = torch.nn.Linear(2048, 13)

    def forward(self, X):
        return self.resnet50(X)

def train(model, dataset, lr, epochs=10, model_name = 'test', wd = 0, test_dataset = None):

    optimiser = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=wd)

    writer =  SummaryWriter()
    batch_idx = 0
    for epoch in range(epochs):
        print(epoch)
        for batch in dataset:
            features, labels = batch
            prediction = model(features)
            loss = torch.nn.functional.cross_entropy(prediction, labels.long())
            loss.backward()
            # print(f'Epoch: {epoch}, Loss: {loss.item()}')
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar('loss', loss.item(), batch_idx)
            batch_idx += 1
        if test_dataset != None:
            accuracy(model, test_dataset)
        save_model(model, model_name)


def accuracy(model, test_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        model.eval()
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

def save_model(model, model_name):
    try:
        os.mkdir('model_evaluation')
    except FileExistsError:
        pass

    try:
        os.mkdir(os.path.join('model_evaluation', model_name))
    except FileExistsError:
        pass

    try:
        os.mkdir(os.path.join('model_evaluation', model_name,'weights'))
    except FileExistsError:
        pass
    
    filename = str(datetime.datetime.now())+'.pt'
    torch.save(model.state_dict(), os.path.join('model_evaluation', model_name,'weights', filename))




if __name__ == "__main__":
    dataset = MNIST(root = './data', download = True, transform = ToTensor())
    train_set, test_set = torch.utils.data.random_split(dataset, [59992, 8])
    train_loader = DataLoader(train_set, shuffle = True, batch_size = 8)
    # print(len(dataset))
    model = CNN(1, 4096)
    accuracy(model, test_set)
    train(model, train_loader, 0.01, epochs = 3)
    accuracy(model, test_set)
    
    # new_model = CNN(1, 4096)
    # state_dict = torch.load('2022-10-02 13:16:36.311449.pt')
    # new_model.load_state_dict(state_dict)
    # accuracy(new_model, dataset)







