import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torch.autograd as autograd
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Regression(nn.Module):
    def __init__(self, input_dim, output):
        super(Regression, self).__init__()
        self.fc = nn.Linear(input_dim, output)

    def forward(self, inputs):
        # out1 = self.fc(inputs)
        out = F.relu(self.fc(inputs))
        # out2 = self.linear(out1)
        return out

def data_to_images_labels(inputs, labels):
    if torch.cuda.is_available():
        inputs, labels = inputs.to(device), labels.to(device)
    return inputs, labels

class VideoLearn():
    def __init__(self, input_size, batch_size, lr = 0.01):
        self.model = Regression(13, 2)
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        if torch.cuda.is_available():
            self.model.to(device)

    def learn(self, data, target, epochs):
        print 'target train 0/1: {}/{}'.format(len(np.where(target == 0)[0]), len(np.where(target == 1)[0]))
        class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in target])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weigth = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        target = torch.from_numpy(target).float()
        train_dataset = Data.TensorDataset(torch.FloatTensor(data), target)

        # train = Data.TensorDataset(torch.FloatTensor(input_data), torch.FloatTensor(labels))
        train_loader = Data.DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler)
        for epoch in range(epochs):
            correct = 0.0
            total = 0.0
            correct_anomaly = 0.0
            total_anomaly = 0.0
            running_loss = 0.0
            for i, (b_x, b_y) in enumerate(train_loader):
                self.optimizer.zero_grad()
                b_x, b_y = data_to_images_labels(b_x, b_y)
                batch_x = autograd.Variable(b_x)
                batch_y = autograd.Variable(b_y)
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                # predicted = outputs.data.cpu().numpy().argmax(axis=1)[0]
                loss = self.criterion(outputs, batch_y.long())
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                total += b_x.shape[-1]
                if int(batch_y.item()) == 1:
                    total_anomaly += 1
                    if predicted == 1:
                        correct_anomaly += 1
                correct += (predicted == labels).sum().item()
            print "accuracy", "loss", "accuracy on anomalies"
            print correct/total, running_loss/total, correct_anomaly/total_anomaly
        torch.save(self.model.state_dict(), 'video_extract.pt')

class VideoCLassifier():
    def __init__(self):
        self.model = Regression(13, 2)
        self.model.load_state_dict(torch.load('video_extract.pt'))
        if torch.cuda.is_available():
            self.model.to(device)

    def predict(self, input_data, label):
        test = Data.TensorDataset(torch.FloatTensor([input_data]), torch.FloatTensor([label]))
        test_loader = Data.DataLoader(test, 1)
        bx, by = test_loader.dataset.tensors
        bx, by = data_to_images_labels(bx, by)
        x = autograd.Variable(bx)
        # y = autograd.Variable(by, requires_grad=False)
        outputs = self.model(x)
        return outputs.data.cpu().numpy().argmax(axis=1)[0]
