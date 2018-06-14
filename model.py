import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torch.autograd as autograd
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Regression(nn.Module):
    def __init__(self, input_dim, output):
        super(Regression, self).__init__()
        self.sequential = nn.Sequential()
        self.linear =  nn.Linear(input_dim, output, bias=False)

    def forward(self, inputs):
        out1 = self.sequential(inputs)
        out2 = self.linear(out1)
        return out2

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

    def learn(self, input_data, labels, epochs):
        train = Data.TensorDataset(torch.FloatTensor(input_data), torch.FloatTensor(labels))
        train_loader = Data.DataLoader(train, batch_size=self.batch_size, shuffle=True)
        for epoch in range(epochs):
            correct = 0.0
            total = 0.0
            correct_anomaly = 0.0
            total_anomaly = 0.0
            running_loss = 0.0
            for i, (b_x, b_y) in enumerate(train_loader):
                self.optimizer.zero_grad()
                b_x, b_y = data_to_images_labels(b_x, b_y)
                batch_x = autograd.Variable(b_x, requires_grad=False)
                batch_y = autograd.Variable(b_y, requires_grad=False)
                outputs = self.model(batch_x)
                predicted = outputs.data.cpu().numpy().argmax(axis=1)[0]
                loss = self.criterion(outputs, batch_y.long())
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                total += b_x.shape[-1]
                if int(batch_y.item()) == 1:
                    total_anomaly += 1
                    if predicted == 1:
                        correct_anomaly += 1
                correct += (predicted == int(batch_y.item()))
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
        x = autograd.Variable(bx, requires_grad=False)
        # y = autograd.Variable(by, requires_grad=False)
        outputs = self.model(x)
        return outputs.data.cpu().numpy().argmax(axis=1)[0]
