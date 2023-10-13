import torch

dropout = 0.7

class DQNModel(torch.nn.Module):
    def __init__(self, options={}):
        super().__init__()
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)
        self.linear1 = torch.nn.Linear(10241,4096)
        self.linear2 = torch.nn.Linear(4096,2048)
        self.linear3 = torch.nn.Linear(2048,1024)
        self.linear4 = torch.nn.Linear(1024,1)
        self.activation = torch.nn.ReLU()
    
    def forward(self, input):
        x = self.linear1(input)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.dropout3(x)
        x = self.linear4(x)
        return x
