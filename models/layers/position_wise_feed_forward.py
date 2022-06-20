from torch import nn


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)  #(512, 2048)
        self.linear2 = nn.Linear(hidden, d_model)  #(2048, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x) #(128, 28, 2048)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x) #(128, 28, 512)
        return x