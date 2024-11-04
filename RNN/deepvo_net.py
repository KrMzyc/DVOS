import torch
import torch.nn as nn


class DeepVONet(nn.Module):
    def __init__(self):
        super(DeepVONet, self).__init__()
        # Assuming the input shape has dimensions similar to the original code
        # Input reshape: reshape to (-1, sequence_length, features), PyTorch LSTMs accept (seq_len, batch, input_size)
        self.reshape = nn.Flatten(start_dim=2)
        self.lstm1 = nn.LSTM(input_size=60, hidden_size=1000, batch_first=True, dropout=0.5)
        self.lstm2 = nn.LSTM(input_size=1000, hidden_size=1000, batch_first=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(1000, 6)

    def forward(self, inputs):
        # Reshape inputs to be compatible with LSTM layers
        batch_size = inputs.size(0)
        x = self.reshape(inputs)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x[:, -1, :])  # Taking only the last time step's output
        x = self.out(x)
        return x
