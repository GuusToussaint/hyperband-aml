import torch
import torch.nn as nn
import torch.nn.functional as F


class FCnet(torch.nn.Module):
    def __init__(self, in_size, out_size, num_layers, dropout_rate, num_fc_units, kernel_size):
        super().__init__()

        self.layers = nn.ModuleList()
        self.input_length = in_size

        if num_layers != 1:
            self.layers.append(nn.Linear(in_size, num_fc_units))
            if num_layers > 2:
                for i in range(1, num_layers-1):
                    self.layers.append(nn.Linear(num_fc_units, num_fc_units))

            self.layers.append(nn.Linear(num_fc_units, out_size))

        else:
            self.layers.append(nn.Linear(in_size, out_size))

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):

        x = x.view(-1, self.input_length)

        if len(self.layers) != 1:
            for i in range(len(self.layers)-1):
                x = self.layers[i](x)
                x = F.relu(x)
                x = self.dropout(x)

        x = self.layers[-1](x)

        return F.log_softmax(x, dim=1)

    def number_of_parameters(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))
