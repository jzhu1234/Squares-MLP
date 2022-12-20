# Since these are all mono-colored squares, CNN's are highly unlikely to be helpful
import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes = 3,
                 batchnorm=False, dropout=True, activation="softmax"):
        super(MultiLayerPerceptron, self).__init__()

        layers = []
        # Input Layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if batchnorm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())


        for idx in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[idx], hidden_sizes[idx+1]))
            if batchnorm:
                layers.append(nn.BatchNorm1d(hidden_sizes[idx+1]))
            layers.append(nn.ReLU())

        #Output layer
        if dropout:
            layers.append(nn.Dropout(0.2))
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))

        if activation == "softmax":
            self.activation = nn.Softmax()
        else:
            self.activation = nn.Sigmoid()

        self.model = nn.Sequential(*layers)

    def forward(self, x, train=True):
        for m in self.model:
            x = m(x)
        if not train:
            x = self.activation(x)
        return x



if __name__ == '__main__':
    input_size = 3 + 1 + 1 # Color, box_size, image_size
    a = MultiLayerPerceptron(input_size, [7,3], 3)

