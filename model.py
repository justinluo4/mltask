import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics


# Define your neural network model
class NeuralNetwork(pl.LightningModule):
    def __init__(self, layers = [], learning_rate = 0.001):
        super(NeuralNetwork, self).__init__()
        self.learning_rate = learning_rate
        self.flatten = nn.Flatten()
        layers.insert(0, 512)
        layers.append(101)
        self.model = nn.Sequential(
            *[nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=101)



    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)

    def shared_step(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        return y_hat, loss, accuracy

    def training_step(self, batch, batch_idx):
        y_hat, loss, accuracy = self.shared_step(batch)
        self.log("train/loss", loss)
        self.log("train/accuracy", accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        y_hat, loss, accuracy = self.shared_step(batch)
        self.log("test/loss", loss)
        self.log("test/accuracy", accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, loss, accuracy = self.shared_step(batch)
        self.log("validation/loss", loss)
        self.log("validation/accuracy", accuracy)
        return torch.argmax(y_hat, dim=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
