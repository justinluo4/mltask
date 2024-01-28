import torch
from torch import nn
import pytorch_lightning as pl
from data_module import Caltech101DataModule
from pytorch_lightning.loggers import WandbLogger
import wandb
# Define your neural network model
class NeuralNetwork(pl.LightningModule):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(512,  256),
            nn.Sigmoid(),
            nn.Linear(256, 101),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("test/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


# Create a PyTorch Lightning trainer
wandb_logger = WandbLogger(project = "mltask", log_model="all")
trainer = pl.Trainer(max_epochs=5, log_every_n_steps=20, logger=wandb_logger)

# Create a PyTorch Lightning data module

# Create an instance of the data module
datamodule = Caltech101DataModule(test_split=0.1)

# Create an instance of the neural network model
model = NeuralNetwork()

# Train the model
trainer.fit(model, datamodule)
trainer.test(model, datamodule)
