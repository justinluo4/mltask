import pytorch_lightning as pl
from data_module import Caltech101DataModule
from pytorch_lightning.loggers import WandbLogger
from model import NeuralNetwork

# Create a PyTorch Lightning trainer
wandb_logger = WandbLogger(project = "mltask", log_model="all")
trainer = pl.Trainer(max_epochs=10, log_every_n_steps=20, logger=wandb_logger)

# Create a PyTorch Lightning data module

# Create an instance of the data module
datamodule = Caltech101DataModule()

# Create an instance of the neural network model
model = NeuralNetwork()

# Train the model
trainer.fit(model, datamodule)
trainer.test(model, datamodule)
