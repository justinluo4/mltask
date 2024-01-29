import pytorch_lightning as pl
from data_module import Caltech101DataModule
from pytorch_lightning.loggers import WandbLogger
from model import NeuralNetwork
import wandb

# Create a PyTorch Lightning trainer

pl.seed_everything(100)
# Create an instance of the data module
datamodule = Caltech101DataModule()

# Create an instance of the neural network model
hidden_layers = [256]
learning_rate = 0.001

run_name = "hl-" + str(hidden_layers) + "-lr-" + str(learning_rate)
model = NeuralNetwork(layers = hidden_layers, learning_rate = 0.001)

wandb.login()
wandb_logger = WandbLogger(project="mltask", name=run_name, log_model="all")

trainer = pl.Trainer(max_epochs=10, logger=wandb_logger)
# Train the model
trainer.fit(model, datamodule)
trainer.test(model, datamodule)
