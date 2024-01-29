import pytorch_lightning as pl
from data_module import Caltech101DataModule
from pytorch_lightning.loggers import WandbLogger
from model import NeuralNetwork
import wandb
import argparse


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(description="Creates a NN model and trains it on the Caltech101 dataset")
    parser.add_argument("-hl", "--HiddenLayers", help="List hidden layers, space separated (e.g. \"256 128\")")
    parser.add_argument("-lr", "--LearningRate", help="Set Learning Rate (Default 0.001)")
    parser.add_argument("-e", "--Epochs", help="Set Max Epochs (default 10)")
    args = parser.parse_args()

    # Neural Network Parameters
    hidden_layers = [256]
    if args.HiddenLayers:
        hidden_layers = [int(l) for l in args.HiddenLayers.split()]

    learning_rate = 0.001
    if args.LearningRate:
        learning_rate = float(args.LearningRate)
    run_name = "hl-" + str(hidden_layers) + "-lr-" + str(learning_rate)

    max_epochs = 10
    if args.Epochs:
        max_epochs = int(args.Epochs)

    pl.seed_everything(100)

    datamodule = Caltech101DataModule(test_split=0.1, valid_split=0.1)

    model = NeuralNetwork(layers=hidden_layers, learning_rate=0.001)

    wandb.login()
    wandb_logger = WandbLogger(project="mltask", name=run_name, log_model="all")

    trainer = pl.Trainer(max_epochs=max_epochs, logger=wandb_logger)
    # Train the model
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
    wandb.finish()
