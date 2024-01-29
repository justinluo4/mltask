import torch
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from torchvision import datasets
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
class ClipTransformer(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __call__(self, img):
        inputs = self.processor(images=img, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].to(self.device)
        outputs = self.model(**inputs)
        return outputs.image_embeds


class Caltech101DataModule(pl.LightningDataModule):
    def __init__(self, test_split = 0.1, valid_split=0.1):
        super().__init__()
        self.test_split = test_split
        self.valid_split = valid_split

    def prepare_data(self):
        # download
        datasets.Caltech101(root="data", download="True")



    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders

        data_full = datasets.Caltech101(root="data", transform=ClipTransformer())

        test_size = int(self.test_split * len(data_full))
        valid_size = int(self.valid_split * len(data_full))
        self.data_train, self.data_test , self.data_valid = random_split(
            data_full,
            [len(data_full) - test_size - valid_size, test_size, valid_size])

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=64)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=64, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_valid, batch_size=64)
