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
        last_hidden_state = outputs.last_hidden_state
        image_embeds = outputs.image_embeds  # pooled CLS states
        return image_embeds


class Caltech101DataModule(pl.LightningDataModule):
    def __init__(self, test_split = 0.2):
        super().__init__()
        self.test_split = test_split

    def prepare_data(self):
        # download
        datasets.Caltech101(root="data", download="True")



    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders

        data_full = datasets.Caltech101(root="data", transform=ClipTransformer())
        train_size = int((1 - self.test_split) * len(data_full))
        self.data_train, self.data_test = random_split(
            data_full,
            [train_size, len(data_full) - train_size],
            generator=torch.Generator().manual_seed(10))

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=64)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=64)
