from torch.utils.data import Dataset, DataLoader, random_split

import torchvision

import os
from PIL import Image

from src.utils import transform

class PetDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.img_dir = data_dir + "/images"
        self.labelledImages = []
        with open(data_dir + "/annotations/list.txt", "r") as f:
            txt = f.read()
            image_infos = txt.split("\n")[6:-1]
        for image_info in image_infos:
            information = image_info.split(" ")
            self.labelledImages.append([information[0], information[1]])

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.labelledImages[index][0]) + ".jpg"
        image = transform(Image.open(img_path).convert("RGB"))
        label = int(self.labelledImages[index][1])
        return image, label
    
    def __len__(self):
        return len(self.labelledImages)

def get_data(image_directory):
    petDataset = PetDataset(image_directory)
    number_training_data = int(2/3 * len(petDataset))
    number_validation_data = len(petDataset) - number_training_data
    training_data, validation_data = random_split(petDataset, [number_training_data, number_validation_data])
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(validation_data, batch_size=64, shuffle=True)
    print("GREAT SUCCESS: Dataset loaded")
    print(f"Samples: {len(petDataset)}")
    return train_dataloader, test_dataloader

def main():
    get_data/("../dataset")

if __name__ == "__main__":
    main()