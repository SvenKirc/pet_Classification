import torch
from torch import nn

from src import config, model, dataset, utils

from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device {device}")


def train(model_pet):

    train_dataloader, test_dataloader = dataset.get_data("dataset")

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(params=model_pet.parameters(), lr=0.01)

    for epoch in range(config.epochs):
        test_loss = 0
        train_loss = 0
        model_pet.train()
        for x_train, y_train in train_dataloader:
            x_train = x_train.to(device)
            y_train = y_train.to(device).long() - 1

            y_pred = model_pet(x_train)
            loss = loss_fn(y_pred, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        model_pet.eval()
        with torch.inference_mode():
            for x_test, y_test in test_dataloader:
                x_test = x_test.to(device)
                y_test = y_test.to(device).long() - 1
                test_pred = model_pet(x_test)
                loss = loss_fn(test_pred, y_test)
                test_loss += loss.item()

            test_loss /= len(test_dataloader)
            print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

        if epoch % 10 == 0:
            torch.save(model_pet.state_dict(), "pet_classifier.pth")
        
def inference(model_pet):
    model_pet.load_state_dict(torch.load("pet_classifier.pth", weights_only=True))
    model_pet.eval()
    with torch.inference_mode():
        print("Please type in the name of the image that you want to classify (filename only):")
        file_name = input()
        print(f"Classifying the pet on the image {file_name}")
        img_path = "dataset/images/" + file_name + ".jpg"
        image = utils.transform(Image.open(img_path).convert("RGB"))
        image = image.unsqueeze(0).to(device)
        logits = model_pet(image)
        predicted_class_idx = torch.argmax(logits, dim=1).item()
        print(f"The pet on the image is a {predicted_class_idx}")

def main():

    model_pet = model.ConvolutionalPet()
    model_pet = model_pet.to(device)
    if config.training_mode == True:
        train(model_pet)
    else:
        inference(model_pet)

if __name__ == "__main__":
    main()