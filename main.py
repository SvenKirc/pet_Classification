import torch
from torch import nn

from src import config, model, dataset, utils


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device {device}")


def train(model_pet):

    train_dataloader, test_dataloader = dataset.get_data("dataset")

    train_loss_values = []
    test_loss_values = []
    epoch_count = []
    test_loss = 0

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(params=model_pet.parameters(), lr=0.01)

    for epoch in range(config.epochs):
        for x_train, y_train in train_dataloader:
            model_pet.train()
            x_train = x_train.to(device)
            y_train = y_train.to(device).long() - 1
            y_pred = model_pet(x_train)
            loss = loss_fn(y_pred, y_train)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        with torch.inference_mode():
            for x_test, y_test in test_dataloader:
                x_test = x_test.to(device)
                y_test = y_test.to(device).long() - 1
                test_pred = model_pet(x_test)
                loss = loss_fn(test_pred, y_test)
                test_loss += loss.item()

            test_loss /= len(test_dataloader)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Train Loss: {loss} | Test Loss: {test_loss}")


def main():

    model_pet = model.ConvolutionalPet()
    model_pet = model_pet.to(device)
    if config.training_mode == True:
        train(model_pet)

if __name__ == "__main__":
    main()