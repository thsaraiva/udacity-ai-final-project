import copy
import os
import time

from torchvision import transforms, datasets, models
import torch


def get_data_loaders(data_dir, batch_size):
    train_dir = data_dir + 'train'
    valid_dir = data_dir + 'valid'

    # Define transforms for the training data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    # Upload training data
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    # Create training DataLoader, define batch size and shuffle configuration ON
    training_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Define transforms for the validation data
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    # Upload testing data
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    # Create testing DataLoader, define batch size and shuffle configuration OFF
    validation_data_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)

    return training_data_loader, validation_data_loader


def train_model(model, optimizer, criterion, epochs, training_data_loader, validation_data_loader, device,
                model_checkpoint):
    start = time.time()
    best_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        model_checkpoint["current_epoch"] = epoch
        print(f'\nEpoch {epoch}/{epochs}')
        print('-' * 10)

        train(training_data_loader, model, optimizer, criterion, device)

        best_acc, best_model_weights = validate(validation_data_loader, model, criterion, device, best_acc,
                                                best_model_weights, model_checkpoint)

    time_elapsed = time.time() - start
    print(f'\nTraining complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s')
    print(f'Best Accuracy: {best_acc:4f}')

    model.load_state_dict(best_model_weights)
    save_checkpoint(model, model_checkpoint)
    return model


def train(training_data_loader, model, optimizer, criterion, device):
    training_loss = 0.0
    partial_correct_predictions = 0.0
    model.train()
    dataset_length = len(training_data_loader.dataset)
    for images, labels in training_data_loader:
        batch_size = len(images)

        # Move input and label tensors to the default device
        images, labels = images.to(device), labels.to(device)

        # make sure past gradients won't influence this training step
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # do one training step
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        # calculate loss
        training_loss += loss.item() * batch_size

        # Calculate accuracy
        _, top_class = torch.exp(output).topk(1, dim=1)
        correct_predictions = torch.sum((top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)).item()
        partial_correct_predictions += correct_predictions

    training_loss = training_loss / dataset_length
    training_accuracy = (partial_correct_predictions / dataset_length) * 100
    print(f'Training ended. Loss:     {training_loss:.4f}, Accuracy:    {training_accuracy:.2f}%')


def validate(validation_data_loader, model, criterion, device, best_acc, best_model_weights, model_checkpoint):
    validation_loss = 0.0
    partial_correct_predictions = 0.0
    model.eval()
    dataset_length = len(validation_data_loader.dataset)
    for images, labels in validation_data_loader:
        batch_size = len(images)

        # Move input and label tensors to the default device
        images, labels = images.to(device), labels.to(device)

        with torch.set_grad_enabled(False):
            output = model.forward(images)
            loss = criterion(output, labels)

        # calculate loss
        validation_loss += loss.item() * batch_size

        # Calculate accuracy
        _, top_class = torch.exp(output).topk(1, dim=1)
        correct_predictions = torch.sum((top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)).item()
        partial_correct_predictions += correct_predictions

    validation_loss = validation_loss / dataset_length
    validation_accuracy = (partial_correct_predictions / dataset_length) * 100
    print(f'Validation ended. Loss:     {validation_loss:.4f}, Accuracy:    {validation_accuracy:.2f}%')

    if validation_accuracy > best_acc:
        best_acc = validation_accuracy
        model_checkpoint["best_acc"] = best_acc
        best_model_weights = copy.deepcopy(model.state_dict())
        # save_checkpoint(model, model_checkpoint, is_partial=True)
        # print(f'New best accuracy: {(best_acc):.2f}%.')

    return best_acc, best_model_weights


def save_checkpoint(model, model_checkpoint, is_partial=False):
    save_dir = model_checkpoint["checkpoint_save_dir"]
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    checkpoint_filename = f'{save_dir}{model_checkpoint["arch"]}_state_dict_checkpoint'
    if is_partial:
        checkpoint_filename = checkpoint_filename + '_partial'
    checkpoint_filename = checkpoint_filename + '.pth'
    model_checkpoint["model_state_dict"] = model.state_dict()
    torch.save(model_checkpoint, checkpoint_filename)
    print(f"Saving checkpoint file: {checkpoint_filename}")
