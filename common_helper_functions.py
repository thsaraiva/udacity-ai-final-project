from torch import nn, optim
from torchvision import models
import torch


def get_device_type(use_gpu):
    will_use_gpu = torch.cuda.is_available() and use_gpu
    # print(f"Will use GPU: {will_use_gpu}")
    return torch.device("cuda" if will_use_gpu else "cpu")


def load_pre_trained_network_model(arch, hidden_units, output_units, learning_rate, is_training=True):
    model = None
    params_to_update = []
    if arch == "resnet":
        """ Resnet18
        """
        model = models.resnet18(pretrained=True)
        if is_training:
            disable_grad_for_pretrained_network(model)
        classifier = nn.Sequential(nn.Linear(model.fc.in_features, hidden_units),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(hidden_units, output_units),
                                   nn.LogSoftmax(dim=1))
        model.fc = classifier
        params_to_update = model.fc.parameters()

    elif arch == "alexnet":
        """ Alexnet
        """
        model = models.alexnet(pretrained=True)
        if is_training:
            disable_grad_for_pretrained_network(model)
        classifier = nn.Sequential(model.classifier[0],
                                   model.classifier[1],
                                   model.classifier[2],
                                   model.classifier[3],
                                   model.classifier[4],
                                   model.classifier[5],
                                   nn.Linear(model.classifier[6].in_features, hidden_units),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(hidden_units, output_units),
                                   nn.LogSoftmax(dim=1))
        model.classifier = classifier
        for param in model.classifier.parameters():
            param.requires_grad = True
        params_to_update = model.classifier.parameters()

    elif arch == "vgg":
        """ VGG11_bn
        """
        model = models.vgg11_bn(pretrained=True)
        if is_training:
            disable_grad_for_pretrained_network(model)
        classifier = nn.Sequential(model.classifier[0],
                                   model.classifier[1],
                                   model.classifier[2],
                                   model.classifier[3],
                                   model.classifier[4],
                                   model.classifier[5],
                                   nn.Linear(model.classifier[6].in_features, hidden_units),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(hidden_units, output_units),
                                   nn.LogSoftmax(dim=1))
        model.classifier = classifier
        for param in model.classifier.parameters():
            param.requires_grad = True
        params_to_update = model.classifier.parameters()

    elif arch == "densenet":
        """ Densenet
        """
        model = models.densenet121(pretrained=True)
        if is_training:
            disable_grad_for_pretrained_network(model)
        classifier = nn.Sequential(nn.Linear(model.classifier.in_features, hidden_units),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(hidden_units, output_units),
                                   nn.LogSoftmax(dim=1))
        model.classifier = classifier
        params_to_update = model.classifier.parameters()

    else:
        print("Invalid model name, exiting...")
        exit()

    # print(f"Network architecture: \n{model}\n")

    # TODO: test with SGD optimiser
    # optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    optimizer = optim.Adam(params_to_update, lr=learning_rate)

    # TODO: test with CrossEntropy loss function
    # nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()

    if not is_training:
        disable_grad_for_pretrained_network(model)

    return model, optimizer, criterion


def disable_grad_for_pretrained_network(model):
    for param in model.parameters():
        param.requires_grad = False
