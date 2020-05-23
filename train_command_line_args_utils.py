import argparse


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description="Trains a Neural Network on a specified data set.")
    parser.add_argument('data_dir', help="Directory containing training images")
    parser.add_argument('--save_dir', help="Specify in which directory to save the trained network model checkpoint",
                        default="./")
    parser.add_argument('--arch', help="Specify which neural network architecture to use",
                        choices=["resnet", "alexnet", "vgg", "densenet"],
                        default="densenet")
    parser.add_argument('--learning_rate',
                        help="Specify the learning rate used on the network training",
                        type=float, default=0.003)
    parser.add_argument('--hidden_units',
                        help="Specify the number of hidden units used in the new classifier layer of the network model",
                        type=int, default=512)
    parser.add_argument('--output_units',
                        help="Specify the number of output units used in the last layer of the network model",
                        type=int, default=102)
    parser.add_argument('--batch_size',
                        help="Specify the size of the images batches for training and validation",
                        type=int, default=64)
    parser.add_argument('--epochs', help="Specify the number of epochs used in training the network", type=int,
                        default=5)
    parser.add_argument('--use_gpu', help="Specify if Cuda GPU should be used for calculations IF AVAILABLE",
                        action="store_true", default=True)
    parser.add_argument('-v', '--verbose', help="Enable debug logs",
                        action="store_true", default=False)
    return parser.parse_args()
