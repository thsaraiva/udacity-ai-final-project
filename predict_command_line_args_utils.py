import argparse


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="Predicts flower name from an image. Returns the flower name and its probability")
    parser.add_argument('flower_image_file', help="Flower image to be predicted.")
    parser.add_argument('checkpoint_file',
                        help="Checkpoint file containing information about the neural netowrk model to be used in the inference")
    parser.add_argument('--topk', help="Return top K most likely classes",
                        type=int, default="3")
    parser.add_argument('--category_names', help="JSON file to be used as mapping of categories to real names.")
    parser.add_argument('--use_gpu', help="Specify if Cuda GPU should be used for inference IF AVAILABLE",
                        action="store_true", default=True)
    parser.add_argument('-v', '--verbose', help="Enable debug logs",
                        action="store_true", default=False)
    return parser.parse_args()
