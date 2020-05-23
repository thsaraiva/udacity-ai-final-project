import argparse
import os
import shutil
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images_dir', help="Directory containing training images.")
    parser.add_argument('output_dir', help="Directory to copy test images to.")
    parser.add_argument('-v', '--verbose', help="Enable debug logs",
                        action="store_true", default=False)
    arguments = parser.parse_args()

    root_dir = os.getcwd() + "/"

    # if input images dir doesn't exist, exits and prints error
    if not os.path.isdir(arguments.images_dir):
        print("Images directory does not exist. Exiting program... ")
        exit()
    images_dir = root_dir + arguments.images_dir + "/"

    # if output dir doesn't exist, creates it
    if not os.path.isdir(arguments.output_dir):
        os.makedirs(arguments.output_dir)
    output_dir = root_dir + arguments.output_dir + "/"

    for category in os.listdir(images_dir):
        if category.isnumeric():
            copy_random_image_from_dir(category, images_dir, output_dir)


def copy_random_image_from_dir(dir, images_dir, output_dir):
    images = os.listdir(images_dir + dir)
    random_image = images[random.randint(0, len(images) - 1)]
    file_format = random_image.split('.')[-1]
    new_file_name = "test_image_" + dir + "." + file_format
    shutil.copy(images_dir + dir + "/" + random_image, output_dir + new_file_name)


if __name__ == "__main__":
    main()
