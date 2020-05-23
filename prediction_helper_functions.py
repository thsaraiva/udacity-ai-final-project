import json

import torch
from PIL import Image
import numpy as np


def resize_keep_ratio(image, smallest_side=256):
    width, height = image.size
    if width < height:
        width = smallest_side
        height = height * smallest_side / width
    else:
        width = width * smallest_side / height
        height = smallest_side
    resized_image = image.copy()
    resized_image.thumbnail((width, height))
    return resized_image


def center_crop(image, crop_width, crop_height):
    width, height = image.size
    # left, top, right, bottom
    left = (width - crop_width) / 2
    top = (height - crop_height) / 2
    right = left + crop_width
    bottom = top + crop_height

    cropped_image = image.crop((left, top, right, bottom))
    cropped_image.load()
    return cropped_image


def transform_colour_channels(image, means, std_deviations):
    np_scaled_image = np.array(image) / 255
    normalized_image = (np_scaled_image - means) / std_deviations
    transposed_image = normalized_image.transpose(2, 0, 1)
    return transposed_image


def process_image(image_file_name, crop_width, crop_height):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    with Image.open(image_file_name) as image:
        # print(f"Original Image format: {image.format}, size: {image.size}, mode: {image.mode}")
        resized_image = resize_keep_ratio(image, 256)
        # print(f"Resized Image format: {resized_image.format}, size: {resized_image.size}, mode: {resized_image.mode}")
        cropped_image = center_crop(resized_image, crop_width, crop_height)
        # print(f"Cropped Image format: {cropped_image.format}, size: {cropped_image.size}, mode: {cropped_image.mode}")
        processed_image_nparray = transform_colour_channels(cropped_image,
                                                            np.array([0.485, 0.456, 0.406]),
                                                            np.array([0.229, 0.224, 0.225]))

        return torch.from_numpy(processed_image_nparray.reshape(1, 3, crop_width, crop_height)).float()


def predict_flower(model, image_tensor):
    model.eval()
    return torch.exp(model.forward(image_tensor))


def process_results(results, topk, categories_to_index_dic, category_names_file):
    probabilities, indexes = results.topk(topk, dim=1)
    probabilities, indexes = probabilities.flatten() * 100, indexes.flatten()
    categories = get_categories_from_indexes(categories_to_index_dic, indexes)
    if category_names_file is None:
        names = [" - " for _ in categories]
    else:
        names = get_names_from_categories(categories, category_names_file)

    for name, category, probability in zip(names, categories, probabilities):
        print(f"Flower name: {name}, Category: {category}, Probability: {probability:.2f}%")


def get_categories_from_indexes(dictionary, top_indexes):
    inverted_dict = dict(map(reversed, dictionary.items()))
    return [inverted_dict[int(index)] for index in top_indexes]


def get_names_from_categories(categories, category_names_dic):
    with open(category_names_dic, 'r') as file:
        category_names = json.load(file)
        return [category_names[cat] for cat in categories]
