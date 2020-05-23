import torch

from predict_command_line_args_utils import parse_command_line_arguments
from common_helper_functions import load_pre_trained_network_model, get_device_type
from prediction_helper_functions import process_image, process_results, predict_flower

arguments = parse_command_line_arguments()

device = get_device_type(arguments.use_gpu)

image_tensor = process_image(arguments.flower_image_file, 224, 224)
image_tensor.to(device)

model_checkpoint = torch.load(arguments.checkpoint_file, map_location=device)
# print(f"best_acc: {model_checkpoint['best_acc']}")

model, _, _ = load_pre_trained_network_model(model_checkpoint["arch"],
                                             model_checkpoint["hidden_units"],
                                             model_checkpoint["output_units"],
                                             model_checkpoint["learning_rate"],
                                             is_training=False)

model.load_state_dict(model_checkpoint['model_state_dict'])
model.to(device)

results = predict_flower(model, image_tensor)

process_results(results, arguments.topk, model_checkpoint["class_to_idx"], arguments.category_names)
