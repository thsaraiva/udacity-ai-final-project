from common_helper_functions import get_device_type, load_pre_trained_network_model
from train_command_line_args_utils import parse_command_line_arguments
from training_helper_functions import get_data_loaders, train_model

arguments = parse_command_line_arguments()

model_checkpoint = {"arch": arguments.arch,
                    "output_units": arguments.output_units,
                    "hidden_units": arguments.hidden_units,
                    "batch_size": arguments.batch_size,
                    "epochs": arguments.epochs,
                    "learning_rate": arguments.learning_rate,
                    "checkpoint_save_dir": arguments.save_dir}
# Load images data sets
training_data_loader, validation_data_loader = get_data_loaders(arguments.data_dir, arguments.batch_size)
model_checkpoint["class_to_idx"] = training_data_loader.dataset.class_to_idx

# Make code agnostic to use CPU or GPU(if available)
device = get_device_type(arguments.use_gpu)

# Load pre-trained network model
model, optimizer, criterion = load_pre_trained_network_model(model_checkpoint["arch"],
                                                             model_checkpoint["hidden_units"],
                                                             model_checkpoint["output_units"],
                                                             model_checkpoint["learning_rate"])

model = model.to(device)

# train model
model = train_model(model, optimizer, criterion, arguments.epochs, training_data_loader, validation_data_loader, device,
                    model_checkpoint)
