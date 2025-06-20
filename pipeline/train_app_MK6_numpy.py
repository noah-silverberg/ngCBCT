import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # specify which GPU(s) to be used

import datetime
import time
import gc
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, NAdam
from .dsets import PairNumpySet
from . import network_instance
import logging
from tqdm import tqdm
from .utils import ensure_dir
import ast


# Set up logging
logger = logging.getLogger("pipeline")

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda:0")
    logger.debug("Using CUDA; {} devices.".format(torch.cuda.device_count()))
else:
    device = torch.device("cpu")
    logger.error(
        "CUDA is not available. Please check your PyTorch installation or GPU setup. Proceeding with CPU...this will be slow."
    )


def init_model(config: dict):
    """Initialize the CNN model and move it to the GPU."""
    model = getattr(network_instance, config["network_name"])()
    model = model.to(device)

    return model


def init_loss(config: dict):
    loss = nn.SmoothL1Loss()
    return loss


def init_optimizer(config: dict, model):
    if isinstance(config["learning_rate"], float):
        # If learning rate is a float, convert it to a list
        learning_rates = [config["learning_rate"]] * config["epochs"]
    else:
        learning_rates = config["learning_rate"]

    if config["optimizer"] == "SGD":
        logger.debug(
            f"Optimizer: SGD with learning rate {learning_rates[0]}, momentum {config['momentum_SGD']}, weight decay {config['weight_decay_SGD']}"
        )
        optimizer = SGD(
            model.parameters(),
            lr=learning_rates[0],
            momentum=config["momentum_SGD"],
            weight_decay=config["weight_decay_SGD"],
        )
    elif config["optimizer"] == "Adam":
        logger.debug(f"Optimizer: Adam with learning rate {learning_rates[0]}")
        optimizer = Adam(model.parameters(), lr=learning_rates[0])
    elif config["optimizer"] == "NAdam":
        logger.debug(
            f"Optimizer: NAdam with learning rate {learning_rates[0]}, betas {config['betas_NAdam']}, momentum_decay {config['momentum_decay_NAdam']}"
        )
        optimizer = NAdam(
            model.parameters(),
            lr=learning_rates[0],
            betas=config["betas_NAdam"],
            momentum_decay=config["momentum_decay_NAdam"],
        )
    else:
        raise NotImplementedError(
            f"Optimizer {config['optimizer']} is not implemented. Supported optimizers are: SGD, Adam, NAdam."
        )

    # Initialize the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: learning_rates[min(epoch, len(learning_rates) - 1)]
        / learning_rates[0],
    )

    return optimizer, scheduler


def init_tensorboard_writers(config: dict, time_str):
    """Initialize TensorBoard writers for training and validation."""
    log_dir = os.path.join("runs", config["model_version"], time_str)

    trn_writer = SummaryWriter(
        log_dir=log_dir + "-trn_cls-" + config["tensor_board_comment"]
    )
    val_writer = SummaryWriter(
        log_dir=log_dir + "-val_cls-" + config["tensor_board_comment"]
    )

    logger.debug(f"TensorBoard writers initialized at {log_dir}")

    return trn_writer, val_writer


def get_data_sub_path(
    config: dict,
    sample,
    truth,
):
    """Get the sub-path for the data based on the data type."""
    domain = config["domain"]
    scan_type = config["scan_type"]
    input_type = config["input_type"]

    # We need to know the input type to get the right data
    sub_path = f"{domain}_{'gated' if truth else 'ng'}_{scan_type}_{sample}.npy"

    # If the input is PL we just add "_PL" to the end of the file name
    if domain == "IMAG":
        if input_type == "PL":
            # Add "_PL" to the end of the file name
            sub_path = sub_path.replace(".npy", "_PL.npy")
        elif input_type != "FDK":
            raise ValueError(
                f"Input type {input_type} is not supported. Supported input types are: FDK, PL."
            )

    return sub_path


def init_dataloader(config: dict, sample):
    """Initialize the DataLoader for a specific sample ('TRAIN', 'VALIDATION', or 'TEST')."""
    # Get the path to the training data within the aggregation directory
    images_sub_path = get_data_sub_path(config, sample, False)
    images_path = os.path.join(config["AGG_DIR"], images_sub_path)
    logger.debug(f"{sample} images path: {images_path}")

    # Get the path to the ground truth data
    truth_images_sub_path = get_data_sub_path(config, sample, True)
    truth_images_path = os.path.join(config["AGG_DIR"], truth_images_sub_path)
    logger.debug(f"{sample} ground truth images path: {truth_images_path}")

    # Load the dataset
    dataset = PairNumpySet(images_path, truth_images_path, device)
    logger.debug(
        f"{sample} dataset loaded with {len(dataset)} samples, each with shape {dataset[0][0].shape}."
    )

    n_batches = config["batch_size"]
    n_workers = config["num_workers"]
    bool_shuffle = config["shuffle"]

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=n_batches,
        num_workers=n_workers,
        pin_memory=False,
        shuffle=bool_shuffle,
    )
    logger.debug(
        f"{sample} dataloader initialized with {len(dataloader)} batches of size {n_batches}, with {n_workers} workers, shuffle={bool_shuffle}, and pin_memory={False}."
    )

    return dataloader


class TrainingApp:
    def __init__(self, config, domain, DEBUG, MODEL_DIR, AGG_DIR):
        if domain == "PROJ":
            self.config = config["PD_settings"]
        elif domain == "IMAG":
            self.config = config["ID_settings"]
        else:
            raise ValueError(
                f"Domain {domain} is not supported. Supported domains are: PROJ, IMAG."
            )

        self.config["domain"] = domain
        self.config["MODEL_DIR"] = MODEL_DIR
        self.config["AGG_DIR"] = AGG_DIR

        # We need to correct some of the config settings
        # since some values are not read correctly
        if isinstance(self.config["learning_rate"], str):
            self.config["learning_rate"] = ast.literal_eval(
                self.config["learning_rate"]
            )
        elif isinstance(self.config["learning_rate"], list):
            # If learning rate is a list, evaluate each element
            self.config["learning_rate"] = [
                ast.literal_eval(lr) for lr in self.config["learning_rate"]
            ]
        if isinstance(self.config["grad_max"], str):
            self.config["grad_max"] = ast.literal_eval(self.config["grad_max"])
        if isinstance(self.config["betas_NAdam"], str):
            self.config["betas_NAdam"] = ast.literal_eval(self.config["betas_NAdam"])
        if isinstance(self.config["momentum_decay_NAdam"], str):
            self.config["momentum_decay_NAdam"] = ast.literal_eval(
                self.config["momentum_decay_NAdam"]
            )
        if isinstance(self.config["momentum_SGD"], str):
            self.config["momentum_SGD"] = tuple(self.config["momentum_SGD"])
        if isinstance(self.config["weight_decay_SGD"], str):
            self.config["weight_decay_SGD"] = ast.literal_eval(
                self.config["weight_decay_SGD"]
            )

        # Set logging level
        if DEBUG:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Current time string for tensorboard
        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

        # Initialize model, loss, and optimizer
        self.model = init_model(self.config)
        self.criterion = init_loss(self.config)

    def main(self):
        # We will save all the outputs from training in this directory
        save_directory = os.path.join(
            self.config["MODEL_DIR"], self.config["model_version"]
        )
        # Make sure we're not accidentally overwriting an existing model version
        if os.path.exists(save_directory):
            logger.error(
                f"Save directory {save_directory} already exists. Please choose a different model version or delete the existing directory, or manually delete this."
            )
            return
        # Create the directory
        ensure_dir(save_directory)

        logger.debug("Starting {}, {}".format(type(self).__name__, self.config))

        # Get the dataloaders for training and validation
        train_dl = init_dataloader(self.config, "TRAIN")
        val_dl = init_dataloader(self.config, "VALIDATION")

        # Initialize the tensorboard writers if enabled
        if self.config["tensor_board"]:
            self.trn_writer, self.val_writer = init_tensorboard_writers(
                self.config, self.time_str
            )

        # Initialize the optimizer and learning rate scheduler
        self.optimizer, self.scheduler = init_optimizer(self.config, self.model)

        # Summarize training settings
        logger.info("TRAINING SETTINGS:")
        logger.info(self.config)

        # Save start time of training
        training_start_time = time.time()

        # Keep track of the average training and validation loss for each epoch
        avg_train_loss_values = []
        avg_val_loss_values = []

        # Train for the chosen number of epochs
        logger.info("STARTING TRAINING...")
        # NOTE: epoch_ndx is 1-indexed!!
        for epoch_ndx in range(1, self.config["epochs"] + 1):
            logger.debug(
                f"Learning rate is {self.scheduler.get_last_lr()[0]} at epoch {epoch_ndx}."
            )

            # Set the model to training mode
            self.model.train()
            logger.debug("Model set to training mode for training.")

            # Monitor training loss
            epoch_total_train_loss = 0.0

            # Start time of the training phase
            epoch_train_start_time = time.time()

            # Loop throught the batches in the training dataloader
            for train_set in tqdm(train_dl, desc=f"Epoch {epoch_ndx} Training"):

                # Extract the input and ground truth (they are already on GPU)
                train_inputs = train_set[0].float()
                train_truths = train_set[1].float()

                # Zero out the gradients, do a forward pass, compute the loss, and backpropagate
                self.optimizer.zero_grad()
                train_outputs = self.model(train_inputs)
                train_loss = self.criterion(train_outputs, train_truths)
                train_loss.backward()

                # Clip gradients if needed
                if self.config["grad_clip"]:
                    torch.nn.utils.clip_grad_value_(
                        self.model.parameters(), clip_value=self.config["grad_max"]
                    )

                # Parameter update
                self.optimizer.step()

                # Update epoch training loss
                epoch_total_train_loss += train_loss.item() * train_inputs.size(0)

            # Save avg training statistics
            epoch_avg_train_loss = epoch_total_train_loss / len(train_dl.dataset)
            avg_train_loss_values.append(epoch_avg_train_loss)

            # Store loss in TensorBoard if enabled
            if self.config["tensor_board"]:
                self.trn_writer.add_scalar("Loss", epoch_avg_train_loss, epoch_ndx)

            logger.info(
                "Epoch: {} \tAvg Training Loss: {:.6f}  \tTime(s) {:.4f}".format(
                    epoch_ndx,
                    epoch_avg_train_loss,
                    time.time() - epoch_train_start_time,
                )
            )

            # Validation phase
            if self.config["train_during_inference"]:
                # If training during inference, set the model to training mode
                # for MC dropout, for example
                logger.warning(
                    "Model set to training mode for validation -- please ensure this is intended!"
                )
                self.model.train()
            else:
                # Put the model in eval mode
                logger.debug("Model set to evaluation mode for validation.")
                self.model.eval()

            # We don't need to compute gradients during validation
            with torch.no_grad():

                # Monitor validation loss
                epoch_total_val_loss = 0.0

                # Start time of the validation phase
                epoch_val_start_time = time.time()

                # Loop through the batches in the validation dataloader
                for val_set in tqdm(val_dl, desc=f"Epoch {epoch_ndx} Validation"):

                    # Extract the input and ground truth, and send to GPU
                    val_inputs = val_set[0].to(device)
                    val_truths = val_set[1].to(device)

                    # Do a forward pass and calculate the loss
                    val_outputs = self.model(val_inputs)
                    val_loss = self.criterion(val_outputs, val_truths)

                    # Update epoch validation loss
                    epoch_total_val_loss += val_loss.item() * val_inputs.size(0)

                # Save avg validation statistics
                epoch_avg_val_loss = epoch_total_val_loss / len(val_dl.dataset)
                avg_val_loss_values.append(epoch_avg_val_loss)

                # Store loss in TensorBoard if enabled
                if self.config["tensor_board"]:
                    self.val_writer.add_scalar("Loss", epoch_avg_val_loss, epoch_ndx)

                logger.info(
                    "Epoch: {} \tAvg Validation Loss: {:.6f}  \tTime(s) {:.4f}\n".format(
                        epoch_ndx,
                        epoch_avg_val_loss,
                        time.time() - epoch_val_start_time,
                    )
                )

            # Save model if needed (either at checkpoint or at end of training)
            if (
                epoch_ndx % self.config["checkpoint_save_step"] == 0
                or epoch_ndx == self.config["epochs"]
            ):
                save_path = os.path.join(
                    save_directory,
                    f"epoch-{epoch_ndx:02}_{'ID' if self.config['domain'] == 'IMAG' else 'PD'}.pkl",
                )
                torch.save(
                    {
                        "epoch": epoch_ndx,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    },
                    save_path,
                )
                logger.info(
                    "Checkpoint saved at epoch {}: {}\n".format(epoch_ndx, save_path)
                )

                # Delete most recent checkpoint to save space
                if epoch_ndx > self.config["checkpoint_save_step"]:
                    old_save_path = os.path.join(
                        save_directory,
                        f"epoch-{epoch_ndx - self.config['checkpoint_save_step']:02}_{'ID' if self.config['domain'] == 'IMAG' else 'PD'}.pkl",
                    )
                    if os.path.exists(old_save_path):
                        os.remove(old_save_path)
                        logger.debug(
                            "Old checkpoint removed from epoch {}: {}".format(
                                epoch_ndx - self.config["checkpoint_save_step"],
                                old_save_path,
                            )
                        )

            # Update the learning rate
            self.scheduler.step()

        logger.info(
            "Training finished, took {:.2f}s\n".format(
                time.time() - training_start_time
            )
        )

        # Training is done
        logger.info("Saving training results...")
        # Save the final model state
        model_path = os.path.join(
            save_directory,
            f"{self.config['network_name']}_{self.config['model_version']}_DS{self.config['data_version']}_{self.config['scan_type']}_{'ID' if self.config['domain'] == 'IMAG' else 'PD'}.pth",
        )
        torch.save(
            self.model.state_dict(),
            model_path,
        )
        logger.info(f"Model saved to {model_path}")
        # Save the training and validation loss values
        torch.save(
            avg_train_loss_values,
            os.path.join(
                save_directory,
                f"train_loss_{'ID' if self.config['domain'] == 'IMAG' else 'PD'}.pth",
            ),
        )
        torch.save(
            avg_val_loss_values,
            os.path.join(
                save_directory,
                f"validation_loss_{'ID' if self.config['domain'] == 'IMAG' else 'PD'}.pth",
            ),
        )
        logger.info(f"Training and validation loss saved to {save_directory}\n")

        # Clean up tensor board writers
        if self.config["tensor_board"]:
            self.trn_writer.flush()
            self.trn_writer.close()
            self.val_writer.flush()
            self.val_writer.close()

        # Clean up memory
        logger.debug("Cleaning up memory...")
        gc.collect()

        # We don't want to have the whole pipeline break if we fail to clean up memory
        # So we catch any exceptions that might occur during cleanup
        try:
            del self.model
            if self.config["tensor_board"]:
                del self.trn_writer, self.val_writer
            del self.config
            del self.optimizer, self.scheduler
            del self.criterion
            del train_inputs, train_truths
            del val_inputs, val_truths
            del train_dl, val_dl
            del train_outputs, val_outputs
            del train_loss, val_loss
            del epoch_total_train_loss, epoch_total_val_loss
            del epoch_avg_train_loss, epoch_avg_val_loss
            del avg_train_loss_values, avg_val_loss_values
            del self.time_str
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")

        with torch.no_grad():
            torch.cuda.empty_cache()

        logger.debug("Memory cleanup done.")
        logger.debug("Training application finished successfully.")
