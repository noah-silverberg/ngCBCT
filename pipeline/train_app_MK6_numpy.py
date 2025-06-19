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


# Set up logging
log = logging.getLogger(__name__)

use_cuda = torch.cuda.is_available()
if not use_cuda:
    raise RuntimeError(
        "CUDA is not available. Please check your PyTorch installation or GPU setup."
    )
device = torch.device("cuda:0" if use_cuda else "cpu")
log.debug("Using CUDA; {} devices.".format(torch.cuda.device_count()))


def init_model(config: dict):
    """Initialize the CNN model and move it to the GPU."""
    model = getattr(network_instance, config["network_name"])()
    log.debug(f"Network Selected: {config['network_name']}")

    model = model.to(device)

    return model


def init_loss(config: dict):
    loss = nn.SmoothL1Loss()
    log.debug("Loss function: SmoothL1Loss")
    return loss


def init_optimizer(learning_rate, config: dict, model):
    if config["optimizer"] == "SGD":
        log.debug(
            f"Optimizer: SGD with learning rate {learning_rate}, momentum {config['momentum_SGD']}, weight decay {config['weight_decay_SGD']}"
        )
        return SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=config["momentum_SGD"],
            weight_decay=config["weight_decay_SGD"],
        )
    elif config["optimizer"] == "Adam":
        log.debug(f"Optimizer: Adam with learning rate {learning_rate}")
        return Adam(model.parameters(), lr=learning_rate)
    elif config["optimizer"] == "NAdam":
        log.debug(
            f"Optimizer: NAdam with learning rate {learning_rate}, betas {config['betas_NAdam']}, momentum_decay {config['momentum_decay_NAdam']}"
        )
        return NAdam(
            model.parameters(),
            lr=learning_rate,
            betas=config["betas_NAdam"],
            momentum_decay=config["momentum_decay_NAdam"],
        )

    raise NotImplementedError(
        f"Optimizer {config['optimizer']} is not implemented. Supported optimizers are: SGD, Adam, NAdam."
    )


def init_tensorboard_writers(config: dict, time_str):
    """Initialize TensorBoard writers for training and validation."""
    log_dir = os.path.join("runs", config["model_version"], time_str)

    trn_writer = SummaryWriter(
        log_dir=log_dir + "-trn_cls-" + config["tensor_board_comment"]
    )
    val_writer = SummaryWriter(
        log_dir=log_dir + "-val_cls-" + config["tensor_board_comment"]
    )

    log.debug(f"TensorBoard writers initialized at {log_dir}")

    return trn_writer, val_writer


def get_data_sub_path(
    config: dict,
    sample,
    truth,
):
    """Get the sub-path for the data based on the data type."""
    augment = config["augment"]
    domain = config["domain"]
    scan_type = config["scan_type"]
    input_type = config["input_type"]

    # We need to know the input type and augmentation setting to get the right data
    if augment:
        sub_path = f"{domain}_{'gated' if truth else 'ng'}_{scan_type}_{sample}_aug.npy"
    else:
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

    log.debug(
        f"Data type: {domain} domain {sample} data for {scan_type} {'with' if augment else 'without'} augmentation {'with input type PL' if input_type == 'PL' else ''}"
    )

    return sub_path


def init_dataloader(config: dict, sample):
    """Initialize the DataLoader for a specific sample ('TRAIN', 'VALIDATION', or 'TEST')."""
    # Get the path to the training data within the aggregation directory
    images_sub_path = get_data_sub_path(config, sample, False)
    images_path = os.path.join(config["AGG_DIR"], images_sub_path)
    log.debug(f"{sample} images path: {images_path}")

    # Get the path to the ground truth data
    truth_images_sub_path = get_data_sub_path(config, sample, True)
    truth_images_path = os.path.join(config["AGG_DIR"], truth_images_sub_path)
    log.debug(f"{sample} ground truth images path: {truth_images_path}")

    # Load the dataset
    dataset = PairNumpySet(images_path, truth_images_path)
    log.debug(
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
        pin_memory=bool_shuffle,  # TODO why is this the same as shuffle?
        shuffle=bool_shuffle,
    )
    log.debug(
        f"{sample} dataloader initialized with {len(dataloader)} batches of size {n_batches}, with {n_workers} workers, shuffle={bool_shuffle}, and pin_memory={bool_shuffle}."
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

        # Set logging level
        if DEBUG:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)

        # Current time string for tensorboard
        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

        # Initialize model, loss, and optimizer
        self.model = init_model(self.config)
        self.criterion = init_loss(self.config)

    def main(self):
        log.debug("Starting {}, {}".format(type(self).__name__, self.config))

        # Get the dataloaders for training and validation
        train_dl = init_dataloader(self.config, "TRAIN")
        log.debug(f"Initialized training dataloader with {len(train_dl)} batches.")
        val_dl = init_dataloader(self.config, "VALIDATION")
        log.debug(f"Initialized validation dataloader with {len(val_dl)} batches.")

        # Initialize the tensorboard writers if enabled
        if self.config["tensor_board"]:
            self.trn_writer, self.val_writer = init_tensorboard_writers(
                self.config, self.time_str
            )

        # Make sure we have a valid directory for saving the model/checkpoints
        save_directory = os.path.join(
            self.config["MODEL_DIR"], self.config["model_version"]
        )
        ensure_dir(save_directory)

        # Summarize training settings
        log.info("TRAINING SETTINGS:")
        log.info(self.config)

        # Save start time of training
        training_start_time = time.time()

        # Keep track of the average training and validation loss for each epoch
        avg_train_loss_values = []
        avg_val_loss_values = []

        # Train for the chosen number of epochs
        log.info("STARTING TRAINING...")
        # NOTE: epoch_ndx is 1-indexed!!
        for epoch_ndx in range(1, self.config["epochs"] + 1):
            # Set the model to training mode
            self.model.train()
            log.debug("Model set to training mode for training.")

            if isinstance(self.config["learning_rate"], float):
                # If learning rate is a float, use it directly
                learning_rate = self.config["learning_rate"]
            else:
                # Otherwise, use the learning rate schedule
                learning_rate = self.config["learning_rate"][
                    min(epoch_ndx - 1, len(self.config["learning_rate"]) - 1)
                ]

            log.debug(f"Epoch: {epoch_ndx}, Learning Rate: {learning_rate}")

            # Initialize the optimizer with the current learning rate
            self.optimizer = init_optimizer(learning_rate, self.config, self.model)

            # Monitor training loss
            epoch_total_train_loss = 0.0

            # Start time of the training phase
            epoch_train_start_time = time.time()

            # Loop throught the batches in the training dataloader
            for train_set in tqdm(train_dl, desc=f"Epoch {epoch_ndx} Training"):

                # Extract the input and ground truth, and send to GPU
                train_inputs = train_set[0].to(device)
                train_truths = train_set[1].to(device)

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

            log.info(
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
                log.warning(
                    "Model set to training mode for validation -- please ensure this is intended!"
                )
                self.model.train()
            else:
                # Put the model in eval mode
                log.debug("Model set to evaluation mode for validation.")
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

                log.info(
                    "Epoch: {} \tAvg Validation Loss: {:.6f}  \tTime(s) {:.4f}".format(
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
                save_path = os.path.join(save_directory, "epoch-%d.pkl" % epoch_ndx)
                torch.save(
                    {
                        "epoch": epoch_ndx,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    },
                    save_path,
                )
                log.info(
                    "Checkpoint saved at epoch {}: {}".format(epoch_ndx, save_path)
                )

        log.info(
            "Training finished, took {:.2f}s".format(time.time() - training_start_time)
        )

        # Training is done
        log.info("Saving training results...")
        # Save the final model state
        model_path = os.path.join(
            save_directory, "FINAL-epoch-%d.pth" % self.config["epochs"]
        )
        torch.save(
            self.model.state_dict(),
            model_path,
        )
        log.info(f"Model saved to {model_path}")
        # Save the training and validation loss values
        torch.save(
            avg_train_loss_values,
            os.path.join(save_directory, "train_loss.pth"),
        )
        torch.save(
            avg_val_loss_values,
            os.path.join(save_directory, "validation_loss.pth"),
        )
        log.info(f"Training and validation loss saved to {save_directory}")

        # Clean up tensor board writers
        if self.config["tensor_board"]:
            self.trn_writer.flush()
            self.trn_writer.close()
            self.val_writer.flush()
            self.val_writer.close()

        # Clean up memory
        log.debug("Cleaning up memory...")
        gc.collect()

        # We don't want to have the whole pipeline break if we fail to clean up memory
        # So we catch any exceptions that might occur during cleanup
        try:
            del self.model
            del self.trn_writer, self.val_writer
            del self.config
            del self.optimizer
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
            log.error(f"Error during memory cleanup: {e}")

        with torch.no_grad():
            torch.cuda.empty_cache()

        log.debug("Memory cleanup done.")
        log.info("Training application finished successfully.")
