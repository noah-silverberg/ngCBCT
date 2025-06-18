import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # specify which GPU(s) to be used

import argparse
import datetime
import time
import math
import sys
import gc
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, NAdam
from torch.utils.data import DataLoader
from .dsets import PairSet, PairNumpySet, CTSet
from . import network_instance
import logging
from dataclasses import dataclass
from tqdm import tqdm
from typing import Union
from .utils import ensure_dir
from .config import (
    AGG_DIR,
    MODEL_DIR,
    DEBUG,
    PD_epochs,
    PD_learning_rate,
    PD_network_name,
    PD_model_name,
    PD_batch_size,
    PD_optimizer,
    PD_num_workers,
    PD_shuffle,
    PD_grad_clip,
    PD_grad_max,
    PD_betas_NAdam,
    PD_momentum_decay_NAdam,
    PD_momentum_SGD,
    PD_weight_decay_SGD,
    PD_checkpoint_save_freq,
    PD_tensor_board,
    PD_tensor_board_comment,
    PD_train_during_inference,
    ID_epochs,
    ID_learning_rate,
    ID_network_name,
    ID_model_name,
    ID_batch_size,
    ID_optimizer,
    ID_num_workers,
    ID_shuffle,
    ID_grad_clip,
    ID_grad_max,
    ID_betas_NAdam,
    ID_momentum_decay_NAdam,
    ID_momentum_SGD,
    ID_weight_decay_SGD,
    ID_augment,
    ID_checkpoint_save_freq,
    ID_tensor_board,
    ID_tensor_board_comment,
    ID_train_during_inference,
)


# Set up logging
log = logging.getLogger(__name__)

use_cuda = torch.cuda.is_available()
if not use_cuda:
    raise RuntimeError(
        "CUDA is not available. Please check your PyTorch installation or GPU setup."
    )
device = torch.device("cuda:0" if use_cuda else "cpu")
log.debug("Using CUDA; {} devices.".format(torch.cuda.device_count()))


# TODO add PL back
@dataclass
class TrainingArgs:
    epoch: int
    learning_rate: Union[float, list]
    network: str
    model_name: str
    batch_size: int
    optimizer: str
    num_workers: int
    shuffle: bool
    grad_clip: bool
    grad_max: float
    betas_NAdam: tuple
    momentum_decay_NAdam: float
    momentum_SGD: float
    weight_decay_SGD: float
    checkpoint_save_step: int
    tensor_board: bool
    comment: str
    augment: bool
    scan_type: str
    domain: str
    train_during_inference: bool

    def __str__(self):  # nice printing of the training args
        lines = [
            f"epoch: {self.epoch}",
            f"learning_rate: {self.learning_rate}",
            f"network: {self.network}",
            f"model_name: {self.model_name}",
            f"batch_size: {self.batch_size}",
            f"optimizer: {self.optimizer}",
            f"num_workers: {self.num_workers}",
            f"shuffle: {self.shuffle}",
            f"grad_clip: {self.grad_clip}",
        ]

        # include optimizer-specific args
        if self.grad_clip:
            lines.append(f"grad_max: {self.grad_max}")
        if self.optimizer == "NAdam":
            lines += [
                f"betas_NAdam: {self.betas_NAdam}",
                f"momentum_decay_NAdam: {self.momentum_decay_NAdam}",
            ]
        elif self.optimizer == "SGD":
            lines += [
                f"momentum_SGD: {self.momentum_SGD}",
                f"weight_decay_SGD: {self.weight_decay_SGD}",
            ]

        lines += [
            f"checkpoint_save_step: {self.checkpoint_save_step}",
            f"tensor_board: {self.tensor_board}",
            f"comment: {self.comment}",
            f"augment: {self.augment}",
            f"scan_type: {self.scan_type}",
            f"domain: {self.domain}",
            f"train_during_inference: {self.train_during_inference}",
        ]

        return "\n".join(lines)


def get_training_args(domain, scan_type):
    if domain == "PROJ":
        # Create dataclass instance
        args = TrainingArgs(
            epoch=PD_epochs,
            learning_rate=PD_learning_rate,
            network=PD_network_name,
            model_name=PD_model_name,
            batch_size=PD_batch_size,
            optimizer=PD_optimizer,
            num_workers=PD_num_workers,
            shuffle=PD_shuffle,
            grad_clip=PD_grad_clip,
            grad_max=PD_grad_max,
            betas_NAdam=PD_betas_NAdam,
            momentum_decay_NAdam=PD_momentum_decay_NAdam,
            momentum_SGD=PD_momentum_SGD,
            weight_decay_SGD=PD_weight_decay_SGD,
            checkpoint_save_step=PD_checkpoint_save_freq,
            tensor_board=PD_tensor_board,
            comment=PD_tensor_board_comment,
            augment=False,  # PD does not use augmentation
            scan_type=scan_type,
            domain=domain,
            train_during_inference=PD_train_during_inference,
        )
    elif domain == "IMAG":
        args = TrainingArgs(
            epoch=ID_epochs,
            learning_rate=ID_learning_rate,
            network=ID_network_name,
            model_name=ID_model_name,
            batch_size=ID_batch_size,
            optimizer=ID_optimizer,
            num_workers=ID_num_workers,
            shuffle=ID_shuffle,
            grad_clip=ID_grad_clip,
            grad_max=ID_grad_max,
            betas_NAdam=ID_betas_NAdam,
            momentum_decay_NAdam=ID_momentum_decay_NAdam,
            momentum_SGD=ID_momentum_SGD,
            weight_decay_SGD=ID_weight_decay_SGD,
            checkpoint_save_step=ID_checkpoint_save_freq,
            tensor_board=ID_tensor_board,
            comment=ID_tensor_board_comment,
            augment=ID_augment,
            scan_type=scan_type,
            domain=domain,
            train_during_inference=ID_train_during_inference,
        )
    else:
        raise ValueError(
            f"Domain {domain} is not supported. Supported domains are: PROJ, IMAG."
        )

    return args


def init_model(args: TrainingArgs):
    """Initialize the CNN model and move it to the GPU."""
    model = getattr(network_instance, args.network)()
    log.debug(f"Network Selected: {args.network}")

    model = model.to(device)

    return model


def init_loss(args: TrainingArgs):
    loss = nn.SmoothL1Loss()
    log.debug("Loss function: SmoothL1Loss")
    return loss


def init_optimizer(learning_rate, args: TrainingArgs, model):
    if args.optimizer == "SGD":
        log.debug(
            f"Optimizer: SGD with learning rate {learning_rate}, momentum {args.momentum_SGD}, weight decay {args.weight_decay_SGD}"
        )
        return SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=args.momentum_SGD,
            weight_decay=args.weight_decay_SGD,
        )
    elif args.optimizer == "Adam":
        log.debug(f"Optimizer: Adam with learning rate {learning_rate}")
        return Adam(model.parameters(), lr=learning_rate)
    elif args.optimizer == "NAdam":
        log.debug(
            f"Optimizer: NAdam with learning rate {learning_rate}, betas {args.betas_NAdam}, momentum_decay {args.momentum_decay_NAdam}"
        )
        return NAdam(
            model.parameters(),
            lr=learning_rate,
            betas=args.betas_NAdam,
            momentum_decay=args.momentum_decay_NAdam,
        )

    raise NotImplementedError(
        f"Optimizer {args.optimizer} is not implemented. Supported optimizers are: SGD, Adam, NAdam."
    )


def init_tensorboard_writers(args: TrainingArgs, time_str):
    """Initialize TensorBoard writers for training and validation."""
    log_dir = os.path.join("runs", args.model_name, time_str)

    trn_writer = SummaryWriter(log_dir=log_dir + "-trn_cls-" + args.comment)
    val_writer = SummaryWriter(log_dir=log_dir + "-val_cls-" + args.comment)

    log.debug(f"TensorBoard writers initialized at {log_dir}")

    return trn_writer, val_writer


def get_data_sub_path(
    args: TrainingArgs,
    sample,
    truth,
):
    """Get the sub-path for the data based on the data type."""
    augment = args.augment
    domain = args.domain
    scan_type = args.scan_type

    # We need to know the input type and augmentation setting to get the right data
    if augment:
        sub_path = f"{domain}_{'gated' if truth else 'ng'}_{scan_type}_{sample}_aug.npy"
    else:
        sub_path = f"{domain}_{'gated' if truth else 'ng'}_{scan_type}_{sample}.npy"

    log.debug(
        f"Data type: {domain} domain {sample} data for {scan_type} {'with' if augment else 'without'} augmentation"
    )

    return sub_path


def init_dataloader(args: TrainingArgs, sample):
    """Initialize the DataLoader for a specific sample ('TRAIN', 'VALIDATION', or 'TEST')."""
    # Get the path to the training data within the aggregation directory
    images_sub_path = get_data_sub_path(args, sample, False)
    images_path = os.path.join(AGG_DIR, images_sub_path)
    log.debug(f"{sample} images path: {images_path}")

    # Get the path to the ground truth data
    truth_images_sub_path = get_data_sub_path(args, sample, True)
    truth_images_path = os.path.join(AGG_DIR, truth_images_sub_path)
    log.debug(f"{sample} ground truth images path: {truth_images_path}")

    # Load the dataset
    dataset = PairNumpySet(images_path, truth_images_path)
    log.debug(
        f"{sample} dataset loaded with {len(dataset)} samples, each with shape {dataset[0][0].shape}."
    )

    n_batches = args.batch_size
    n_workers = args.num_workers
    bool_shuffle = args.shuffle

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
    def __init__(self, domain, scan_type):
        self.args = get_training_args(
            domain, scan_type
        )  # TODO implement this as a yaml file?

        # Set logging level
        if DEBUG:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)

        # Current time string for tensorboard
        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

        # Initialize model, loss, and optimizer
        self.model = init_model(self.args)
        self.criterion = init_loss(self.args)

    def main(self):
        log.debug("Starting {}, {}".format(type(self).__name__, self.args))

        # Get the dataloaders for training and validation
        train_dl = init_dataloader(self.args, "TRAIN")
        log.debug(f"Initialized training dataloader with {len(train_dl)} batches.")
        val_dl = init_dataloader(self.args, "VALIDATION")
        log.debug(f"Initialized validation dataloader with {len(val_dl)} batches.")

        # Initialize the tensorboard writers if enabled
        if self.args.tensor_board:
            self.trn_writer, self.val_writer = init_tensorboard_writers(
                self.args, self.time_str
            )

        # Make sure we have a valid directory for saving the model/checkpoints
        save_directory = os.path.join(MODEL_DIR, self.args.model_name)
        ensure_dir(save_directory)

        # Summarize training settings
        log.info("TRAINING SETTINGS:")
        log.info(self.args)

        # Save start time of training
        training_start_time = time.time()

        # Keep track of the average training and validation loss for each epoch
        avg_train_loss_values = []
        avg_val_loss_values = []

        # Train for the chosen number of epochs
        log.info("STARTING TRAINING...")
        # NOTE: epoch_ndx is 1-indexed!!
        for epoch_ndx in range(1, self.args.epoch + 1):
            # Set the model to training mode
            self.model.train()
            log.debug("Model set to training mode for training.")

            if isinstance(self.args.learning_rate, float):
                # If learning rate is a float, use it directly
                learning_rate = self.args.learning_rate
            else:
                # Otherwise, use the learning rate schedule
                learning_rate = self.args.learning_rate[
                    min(epoch_ndx - 1, len(self.args.learning_rate) - 1)
                ]

            log.debug(f"Epoch: {epoch_ndx}, Learning Rate: {learning_rate}")

            # Initialize the optimizer with the current learning rate
            self.optimizer = init_optimizer(learning_rate, self.args, self.model)

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
                if self.args.grad_clip:
                    torch.nn.utils.clip_grad_value_(
                        self.model.parameters(), clip_value=self.args.grad_max
                    )

                # Parameter update
                self.optimizer.step()

                # Update epoch training loss
                epoch_total_train_loss += train_loss.item() * train_inputs.size(0)

            # Save avg training statistics
            epoch_avg_train_loss = epoch_total_train_loss / len(train_dl.dataset)
            avg_train_loss_values.append(epoch_avg_train_loss)

            # Store loss in TensorBoard if enabled
            if self.args.tensor_board:
                self.trn_writer.add_scalar("Loss", epoch_avg_train_loss, epoch_ndx)

            log.info(
                "Epoch: {} \tAvg Training Loss: {:.6f}  \tTime(s) {:.4f}".format(
                    epoch_ndx,
                    epoch_avg_train_loss,
                    time.time() - epoch_train_start_time,
                )
            )

            # Validation phase
            if self.args.train_during_inference:
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
                if self.args.tensor_board:
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
                epoch_ndx % self.args.checkpoint_save_step == 0
                or epoch_ndx == self.args.epoch
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
            save_directory, "FINAL-epoch-%d.pth" % self.args.epoch
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
        if self.args.tensor_board:
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
            del self.args
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
