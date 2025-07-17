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
from .dsets import PairNumpySet, normalizeInputsClip
from . import network_instance
import logging
from tqdm import tqdm
from .utils import ensure_dir
import ast
from pipeline.paths import Files
from .aggregate_ct import aggregate_saved_recons
import numpy as np


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
    model = getattr(network_instance, config["network_name"])(**config["network_kwargs"])
    model = model.to(device)
    
    logger.debug(model)

    return model

def nig_nll(gamma, nu, alpha, beta, y_true):
    omega = 2.0 * beta * (1.0 + nu)
    term1 = 0.5 * torch.log(torch.pi / nu)
    term2 = -alpha * torch.log(omega)
    term3 = (alpha + 0.5) * torch.log((y_true - gamma) ** 2 * nu + omega)
    term4 = torch.special.gammaln(alpha) - torch.special.gammaln(alpha + 0.5)

    return term1 + term2 + term3 + term4

def nig_reg(gamma, nu, alpha, beta, y_true):
    return torch.abs(y_true - gamma) * (2.0 * nu + alpha)

def nig_u_reg(gamma, nu, alpha, beta, y_true):
    # From Wu et al. "The Evidence Contraction Issue in Deep Evidential Regression: Discussion and Solution"
    return (y_true - gamma) ** 2 * nu * (alpha - 1) / (beta * (nu + 1))

def init_loss(config: dict, is_bayesian: bool, is_evidential: bool = False):
    """
    Initializes the loss function. For BBB, we use SmoothL1Loss with 'sum' reduction.
    This calculates the Negative Log-Likelihood (NLL) term for the entire minibatch,
    which is consistent with how the total KL divergence for the model is calculated.
    """
    if is_bayesian and is_evidential:
        raise ValueError("Both Bayesian and Evidential settings cannot be enabled at the same time.")
    if is_bayesian:
        # Use 'sum' to get the total NLL for the batch.
        loss = nn.SmoothL1Loss(reduction='sum')
    elif is_evidential:
        loss = nn.SmoothL1Loss(reduction='mean')
    else:
        # 'mean' is standard for deterministic models.
        loss = nn.SmoothL1Loss(reduction='mean')
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


def init_dataloader(config: dict, files: Files, sample: str, input_type: str, domain: str, augment_on_fly: bool = False, recon_len: int = None):
    """Initialize the DataLoader for a specific sample ('TRAIN', 'VALIDATION', or 'TEST')."""
    # Get the paths to the training data
    if domain == "PROJ":
        images_path = files.get_projections_aggregate_filepath(sample, gated=False)
        truth_images_path = files.get_projections_aggregate_filepath(sample, gated=True)
    else:
        images_path = files.get_images_aggregate_filepath(input_type, sample, gated=False)
        truth_images_path = files.get_images_aggregate_filepath('fdk', sample, gated=True) # always use FDK for ground truth
    
    logger.debug(f"{sample} images path: {images_path}")
    logger.debug(f"{sample} ground truth images path: {truth_images_path}")

    # Load the dataset
    dataset = PairNumpySet(images_path, truth_images_path, device, augment_on_fly, recon_len)
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
    def __init__(
        self,
        config: dict,
        domain: str,
        DEBUG: bool,
        files: Files,
        checkpoint_epoch: int = None,
        optimizer_state: dict = None,
        network_state: dict = None,
        scans_agg: dict = None,
    ):
        if domain == "PROJ":
            self.config = config["PD_settings"]
        elif domain == "IMAG":
            self.config = config["ID_settings"]
        else:
            raise ValueError(
                f"Domain {domain} is not supported. Supported domains are: PROJ, IMAG."
            )

        self.config["domain"] = domain
        self.files = files
        self.is_bayesian = self.config["is_bayesian"]
        self.is_evidential = self.config["is_evidential"]

        # Store the list of scans needed for on-the-fly aggregation
        if scans_agg is None:
            self.scans_agg_train = None
            self.scans_agg_val = None
        else:
            self.scans_agg_train = scans_agg['TRAIN']
            self.scans_agg_val = scans_agg['VALIDATION']
            
        # --- Type Correction for Config Values ---
        # This section ensures that config values read from a file as strings
        # are converted to their correct Python types (lists, floats, etc.).
        for key, value in self.config.items():
            if isinstance(value, str):
                try:
                    self.config[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    # Keep as string if it's not a valid literal
                    pass
        
        # Special handling for list of learning rates
        if isinstance(self.config.get("learning_rate"), list):
            self.config["learning_rate"] = [
                ast.literal_eval(lr) if isinstance(lr, str) else lr 
                for lr in self.config["learning_rate"]
            ]

        # Set logging level
        if DEBUG:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Current time string for tensorboard
        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

        # Initialize model, loss, and optimizer
        self.model = init_model(self.config)
        self.criterion = init_loss(self.config, self.is_bayesian, self.is_evidential)
        self.optimizer, self.scheduler = init_optimizer(self.config, self.model)
        self.start_epoch = 1

        # Resume from checkpoint logic
        if (
            checkpoint_epoch is not None
            and optimizer_state is not None
            and network_state is not None
        ):
            self._load_checkpoint(checkpoint_epoch, optimizer_state, network_state)

    def _load_checkpoint(
        self, checkpoint_epoch: int, optimizer_state: dict, network_state: dict
    ):
        """Load model, optimizer, and scheduler states from provided checkpoint data."""
        self.model.load_state_dict(network_state)
        self.optimizer.load_state_dict(optimizer_state)
        self.start_epoch = checkpoint_epoch + 1  # Resume from the next epoch
        logger.info(f"Resumed training from checkpoint: Epoch {checkpoint_epoch}")

    def main(self):
        save_dir = self.files.directories.get_model_dir(self.config["model_version"], self.config["domain"], ensure_exists=False)
        if os.path.exists(save_dir) and self.start_epoch == 1:
            logger.error(f"Save directory {save_dir} already exists. Please choose a different model version or delete the existing directory.")
            return

        logger.debug("Starting {}, {}".format(type(self).__name__, self.config))

        if self.scans_agg_train is None:
            logger.debug("Using same dataset for all epochs...")
            # Get the dataloaders for training and validation
            train_dl = init_dataloader(self.config, self.files, "TRAIN", self.config["input_type"], self.config['domain'])
            val_dl = init_dataloader(self.config, self.files, "VALIDATION", self.config["input_type"], self.config['domain'])

        # Initialize the tensorboard writers if enabled
        if self.config["tensor_board"]:
            self.trn_writer, self.val_writer = init_tensorboard_writers(
                self.config, self.time_str
            )

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
        total_epochs = self.config["epochs"]
        for epoch_ndx in range(self.start_epoch, total_epochs + 1):
            if self.scans_agg_train is not None:
                logger.debug(f"Starting on-the-fly aggregation for epoch {epoch_ndx}...")

                input_type = self.config['input_type']

                # Map epoch number (1-50) to passthrough number (0-49)
                passthrough_num = epoch_ndx - 1

                # 1. Get the list of reconstruction file paths for the current passthrough
                ng_train_paths = [self.files.get_recon_filepath(
                                model_version=input_type,
                                patient=patient,
                                scan=scan,
                                scan_type=st,
                                gated=False,
                                passthrough_num=passthrough_num
                            ) for patient, scan, st in self.scans_agg_train]
                ng_val_paths = [self.files.get_recon_filepath(
                                model_version=input_type,
                                patient=patient,
                                scan=scan,
                                scan_type=st,
                                gated=False,
                                passthrough_num=passthrough_num
                            ) for patient, scan, st in self.scans_agg_val]

                # 3. Aggregate and save the reconstructions from the file paths
                ng_agg_train_path = self.files.get_images_aggregate_filepath(input_type, "TRAIN", gated=False)
                aggregate_saved_recons(ng_train_paths, augment=False, out_path=ng_agg_train_path)
                logger.info(f"Aggregated and saved training data for epoch {epoch_ndx}.")

                # 4. Aggregate and save the validation reconstructions
                ng_agg_val_path = self.files.get_images_aggregate_filepath(input_type, "VALIDATION", gated=False)
                aggregate_saved_recons(ng_val_paths, augment=False, out_path=ng_agg_val_path)
                logger.info(f"Aggregated and saved validation data for epoch {epoch_ndx}.")

                # 5. Initialize the training dataloader using the file we just created
                logger.info(f"Initializing training dataloaders for epoch {epoch_ndx}.")
                train_dl = init_dataloader(self.config, self.files, "TRAIN", self.config["input_type"], self.config['domain'], augment_on_fly=True, recon_len=160)
                val_dl = init_dataloader(self.config, self.files, "VALIDATION", self.config["input_type"], self.config['domain'], augment_on_fly=True, recon_len=160)

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

            # Loop through the batches in the training dataloader
            num_batches = len(train_dl)
            save_interval = max(1, num_batches // 50)  # Save loss 50 times per epoch if possible
            for batch_idx, train_set in enumerate(tqdm(train_dl, desc=f"Epoch {epoch_ndx} Training")):

                # Extract the input and ground truth (they are already on GPU)
                train_inputs = train_set[0].float()
                train_truths = train_set[1].float()

                # Zero out the gradients, do a forward pass, compute the loss, and backpropagate
                self.optimizer.zero_grad()
                train_outputs = self.model(train_inputs)

                if self.is_bayesian:
                    # --- BBB Loss Calculation ---
                    # The loss is: NLL + beta * KL
                    # where NLL = -log p(D|theta) and KL = log q(theta|D) - log p(theta)
                    
                    # 1. Negative Log-Likelihood (NLL)
                    # This is the reconstruction loss, which measures how well the output matches the truth.
                    # We use SmoothL1Loss with reduction='sum' to get the total loss for the batch.
                    reconstruction_loss = self.criterion(train_outputs, train_truths)

                    # 2. KL Divergence Term
                    # This is calculated in the model's forward pass via sampling.
                    # We scale it by the number of batches to balance it with the NLL term.
                    # The `beta_BBB` config parameter provides an additional weighting factor.
                    kl_term = self.model.kl_divergence()
                    kl_loss = kl_term / num_batches
                    
                    train_loss = reconstruction_loss + self.config['beta_BBB'] * kl_loss

                    if batch_idx % 50 == 0:
                        logger.debug(f"Epoch {epoch_ndx}, Batch {batch_idx}: Total Loss: {train_loss.item():.2f}, SmoothL1Loss: {reconstruction_loss.item():.2f}, KL Term: {kl_loss.item():.2f}")
                        # Print out mean reduction loss
                        num_pixels = train_outputs.numel()
                        mean_recon_loss = reconstruction_loss.item() / num_pixels
                        logger.debug(f"Epoch {epoch_ndx}, Batch {batch_idx}: Mean Reconstruction Loss: {mean_recon_loss:.6f}")
                elif self.is_evidential:
                    # --- Evidential Loss Calculation ---
                    # The loss is NLL + beta1 * reg + beta2 * SmoothL1Loss
                    gamma, nu, alpha, beta = train_outputs
                    nll = nig_nll(gamma, nu, alpha, beta, train_truths).mean()
                    reg = nig_reg(gamma, nu, alpha, beta, train_truths).mean()
                    u_reg = nig_u_reg(gamma, nu, alpha, beta, train_truths).mean()
                    evidential = self.config['beta_evidential_nll'] * nll + self.config['beta_evidential_reg'] * reg + self.config['beta_evidential_u_reg'] * u_reg
                    smooth_l1 = self.criterion(gamma, train_truths)
                    train_loss = evidential + self.config['beta_evidential_smooth_l1'] * smooth_l1
                    avg_evidence = torch.mean(2 * nu + alpha).item()

                    if batch_idx % 50 == 0:
                        logger.debug(f"Epoch {epoch_ndx}, Batch {batch_idx}: NLL: {nll.item():.4f}, Reg: {reg.item():.4f}, UReg: {u_reg.item():.4f}, SmoothL1Loss: {smooth_l1.item():.4f}, Avg Evidence: {avg_evidence:.4f}")
                else:
                    # Standard loss for a deterministic model
                    train_loss = self.criterion(train_outputs, train_truths)

                train_loss.backward()

                # Clip gradients if needed
                if self.config["grad_clip"]:
                    torch.nn.utils.clip_grad_value_(
                        self.model.parameters(), clip_value=self.config["grad_max"]
                    )

                # Parameter update
                self.optimizer.step()
                epoch_total_train_loss += train_loss.item()
                # Save intermediate training loss to the list
                if batch_idx % save_interval == 0 or batch_idx == num_batches - 1:
                    intermediate_loss = epoch_total_train_loss / (batch_idx + 1)
                    avg_train_loss_values.append(intermediate_loss)
            # Save avg training statistics
            epoch_avg_train_loss = epoch_total_train_loss / num_batches
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
            if self.config["train_at_inference"]:
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
                num_val_batches = len(val_dl)
                val_save_interval = max(1, num_val_batches // 50)  # Save loss 50 times per epoch if possible
                for val_batch_idx, val_set in enumerate(tqdm(val_dl, desc=f"Epoch {epoch_ndx} Validation")):

                    # Extract the input and ground truth, and send to GPU
                    val_inputs = val_set[0].to(device)
                    val_truths = val_set[1].to(device)

                    # Do a forward pass and calculate the loss
                    val_outputs = self.model(val_inputs)

                    if self.is_bayesian:
                        reconstruction_loss = self.criterion(val_outputs, val_truths)
                        kl_term = self.model.kl_divergence()
                        kl_loss = kl_term / num_val_batches # Scale by number of validation batches
                        val_loss = reconstruction_loss + self.config['beta_BBB'] * kl_loss
                    elif self.is_evidential:
                        gamma, nu, alpha, beta = val_outputs
                        nll = nig_nll(gamma, nu, alpha, beta, val_truths).mean()
                        reg = nig_reg(gamma, nu, alpha, beta, val_truths).mean()
                        u_reg = nig_u_reg(gamma, nu, alpha, beta, val_truths).mean()
                        evidential = nll + self.config['beta_evidential_reg'] * reg + self.config['beta_evidential_u_reg'] * u_reg
                        smooth_l1 = self.criterion(gamma, val_truths)
                        val_loss = evidential + self.config['beta_evidential_smooth_l1'] * smooth_l1

                        if val_batch_idx % 50 == 0:
                            logger.debug(f"Epoch {epoch_ndx}, Batch {val_batch_idx}: NLL: {nll.item():.4f}, Reg: {reg.item():.4f}, UReg: {u_reg.item():.4f}, SmoothL1Loss: {smooth_l1.item():.4f}")
                    else:
                        val_loss = self.criterion(val_outputs, val_truths)
                    
                    epoch_total_val_loss += val_loss.item()

                    # Save intermediate validation loss to the list
                    if val_batch_idx % val_save_interval == 0 or val_batch_idx == num_val_batches - 1:
                        intermediate_val_loss = epoch_total_val_loss / (val_batch_idx + 1)
                        avg_val_loss_values.append(intermediate_val_loss)
                epoch_avg_val_loss = epoch_total_val_loss / num_val_batches
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

            if self.scans_agg_train is not None:
                # Delete the dataloaders to free up memory
                del train_dl.dataset.tensor_1, val_dl.dataset.tensor_1, train_dl.dataset.tensor_2, val_dl.dataset.tensor_2
                del train_dl.dataset, val_dl.dataset
                del train_dl, val_dl
                gc.collect()

            # --- Save model checkpoint ---
            if (
                epoch_ndx % self.config["checkpoint_save_step"] == 0
                or epoch_ndx == self.config["epochs"]
            ):
                save_path = self.files.get_model_filepath(self.config["model_version"], self.config["domain"], checkpoint=epoch_ndx)
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

            # Update the learning rate
            self.scheduler.step()

        logger.info(
            "Training finished, took {:.2f}s\n".format(
                time.time() - training_start_time
            )
        )

        # Training is done
        logger.info("Saving training results...")

        # Standard model saving
        model_path = self.files.get_model_filepath(self.config["model_version"], self.config["domain"])
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

        train_loss_path = self.files.get_train_loss_filepath(self.config["model_version"], self.config["domain"])
        validation_loss_path = self.files.get_validation_loss_filepath(self.config["model_version"], self.config["domain"])

        torch.save(avg_train_loss_values, train_loss_path)
        torch.save(avg_val_loss_values, validation_loss_path)
        logger.info(f"Training and validation loss saved\n")

        # Clean up tensor board writers
        if self.config["tensor_board"]:
            self.trn_writer.flush()
            self.trn_writer.close()
            self.val_writer.flush()
            self.val_writer.close()

        # Clean up memory
        logger.debug("Cleaning up memory...")
        gc.collect()
        
        # Try to delete on-the-fly aggregated reconstructions
        if self.scans_agg_train is not None:
            try:
                os.remove(self.files.get_images_aggregate_filepath(self.config["input_type"], "TRAIN", gated=False))
                os.remove(self.files.get_images_aggregate_filepath(self.config["input_type"], "VALIDATION", gated=False))
                logger.info("Aggregated reconstructions deleted successfully.")
            except Exception as e:
                logger.error(f"Error deleting aggregated reconstructions: {e}")

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
            if self.scans_agg_train is None:
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
