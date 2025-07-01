import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # specify which GPU(s) to be used

import datetime
import time
import gc
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import SequentialLR
import torch.optim as optim
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, NAdam
from .dsets import PairNumpySet
from . import network_instance
import logging
from tqdm import tqdm
from .utils import ensure_dir
import ast
from pipeline.paths import Files
import math # Added for BBB beta calculation


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


def init_loss(config: dict, is_bayesian: bool):
    if is_bayesian:
        loss = nn.SmoothL1Loss(reduction='sum')
    else:
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


def init_dataloader(config: dict, files: Files, sample: str, input_type: str, domain: str):
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


# TODO we should probably extract everything from the config
#      and just make them attributes of the training app
# TODO this code should also just generally be cleaned up a bit
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
        self.swag_enabled = self.config["swag_enabled"]

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
        if isinstance(self.config["beta_BBB"], str):
            self.config["beta_BBB"] = ast.literal_eval(self.config["beta_BBB"])

        # Set logging level
        if DEBUG:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Current time string for tensorboard
        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

        # Initialize model, loss, and optimizer
        self.model = init_model(self.config)
        self.criterion = init_loss(self.config, self.is_bayesian)
        
        # SWAG Initialization
        if self.swag_enabled:
            logger.info("SWAG training is enabled. Loading base model checkpoint.")
            
            # 1. Load the specified checkpoint
            start_model_version = self.config['swag_start_model_version']
            start_epoch = self.config['swag_start_checkpoint_epoch']
            
            checkpoint_path = self.files.get_model_filepath(
                model_version=start_model_version,
                domain=self.config["domain"],
                checkpoint=start_epoch,
                ensure_exists=False
            )
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint['state_dict'])
            logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} of model '{start_model_version}'.")

            # 2. Set up the SWAG optimizer
            optimizer_choice = self.config['swag_optimizer']
            if optimizer_choice == 'keep':
                logger.info("Keeping optimizer from loaded checkpoint.")
                self.optimizer, burn_in_scheduler = init_optimizer(self.config, self.model)
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif optimizer_choice == 'SGD':
                logger.info("Initializing new SGD optimizer for SWAG.")
                self.optimizer = optim.SGD(self.model.parameters(), lr=self.config['learning_rate'])
                # Create a standard scheduler for the burn-in phase
                _, burn_in_scheduler = init_optimizer(self.config, self.model)
            else:
                raise ValueError(f"Unsupported SWAG optimizer: {optimizer_choice}. Supported optimizers are: 'keep', 'SGD'.")

            # --- NEW SCHEDULER LOGIC ---
            # This scheduler seamlessly handles the burn-in phase followed by the constant LR SWA phase.
            swa_scheduler = SWALR(self.optimizer, swa_lr=self.config['swag_lr'])
            self.scheduler = SequentialLR(self.optimizer, schedulers=[burn_in_scheduler, swa_scheduler], milestones=[self.config['swag_burn_in_epochs']])
            logger.info(f"Using SequentialLR scheduler: {self.config['swag_burn_in_epochs']} burn-in epochs, then SWALR with constant LR.")

            # 3. Initialize the models for SWA and the SWAG-Diagonal second moment
            self.swag_model = AveragedModel(self.model)
            
            # --- NEW: SECOND MOMENT MODEL FOR DIAGONAL COVARIANCE ---
            # This custom average function will compute the average of the squared parameters.
            def avg_fn_sq(p_avg, p, num_averaged):
                return p_avg * num_averaged / (num_averaged + 1.0) + p**2 / (num_averaged + 1.0)
            self.swa_sq_model = AveragedModel(self.model, avg_fn=avg_fn_sq)

            # 4. Initialize storage for the low-rank covariance matrix
            self.swag_cov_mat_list = []

            self.start_epoch = 1
            self.epochs = self.config['swag_burn_in_epochs'] + self.config['swag_epochs']

        else: # Standard (Non-SWAG) Initialization
            self.optimizer, self.scheduler = init_optimizer(self.config, self.model)
            # Resume from checkpoint if provided
            self.start_epoch = 1
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
        # Make sure we're not accidentally overwriting an existing model version
        if self.swag_enabled:
            save_dir = self.files.directories.get_model_dir(self.config["swag_model_version"], self.config["domain"], ensure_exists=False)
        else:
            save_dir = self.files.directories.get_model_dir(self.config["model_version"], self.config["domain"], ensure_exists=False)
        if os.path.exists(save_dir) and self.start_epoch == 1:
            logger.error(
                f"Save directory {save_dir} already exists. Please choose a different model version or delete the existing directory, or manually delete this."
            )
            return

        logger.debug("Starting {}, {}".format(type(self).__name__, self.config))

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

        if self.swag_enabled:
            total_epochs = self.epochs
        else:
            total_epochs = self.config["epochs"]

        # NOTE: epoch_ndx is 1-indexed!!
        for epoch_ndx in range(self.start_epoch, total_epochs + 1):
            
            # SWAG learning rate and phase logic
            is_swa_phase = False
            if self.swag_enabled:
                burn_in_epochs = self.config['swag_burn_in_epochs']
                if epoch_ndx > burn_in_epochs:
                    is_swa_phase = True
                    desc_phase = f"SWA Epoch {epoch_ndx - burn_in_epochs}/{total_epochs - burn_in_epochs}"
                else:
                    desc_phase = f"Burn-in Epoch {epoch_ndx}/{burn_in_epochs}"
            else: # Standard training
                logger.debug(
                    f"Learning rate is {self.scheduler.get_last_lr()[0]} at epoch {epoch_ndx}."
                )
                desc_phase = f"Epoch {epoch_ndx} Training"
            

            # Set the model to training mode
            self.model.train()
            logger.debug("Model set to training mode for training.")

            # Monitor training loss
            epoch_total_train_loss = 0.0

            # Start time of the training phase
            epoch_train_start_time = time.time()

            # Loop throught the batches in the training dataloader
            for batch_idx, train_set in enumerate(tqdm(train_dl, desc=desc_phase)):

                # Extract the input and ground truth (they are already on GPU)
                train_inputs = train_set[0].float()
                train_truths = train_set[1].float()

                # Zero out the gradients, do a forward pass, compute the loss, and backpropagate
                self.optimizer.zero_grad()
                train_outputs = self.model(train_inputs)

                if self.is_bayesian:
                    reconstruction_loss = self.criterion(train_outputs, train_truths)
                    kl_loss = self.model.kl_divergence() / len(train_dl)
                    train_loss = reconstruction_loss + self.config['beta_BBB'] * kl_loss
                    if batch_idx % 500 == 0:
                        logger.debug(f"Epoch {epoch_ndx}, Batch {batch_idx}: Total Loss: {train_loss.item()}, Reconstruction Loss: {reconstruction_loss.item()}, KL Loss: {kl_loss.item()}, Beta: {self.config['beta_BBB']}")
                else:
                    train_loss = self.criterion(train_outputs, train_truths)

                train_loss.backward()

                # Clip gradients if needed
                if self.config["grad_clip"]:
                    torch.nn.utils.clip_grad_value_(
                        self.model.parameters(), clip_value=self.config["grad_max"]
                    )

                # Parameter update
                self.optimizer.step()

                # Update SWAG model parameters if in SWA phase
                if is_swa_phase and batch_idx % self.config['swag_update_freq'] == 0:
                    self.swag_model.update_parameters(self.model)
                    
                    # --- NEW: UPDATE FOR FULL SWAG COVARIANCE ---
                    self.swa_sq_model.update_parameters(self.model) # Update the second moment
                    
                    # Collect deviation vector for the low-rank part of the covariance matrix
                    with torch.no_grad():
                        curr_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
                        swa_params = torch.nn.utils.parameters_to_vector(self.swag_model.parameters())
                        deviation = curr_params - swa_params
                        self.swag_cov_mat_list.append(deviation)
                        if len(self.swag_cov_mat_list) > self.config['swag_cov_mat_rank']:
                            self.swag_cov_mat_list.pop(0)

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
                for val_set in tqdm(val_dl, desc=f"Epoch {epoch_ndx} Validation"):

                    # Extract the input and ground truth, and send to GPU
                    val_inputs = val_set[0].to(device)
                    val_truths = val_set[1].to(device)

                    # Do a forward pass and calculate the loss
                    val_outputs = self.model(val_inputs)

                    if self.is_bayesian:
                        reconstruction_loss = self.criterion(val_outputs, val_truths)
                        kl_loss = self.model.kl_divergence() / len(val_dl)
                        val_loss = reconstruction_loss + self.config['beta_BBB'] * kl_loss
                    else:
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
            
            # Skip checkpoint saving during SWAG training
            if not self.swag_enabled:
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

                    # Delete most recent checkpoint to save space
                    if epoch_ndx > self.config["checkpoint_save_step"]:
                        old_save_path = self.files.get_model_filepath(self.config["model_version"], self.config["domain"], checkpoint=epoch_ndx - self.config["checkpoint_save_step"])
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
        if self.swag_enabled:
            # logger.info("SWAG training complete. Updating batch norm statistics for SWA model.")
            # torch.optim.swa_utils.update_bn(train_dl, self.swag_model, device=device)

            if self.config.get('swag_model_version'):
                save_model_version = self.config['swag_model_version']
            else:
                save_model_version = self.config['model_version']

            save_path = self.files.get_model_filepath(save_model_version, self.config["domain"])
            logger.info(f"Saving final SWAG model (mean, diagonal, and low-rank covariance) to: {save_path}")
            
            # Convert covariance list to a tensor
            swag_cov_mat = torch.stack(self.swag_cov_mat_list)

            # Now saving the mean, the second moment for the diagonal, and the low-rank deviations
            torch.save({
                'state_dict': self.swag_model.state_dict(),
                'state_dict_sq': self.swa_sq_model.state_dict(),
                'cov_mat_list': swag_cov_mat,
            }, save_path)
            logger.info(f"SWAG model and covariance matrix saved to {save_path}")
        else:
            # Save the final model state
            model_path = self.files.get_model_filepath(self.config["model_version"], self.config["domain"])
            torch.save(
                self.model.state_dict(),
                model_path,
            )
            logger.info(f"Model saved to {model_path}")
        
        # Save the training and validation loss values
        train_loss_path = self.files.get_train_loss_filepath(self.config["model_version"], self.config["domain"])
        torch.save(
            avg_train_loss_values,
            train_loss_path,
        )
        validation_loss_path = self.files.get_validation_loss_filepath(self.config["model_version"], self.config["domain"])
        torch.save(
            avg_val_loss_values,
            validation_loss_path,
        )
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
