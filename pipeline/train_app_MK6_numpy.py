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
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from .config import AGG_DIR

# Set up logging
log = logging.getLogger(__name__)

use_cuda = torch.cuda.is_available()
if not use_cuda:
    raise RuntimeError(
        "CUDA is not available. Please check your PyTorch installation or GPU setup."
    )
device = torch.device("cuda:0" if use_cuda else "cpu")
log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))


def parse_sys_argv(sys_argv=None):
    """Parse command line arguments for the training application."""
    if sys_argv is None:
        sys_argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epoch",
        help="Number of epochs to train for",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--network",
        help="Network for training",
        default="FBPCONVNet",
        type=str,
    )
    parser.add_argument("--model_name", type=str, default="test")
    parser.add_argument(
        "--data_ver",
        help="Dataset version",
        type=str,
    )
    parser.add_argument("--optimizer", default="SGD", type=str)
    parser.add_argument("--shuffle", default=True, type=bool)
    parser.add_argument("--DEBUG", default=False, type=bool)
    parser.add_argument(
        "--batch_size",
        help="Batch size to use for training",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--num_workers",
        help="Number of worker processes for background data loading",
        default=0,
        type=int,
    )

    # --------------------------

    parser.add_argument(
        "--data_path", default="D:/MitchellYu/NSG_CBCT/phase4/data/", type=str
    )

    parser.add_argument(
        "--domain",
        help="Domain of the data, either 'PROJ' or 'IMAG'",
        default="IMAG",
        type=str,
    )
    parser.add_argument(
        "--scan_type",
        help="Type of scan, either 'HF' or 'FF'",
        default="HF",
        type=str,
    )

    parser.add_argument("--input_type", default="FDK", type=str)
    parser.add_argument("--pl_ver", default=1, type=int)

    parser.add_argument("--augment", default=False, type=bool)
    parser.add_argument(
        "--learning_rate_Adam",
        help="Learning rate for Adam",
        default=1e-3,
        type=float,
    )
    parser.add_argument(
        "--learning_rate_NAdam",
        help="Learning rate for NAdam",
        default=1e-3,
        type=float,
    )
    parser.add_argument(
        "--betas_NAdam",
        help="Betas for NAdam",
        default=(0.9, 0.999),
        type=tuple,
    )
    parser.add_argument(
        "--momentum_decay_NAdam",
        help="Momentum decay for NAdam",
        default=4e-4,
        type=float,
    )
    parser.add_argument("--grad_clip", default=True, type=bool)
    parser.add_argument(
        "--grad_max",
        help="",
        default=0.01,
        type=float,
    )

    # SGD optimizer parameters
    parser.add_argument(
        "--learning_rate_SGD",
        help="Learning rate for SGD",
        default=np.logspace(-2, -3, 20),
        type=tuple,
    )
    parser.add_argument(
        "--momentum_SGD",
        help="",
        default=0.99,
        type=float,
    )
    parser.add_argument(
        "--weight_decay_SGD",
        help="",
        default=1e-8,
        type=float,
    )

    parser.add_argument("--model_dir", type=str, default="./model/")

    parser.add_argument("--checkpoint_save_step", help="", default=10, type=int)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")

    parser.add_argument("--tensor_board", default=False, type=bool)

    parser.add_argument(
        "comment",
        help="Comment suffix for Tensorboard run.",
        nargs="?",
        default="dwlpt",
    )

    return parser.parse_args(sys_argv)


def init_model(cli_args):
    """Initialize the CNN model and move it to the GPU."""
    model = getattr(network_instance, cli_args.network)()
    log.info(f"Network Selected: {cli_args.network}")

    model = model.to(device)

    return model


def init_loss(cli_args):
    loss = nn.SmoothL1Loss()
    log.info("Loss function: SmoothL1Loss")
    return loss


def init_optimizer(cli_args, model):
    if cli_args.optimizer == "SGD":
        log.info(
            f"Optimizer: SGD with learning rate {cli_args.learning_rate_SGD}, momentum {cli_args.momentum_SGD}, weight decay {cli_args.weight_decay_SGD}"
        )
        return SGD(
            model.parameters(),
            lr=cli_args.learning_rate_SGD,
            momentum=cli_args.momentum_SGD,
            weight_decay=cli_args.weight_decay_SGD,
        )
    elif cli_args.optimizer == "Adam":
        log.info(f"Optimizer: Adam with learning rate {cli_args.learning_rate_Adam}")
        return Adam(model.parameters(), lr=cli_args.learning_rate_Adam)
    elif cli_args.optimizer == "NAdam":
        log.info(
            f"Optimizer: NAdam with learning rate {cli_args.learning_rate_NAdam}, betas {cli_args.betas_NAdam}, momentum_decay {cli_args.momentum_decay_NAdam}"
        )
        return NAdam(
            model.parameters(),
            lr=cli_args.learning_rate_NAdam,
            betas=cli_args.betas_NAdam,
            momentum_decay=cli_args.momentum_decay_NAdam,
        )

    raise NotImplementedError(
        f"Optimizer {cli_args.optimizer} is not implemented. Supported optimizers are: SGD, Adam, NAdam."
    )


def init_tensorboard_writers(cli_args, time_str):
    """Initialize TensorBoard writers for training and validation."""
    log_dir = os.path.join("runs", cli_args.model_name, time_str)

    trn_writer = SummaryWriter(log_dir=log_dir + "-trn_cls-" + cli_args.comment)
    val_writer = SummaryWriter(log_dir=log_dir + "-val_cls-" + cli_args.comment)

    return trn_writer, val_writer


def get_data_sub_path(cli_args, sample):
    """Get the sub-path for the data based on the data type."""
    input_type = cli_args.input_type
    augment = cli_args.augment
    domain = cli_args.domain
    scan_type = cli_args.scan_type
    pl_ver = cli_args.pl_ver

    # We need to know the input type and augmentation setting to get the right data
    if (input_type, augment) == ("FDK", True):
        sub_path = f"{domain}_ng_{scan_type}_{sample}_aug.npy"
    elif (input_type, augment) == ("FDK", False):
        sub_path = f"{domain}_ng_{scan_type}_{sample}.npy"
    elif (input_type, augment) == ("PL", True):
        sub_path = f"{domain}_ng_{scan_type}_{sample}_pl{pl_ver}_aug.npy"
    elif (input_type, augment) == ("PL", False):
        sub_path = f"{domain}_ng_{scan_type}_{sample}_pl{pl_ver}.npy"
    else:
        raise ValueError(
            "Invalid input_type or augment. Please enter either 'FDK' or 'PL' for input_type and True or False for augment."
        )

    log.info(
        f"Data type: {domain.capitalize()} domain {sample.capitalize()} data for {scan_type} {input_type} {'with' if augment else 'without'} augmentation"
    )

    return sub_path


def init_dataloader(cli_args, sample):
    """Initialize the DataLoader for a specific sample ('TRAIN', 'VALIDATION', or 'TEST')."""
    # Get the sub-path to the training data within the aggregation directory
    sub_path = get_data_sub_path(cli_args, sample)

    images_path = os.path.join(AGG_DIR, sub_path)
    log.info(f"{sample} images path: {images_path}")

    # Replace "ng" with "gated" to get the ground truth path
    truth_images_path = images_path.replace("ng", "gated")
    log.info(f"{sample} ground truth images path: {truth_images_path}")

    # Load the dataset
    dataset = PairNumpySet(images_path, truth_images_path)
    log.info(
        f"{sample} dataset loaded with {len(dataset)} samples, each with shape {dataset[0][0].shape}."
    )

    n_batches = cli_args.batch_size
    n_workers = cli_args.num_workers
    bool_shuffle = cli_args.shuffle

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=n_batches,
        num_workers=n_workers,
        pin_memory=bool_shuffle,
        shuffle=bool_shuffle,
    )
    log.info(
        f"{sample} dataloader initialized with {len(dataloader)} batches of size {n_batches}, with {n_workers} workers, shuffle={bool_shuffle}, and pin_memory={bool_shuffle}."
    )

    return dataloader


class TrainingApp:
    def __init__(self, sys_argv=None):
        # Parse command line arguments
        self.cli_args = parse_sys_argv(sys_argv)

        # Set logging level
        if self.cli_args.DEBUG:
            log.setLevel(logging.INFO)
        else:
            log.setLevel(logging.WARNING)

        # Current time string for tensorboard
        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

        # Initialize model, loss, and optimizer
        self.model = init_model(self.cli_args)
        self.criterion = init_loss(self.cli_args)
        self.optimizer = init_optimizer(self.cli_args, self.model)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = init_dataloader(self.cli_args, "TRAIN")
        log.info(f"Initialized training dataloader with {len(train_dl)} batches.")
        val_dl = init_dataloader(self.cli_args, "VALIDATION")
        log.info(f"Initialized validation dataloader with {len(val_dl)} batches.")

        if self.cli_args.tensor_board:
            self.trn_writer, self.val_writer = init_tensorboard_writers(
                self.cli_args, self.time_str
            )

        # trainning settings
        n_epoch = self.cli_args.epoch
        batch_size = self.cli_args.batch_size
        log.info("Training setting:")
        log.info(f"Training name: {self.cli_args.model_name}")
        log.info(f"Number of epoch: {n_epoch}")
        log.info(f"Batch Size: {batch_size}")
        log.info(f"Input type: {self.cli_args.input_type}")
        log.info(f"Dataset Version: DS{self.cli_args.data_ver}")
        log.info(f"Data Shuffle: {self.cli_args.shuffle}")
        log.info(f"Data Augmentation: {self.cli_args.augment}")
        log.info(f"Optimizer: {self.cli_args.optimizer}")
        log.info(f"Momentum: {self.cli_args.momentum}")
        log.info(f"Gradient Clip: {self.cli_args.grad_clip}")
        if self.cli_args.grad_clip:
            log.info(f"Clip Max: {self.cli_args.grad_max}")
        log.info(f"Tensor Board: {self.cli_args.tensor_board}")

        avg_train_loss_values = []
        avg_val_loss_values = []

        log.info("Start training...")
        dur = []
        training_start_time = time.time()

        lr_range = self.cli_args.learning_rate

        for epoch_ndx in range(1, n_epoch + 1):

            ###################
            # train the model #
            ###################
            self.model.train()

            learning_rate = lr_range[
                min(epoch_ndx - 1, len(self.cli_args.learning_rate) - 1)
            ]
            if self.cli_args.DEBUG:
                log.info(f"Epoch: {epoch_ndx}, Learning Rate: {learning_rate}")
            self.optimizer = self.initOptimizer(learning_rate)

            # monitor training loss
            running_train_loss = 0.0
            running_train_psnr = 0.0
            running_train_ssim = 0.0

            # time training time for each epoch
            t_train = time.time()

            for train_set in train_dl:

                train_batch = train_set[0]
                train_truth_batch = train_set[1]

                train_batch = train_batch.to(self.device)
                train_truth_batch = train_truth_batch.to(self.device)

                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                train_outputs = self.model(train_batch)
                # calculate the loss
                train_loss = self.criterion(train_outputs, train_truth_batch)
                # backward pass: compute gradient of the loss with respect to model parameters
                train_loss.backward()
                # clip gradient
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(), clip_value=self.cli_args.grad_max
                )
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                # update running training loss
                running_train_loss += train_loss.item() * train_batch.size(0)

                # log.info(f'output: + {outputs.get_device()}')
                # log.info(f'truth:  + {train_truth_batch.get_device()}')
                # if self.cli_args.tensor_board:
                #     # psnr
                #     train_psnr_batch = psnr(
                #         train_outputs.detach().clone().cpu(), train_truth_batch.detach().clone().cpu())
                #     running_train_psnr += train_psnr_batch.item()
                #     # ssim
                #     train_ssim_batch = ssim(
                #         train_outputs.detach().clone().cpu(), train_truth_batch.detach().clone().cpu())
                #     running_train_ssim += train_ssim_batch.item()
                #     if self.cli_args.DEBUG:
                #         log.info(f'Training Loss: {running_train_loss}')
                #         log.info(f'Training PSNR: {running_train_psnr}')
                #         log.info(f'Training SSIM: {running_train_ssim}')

            # print avg training statistics
            avg_train_loss = running_train_loss / len(train_dl)
            avg_train_loss_values.append(avg_train_loss)

            # store loss (SmoothL1) and SSIM in TensorBoard
            if self.cli_args.tensor_board:
                self.trn_writer.add_scalar("Loss", avg_train_loss, epoch_ndx)
                # avg_train_psnr = running_train_psnr/len(train_dl)
                # self.trn_writer.add_scalar("PSNR", avg_train_psnr, epoch_ndx)
                # avg_train_ssim = running_train_ssim/len(train_dl)
                # self.trn_writer.add_scalar("SSIM", avg_train_ssim, epoch_ndx)

            # dur.append(time.time() - t_train)
            dur = time.time() - t_train

            log.info(
                "Epoch: {} \tTraining Loss: {:.6f}  \tTime(s) {:.4f}".format(
                    epoch_ndx,
                    avg_train_loss,
                    # np.mean(dur)
                    dur,
                )
            )

            ###################
            # validation the model #
            ###################

            self.model.eval()

            with torch.no_grad():

                # monitor validation loss
                running_val_loss = 0.0
                running_val_psnr = 0.0
                running_val_ssim = 0.0

                # time validation time for each epoch
                t_val = time.time()

                for val_set in val_dl:

                    val_batch = val_set[0]
                    val_truth_batch = val_set[1]

                    val_batch = val_batch.to(self.device)
                    val_truth_batch = val_truth_batch.to(self.device)

                    # forward pass: compute predicted outputs by passing inputs to the model
                    val_outputs = self.model(val_batch)
                    # calculate the loss
                    val_loss = self.criterion(val_outputs, val_truth_batch)
                    # update running validation loss
                    running_val_loss += val_loss.item() * val_batch.size(0)

                    # if self.cli_args.tensor_board:
                    #     # psnr
                    #     val_psnr_batch = psnr(
                    #         val_outputs.detach().clone().cpu(), val_truth_batch.detach().clone().cpu())
                    #     running_val_psnr += val_psnr_batch.item()
                    #     # ssim
                    #     val_ssim_batch = ssim(
                    #         val_outputs.detach().clone().cpu(), val_truth_batch.detach().clone().cpu())
                    #     running_val_ssim += val_ssim_batch.item()  # *val_batch.size(0)
                    #     if self.cli_args.DEBUG:
                    #         log.info(f'Validation Loss: {running_val_loss}')
                    #         log.info(f'Validation PSNR: {running_val_psnr}')
                    #         log.info(f'Validation SSIM: {running_val_ssim}')

                # print avg validation statistics
                avg_val_loss = running_val_loss / len(val_dl)
                avg_val_loss_values.append(avg_val_loss)

                # store loss (SmoothL1) and SSIM in TensorBoard
                if self.cli_args.tensor_board:
                    self.val_writer.add_scalar("Loss", avg_val_loss, epoch_ndx)
                    # avg_val_psnr = running_val_psnr/len(val_dl)
                    # self.val_writer.add_scalar("PSNR", avg_val_psnr, epoch_ndx)
                    # avg_val_ssim = running_val_ssim/len(val_dl)
                    # self.val_writer.add_scalar("SSIM", avg_val_ssim, epoch_ndx)

                # dur.append(time.time() - t_val)
                dur = time.time() - t_val

                log.info(
                    "Epoch: {} \tValidation Loss: {:.6f}  \tTime(s) {:.4f}".format(
                        epoch_ndx,
                        avg_val_loss,
                        # np.mean(dur)
                        dur,
                    )
                )

            # save check_point
            if (epoch_ndx + 1) % self.cli_args.checkpoint_save_step == 0 or (
                epoch_ndx + 1
            ) == self.cli_args.epoch:
                if not os.path.exists(self.cli_args.checkpoint_dir):
                    os.mkdir(self.cli_args.checkpoint_dir)
                check_point_path = os.path.join(
                    self.cli_args.checkpoint_dir, "epoch-%d.pkl" % (epoch_ndx + 1)
                )
                torch.save(
                    {
                        "epoch": epoch_ndx + 1,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    },
                    check_point_path,
                )
                print("save checkpoint %s", check_point_path)

        log.info(
            "Training finished, took {:.2f}s".format(time.time() - training_start_time)
        )

        log.info("Saving training results...")
        torch.save(
            self.model.state_dict(),
            self.cli_args.model_dir + self.cli_args.model_name + ".pth",
        )
        log.info(f"Model saved as: {self.cli_args.model_name}")
        torch.save(
            avg_train_loss_values,
            self.cli_args.model_dir
            + "loss/"
            + self.cli_args.model_name
            + "_train_loss.pth",
        )
        torch.save(
            avg_val_loss_values,
            self.cli_args.model_dir
            + "loss/"
            + self.cli_args.model_name
            + "_validation_loss.pth",
        )

        if self.cli_args.tensor_board:
            self.trn_writer.flush()
            self.trn_writer.close()
            self.val_writer.flush()
            self.val_writer.close()

        gc.collect()
        self.model = None
        del self.model
        del train_dl, train_batch, train_truth_batch, val_dl, val_batch, val_truth_batch
        del train_outputs, val_outputs
        del train_loss, running_train_loss, avg_train_loss
        del val_loss, running_val_loss, avg_val_loss
        del self.trn_writer, self.val_writer
        # del train_psnr_batch, running_train_psnr, avg_train_psnr
        # del val_psnr_batch, running_val_psnr, avg_val_psnr
        # del train_ssim_batch, running_train_ssim, avg_train_ssim
        # del val_ssim_batch, running_val_ssim, avg_val_ssim
        with torch.no_grad():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    TrainingApp().main()
