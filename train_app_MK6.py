import argparse
import datetime
import time
import math
import os
import sys
import gc

import numpy as np

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, NAdam
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from dsets import PairSet
from util.logconf import logging

from network_instance import IResNetEvidential

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

psnr = PeakSignalNoiseRatio()
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # specify which GPU(s) to be used

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3


class TrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--num_workers",
            help="Number of worker processes for background data loading",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--batch_size",
            help="Batch size to use for training",
            default=8,
            type=int,
        )
        parser.add_argument(
            "--epoch",
            help="Number of epochs to train for",
            default=1,
            type=int,
        )

        parser.add_argument(
            "--work_path",
            help="",
            default="D:/MitchellYu/NSG_CBCT/phase4/",
            type=str,
        )
        parser.add_argument(
            "--data_path", default="D:/MitchellYu/NSG_CBCT/phase4/data/", type=str
        )

        parser.add_argument(
            "--data_ver",
            help="Dataset version",
            default=1,
            type=str,
        )
        parser.add_argument("--input_type", default="FDK", type=str)
        parser.add_argument("--pl_ver", default=1, type=int)

        parser.add_argument("--reload_data", default=False, type=bool)
        parser.add_argument("--shuffle", default=True, type=bool)
        parser.add_argument("--augment", default=False, type=bool)
        parser.add_argument("--optimizer", default="SGD", type=str)
        parser.add_argument(
            "--learning_rate",
            help="Learning rate for SGD",
            default=np.logspace(-2, -3, 20),
            type=tuple,
        )
        parser.add_argument("--grad_clip", default=True, type=bool)
        parser.add_argument(
            "--grad_max",
            help="",
            default=0.01,
            type=float,
        )
        parser.add_argument(
            "--momentum",
            help="",
            default=0.99,
            type=float,
        )
        parser.add_argument(
            "--lambda_value",
            help="Regularization constant value for the evidential loss (e.g. 1e-4)",
            type=float,
            default=1e-3,
        )

        parser.add_argument("--model_dir", type=str, default="./model/")
        parser.add_argument("--model_name", type=str, default="test")

        parser.add_argument("--sample_step", help="", default=100, type=int)
        parser.add_argument("--sample_dir", help="", default="./samples/", type=str)
        parser.add_argument("--checkpoint_save_step", help="", default=10, type=int)
        parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")

        parser.add_argument("--tensor_board", default=False, type=bool)
        parser.add_argument(
            "--tb-prefix",
            default="IResNetEvidential",
            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )

        parser.add_argument(
            "comment",
            help="Comment suffix for Tensorboard run.",
            nargs="?",
            default="dwlpt",
        )

        parser.add_argument("--DEBUG", default=False, type=bool)
        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.criterion = self.initNigLoss(self.cli_args.lambda_value)

    def initModel(self):
        log.info("Loading IResNetEvidential CNN...")

        model = IResNetEvidential()

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            """
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            """
            model = model.to(self.device)
        return model

    def initOptimizer(self, learning_rate):
        if self.cli_args.optimizer == "SGD":
            if self.cli_args.DEBUG:
                log.info("Optimizer: SGD")
            return SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.99,
                weight_decay=1e-8,
            )
        elif self.cli_args.optimizer == "Adam":
            if self.cli_args.DEBUG:
                log.info("Optimizer: Adam")
            return Adam(self.model.parameters(), lr=0.001)
        elif self.cli_args.optimizer == "NAdam":
            if self.cli_args.DEBUG:
                log.info("Optimizer: NAdam")
            return NAdam(
                self.model.parameters(), lr=1e-4, betas=(0.9, 0.99), momentum_decay=4e-4
            )
        else:
            log.info("This optimizer is not supported yet!")
            # return SGD(self.model.parameters(), lr=learning_rate, momentum=0.99, weight_decay=1e-8)

    @staticmethod
    def nig_nll(gamma, nu, alpha, beta, y_true):
        omega = 2.0 * beta * (1.0 + nu)
        term1 = 0.5 * torch.log(torch.pi / nu)
        term2 = -alpha * torch.log(omega)
        term3 = (alpha + 0.5) * torch.log((y_true - gamma) ** 2 * nu + omega)
        term4 = torch.special.gammaln(alpha) - torch.special.gammaln(alpha + 0.5)
        return term1 + term2 + term3 + term4

    @staticmethod
    def nig_reg(gamma, nu, alpha, beta, y_true):
        return torch.abs(y_true - gamma) * (2.0 * nu + alpha)

    def initNigLoss(self, lambda_reg=1e-3, lambda_l1=1.0):
        def nig_loss_smoothL1(gamma, nu, alpha, beta, y_true):
            # evidential NLL + regularizer
            nll = TrainingApp.nig_nll(gamma, nu, alpha, beta, y_true)
            reg = TrainingApp.nig_reg(gamma, nu, alpha, beta, y_true)
            evidential = (nll + lambda_reg * reg).mean()
            # Smooth L1 between prediction (gamma) and truth
            smooth_l1 = nn.SmoothL1Loss()(gamma, y_true)
            # combine both losses
            return evidential + lambda_l1 * smooth_l1

        return nig_loss_smoothL1

    def initTrainDl(self):
        log.info("Loading Training Data Sets From Saved Tensor...")
        if self.cli_args.augment:
            train_images = torch.load(
                self.cli_args.data_path
                + f"DS{self.cli_args.data_ver}/"
                + "train/ns/train_ns_aug.pt"
            )
            train_truth_images = torch.load(
                self.cli_args.data_path
                + f"DS{self.cli_args.data_ver}/"
                + "train/full/train_full_aug.pt"
            )
        else:
            train_images = torch.load(
                self.cli_args.data_path
                + f"DS{self.cli_args.data_ver}/"
                + "train/ns/train_ns.pt"
            )
            train_truth_images = torch.load(
                self.cli_args.data_path
                + f"DS{self.cli_args.data_ver}/"
                + "train/full/train_full.pt"
            )

        if self.cli_args.DEBUG:
            log.info(f"Training Sample Shape: {train_images.shape}")

        train_set = PairSet(train_images, train_truth_images)

        n_batches = self.cli_args.batch_size
        n_workers = self.cli_args.num_workers
        bool_shuffle = self.cli_args.shuffle

        train_dl = torch.utils.data.DataLoader(
            train_set,
            batch_size=n_batches,
            num_workers=n_workers,
            pin_memory=bool_shuffle,
            shuffle=bool_shuffle,
        )

        return train_dl

    def initValDl(self):

        log.info("Loading Validation Data Sets From Saved Tensor...")

        if self.cli_args.augment:
            val_images = torch.load(
                self.cli_args.data_path
                + f"DS{self.cli_args.data_ver}/"
                + "validation/ns/val_ns_aug.pt"
            )
            val_truth_images = torch.load(
                self.cli_args.data_path
                + f"DS{self.cli_args.data_ver}/"
                + "validation/full/val_full_aug.pt"
            )
        else:
            val_images = torch.load(
                self.cli_args.data_path
                + f"DS{self.cli_args.data_ver}/"
                + "validation/ns/val_ns.pt"
            )
            val_truth_images = torch.load(
                self.cli_args.data_path
                + f"DS{self.cli_args.data_ver}/"
                + "validation/full/val_full.pt"
            )

        if self.cli_args.DEBUG:
            log.info(f"Validation Sample Shape: {val_images.shape}")

        val_set = PairSet(val_images, val_truth_images)

        n_batches = self.cli_args.batch_size
        n_workers = self.cli_args.num_workers
        bool_shuffle = self.cli_args.shuffle

        val_dl = torch.utils.data.DataLoader(
            val_set,
            batch_size=n_batches,
            num_workers=n_workers,
            pin_memory=bool_shuffle,
            shuffle=bool_shuffle,
        )

        return val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join("runs", self.cli_args.model_name, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + "-trn_cls-" + self.cli_args.comment
            )
            self.val_writer = SummaryWriter(
                log_dir=log_dir + "-val_cls-" + self.cli_args.comment
            )

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        if self.cli_args.tensor_board:
            self.initTensorboardWriters()

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
        log.info(f"Regularization Constant (Lambda): {self.cli_args.lambda_value}")
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
                gamma, nu, alpha, beta = self.model(train_batch)
                train_loss = self.criterion(gamma, nu, alpha, beta, train_truth_batch)
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
                    gamma, nu, alpha, beta = self.model(val_batch)
                    val_loss = self.criterion(gamma, nu, alpha, beta, val_truth_batch)
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
        # del train_outputs, val_outputs
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
