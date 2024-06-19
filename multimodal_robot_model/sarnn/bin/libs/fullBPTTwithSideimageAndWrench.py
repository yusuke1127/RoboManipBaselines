#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
import torch.nn as nn
from eipl.utils import LossScheduler
from enum import IntEnum


class Loss(IntEnum):
    FRONT_IMG = 0
    SIDE_IMG = 1
    JOINT = 2
    WRENCH = 3
    FRONT_PT = 4
    SIDE_PT = 5


class fullBPTTtrainerWithSideimageAndWrench:
    """
    Helper class to train recurrent neural networks with numpy sequences

    Args:
        traindata (np.array): list of np.array. First diemension should be time steps
        model (torch.nn.Module): rnn model
        optimizer (torch.optim): optimizer
        input_param (float): input parameter of sequential generation. 1.0 means open mode.
    """

    def __init__(self, model, optimizer, loss_weights=[1.0 for _ in Loss], device="cpu"):
        self.device = device
        self.optimizer = optimizer
        self.loss_weights = loss_weights
        self.scheduler = LossScheduler(decay_end=1000, curve_name="s")
        self.model = model.to(self.device)

    def save(self, epoch, loss, savename):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                #'optimizer_state_dict': self.optimizer.state_dict(),
                "train_loss": loss[0],
                "test_loss": loss[1],
            },
            savename,
        )

    def process_epoch(self, data, training=True):
        if not training:
            self.model.eval()
        else:
            self.model.train()

        total_loss = 0.0
        for n_batch, ((x_front_img, x_side_img, x_joint, x_wrench), (y_front_img, y_side_img, y_joint, y_wrench)) in enumerate(data):
            if "cpu" in self.device:
                x_front_img = x_front_img.to(self.device)
                y_front_img = y_front_img.to(self.device)
                x_side_img = x_side_img.to(self.device)
                y_side_img = y_side_img.to(self.device)
                x_joint = x_joint.to(self.device)
                y_joint = y_joint.to(self.device)
                x_wrench = x_wrench.to(self.device)
                y_wrench = y_wrench.to(self.device)

            state = None
            yif_list, yis_list, yvj_list, yvw_list = [], [], [], []
            dec_f_pts_list, dec_s_pts_list, enc_f_pts_list, enc_s_pts_list = [], [], [], []
            self.optimizer.zero_grad(set_to_none=True)
            for t in range(x_front_img.shape[1] - 1):
                _yif_hat, _yis_hat, _yvj_hat, _yvw_hat, enc_f_ij, enc_s_ij, dec_f_ij, dec_s_ij, state = self.model(
                    x_front_img[:, t], x_side_img[:, t], x_joint[:, t], x_wrench[:, t], state
                )
                yif_list.append(_yif_hat)
                yis_list.append(_yis_hat)
                yvj_list.append(_yvj_hat)
                yvw_list.append(_yvw_hat)
                enc_f_pts_list.append(enc_f_ij)
                enc_s_pts_list.append(enc_s_ij)
                dec_f_pts_list.append(dec_f_ij)
                dec_s_pts_list.append(dec_s_ij)

            yif_hat = torch.permute(torch.stack(yif_list), (1, 0, 2, 3, 4))
            yis_hat = torch.permute(torch.stack(yis_list), (1, 0, 2, 3, 4))
            yvj_hat = torch.permute(torch.stack(yvj_list), (1, 0, 2))
            yvw_hat = torch.permute(torch.stack(yvw_list), (1, 0, 2))

            front_img_loss = nn.MSELoss()(yif_hat, y_front_img[:, 1:]) * self.loss_weights[Loss.FRONT_IMG]
            side_img_loss = nn.MSELoss()(yis_hat, y_side_img[:, 1:]) * self.loss_weights[Loss.SIDE_IMG]
            joint_loss = nn.MSELoss()(yvj_hat, y_joint[:, 1:]) * self.loss_weights[Loss.JOINT]
            wrench_loss = nn.MSELoss()(yvw_hat, y_wrench[:, 1:]) * self.loss_weights[Loss.WRENCH]
            # Gradually change the loss value using the LossScheluder class.
            front_pt_loss = nn.MSELoss()(
                torch.stack(dec_f_pts_list[:-1]), torch.stack(enc_f_pts_list[1:])
            ) * self.scheduler(self.loss_weights[Loss.FRONT_PT])
            side_pt_loss = nn.MSELoss()(
                torch.stack(dec_s_pts_list[:-1]), torch.stack(enc_s_pts_list[1:])
            ) * self.scheduler(self.loss_weights[Loss.SIDE_PT])
            loss = front_img_loss + side_img_loss + joint_loss + wrench_loss + front_pt_loss + side_pt_loss
            total_loss += loss.item()

            if training:
                loss.backward()
                self.optimizer.step()

        return total_loss / (n_batch + 1)
