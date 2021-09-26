import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class QC_Norm(nn.Module):
    def __init__(self, num_features, init_ang_inc=10, momentum=0.1,training = False):
        super(QC_Norm, self).__init__()

        self.x_running_rot = Parameter(torch.zeros(num_features), requires_grad=training)
        self.ang_inc = Parameter(torch.ones(1) * init_ang_inc, requires_grad=training)

        self.momentum = momentum

        self.printed = False
        self.x_mean_ancle = 0
        self.x_mean_rote = 0
        self.input = 0
        self.output = 0



    def forward(self, x, training=True):
        if not training:
            # if not self.printed:
            #     print("self.ang_inc", self.ang_inc)
            #     print("self.x_running_rot", self.x_running_rot)
            #     self.printed = True

            x = x.transpose(0, 1)

            x_ancle = (x * 2 - 1).acos()
            x_final = x_ancle + self.x_running_rot.unsqueeze(-1)
            x_1 = (x_final.cos() + 1) / 2

            x_1 = x_1.transpose(0, 1)

        else:
            self.printed = False
            x = x.transpose(0, 1)
            x_sum = x.sum(-1).unsqueeze(-1).expand(x.shape)
            x_lack_sum = x_sum - x
            x_mean = x_lack_sum / x.shape[-1]

            x_mean_ancle = (x_mean * 2 - 1).acos()

            ang_inc = self.ang_inc.unsqueeze(-1).expand(x_mean_ancle.shape)
            # ang_inc = np.pi/2/(x.max(-1)[0].unsqueeze(-1).expand(x_mean_ancle.shape) -x.min(-1)[0].unsqueeze(-1).expand(x_mean_ancle.shape) )

            # if self.given_ang != -1:
            #     x_mean_rote = (np.pi / 2 - x_mean_ancle) * self.given_ang
            # else:
            x_mean_rote = (np.pi / 2 - x_mean_ancle) * ang_inc

            x_moving_rot = (x_mean_rote.sum(-1) / x.shape[-1])

            # print(self.x_running_rot[:])

            self.x_running_rot[:] = self.momentum * self.x_running_rot + \
                                    (1 - self.momentum) * x_moving_rot

            # print(self.x_running_rot[:])

            x_ancle = (x * 2 - 1).acos()
            x_final = x_ancle + x_mean_rote
            x_1 = (x_final.cos() + 1) / 2
            x_1 = x_1.transpose(0, 1)

        return x_1

    def reset_parameters(self):
        self.reset_running_stats()
        self.ang_inc.data.zeros_()


def print_degree(x, name="x"):
    print(name, x / np.pi * 180)


class QC_Norm_Real(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super(QC_Norm_Real, self).__init__()
        self.x_running_rot = Parameter(torch.zeros(num_features), requires_grad=False)

        self.momentum = momentum

        self.x_max = 0
        self.x_min = 0
        # print("Using Normal without real")

    def forward(self, x, training=True):
        if not training:
            x = x.transpose(0, 1)

            x_ancle = (x * 2 - 1).acos()
            # x_final = x_ancle+self.x_running_rot.unsqueeze(-1)
            x_final = ((x_ancle - self.x_min) / (self.x_max - self.x_min)) * np.pi

            x_1 = (x_final.cos() + 1) / 2
            x_1 = x_1.transpose(0, 1)

        else:

            x = x.transpose(0, 1)
            x_ancle = (x * 2 - 1).acos()
            x_rectify_ancle = (x_ancle.max(-1)[0] - x_ancle.min(-1)[0]).unsqueeze(-1).expand(x.shape)
            x_final = ((x_ancle - x_ancle.min(-1)[0].unsqueeze(-1)) / (x_rectify_ancle)) * np.pi

            x_moving_rot = x_final - x_ancle

            x_moving_rot_mean = x_moving_rot.sum(-1) / x.shape[-1]
            self.x_running_rot[:] = self.momentum * self.x_running_rot + \
                                    (1 - self.momentum) * x_moving_rot_mean

            self.x_max = self.momentum * x_ancle.max(-1)[0].unsqueeze(-1) + \
                         (1 - self.momentum) * self.x_max
            self.x_min = self.momentum * x_ancle.min(-1)[0].unsqueeze(-1) + \
                         (1 - self.momentum) * self.x_min

            x_1 = (x_final.cos() + 1) / 2
            x_1 = x_1.transpose(0, 1)

        return x_1


class QC_Norm_Real_Correction(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super(QC_Norm_Real_Correction, self).__init__()
        self.x_running_rot = Parameter(torch.zeros(num_features), requires_grad=False)
        self.momentum = momentum

    def forward(self, x, training=True):
        if not training:
            x = x.transpose(0, 1)

            x_ancle = (x * 2 - 1).acos()
            x_final = x_ancle + self.x_running_rot.unsqueeze(-1)
            x_1 = (x_final.cos() + 1) / 2
            x_1 = x_1.transpose(0, 1)

        else:

            x = x.transpose(0, 1)
            x_ancle = (x * 2 - 1).acos()
            x_moving_rot = -1 * (x_ancle.min(-1)[0])

            self.x_running_rot[:] = self.momentum * self.x_running_rot + \
                                    (1 - self.momentum) * x_moving_rot
            x_final = x_ancle + x_moving_rot.unsqueeze(-1)
            x_1 = (x_final.cos() + 1) / 2
            x_1 = x_1.transpose(0, 1)

        return x_1


class QC_Norm_Correction(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super(QC_Norm_Correction, self).__init__()
        self.x_running_rot = Parameter(torch.zeros(num_features), requires_grad=False)
        self.momentum = momentum

    def forward(self, x, training=True):
        if not training:
            x = x.transpose(0, 1)

            x_ancle = (x * 2 - 1).acos()
            x_final = x_ancle + self.x_running_rot.unsqueeze(-1)
            x_1 = (x_final.cos() + 1) / 2
            x_1 = x_1.transpose(0, 1)

        else:
            x = x.transpose(0, 1)
            x_sum = x.sum(-1).unsqueeze(-1).expand(x.shape)
            x_mean = x_sum / x.shape[-1]

            x_mean_ancle = (x_mean * 2 - 1).acos()
            x_mean_rote = (np.pi / 2 - x_mean_ancle)

            x_moving_rot = (x_mean_rote.sum(-1) / x.shape[-1])
            self.x_running_rot[:] = self.momentum * self.x_running_rot + \
                                    (1 - self.momentum) * x_moving_rot
            x_ancle = (x * 2 - 1).acos()
            x_final = x_ancle + x_mean_rote
            x_1 = (x_final.cos() + 1) / 2
            x_1 = x_1.transpose(0, 1)

        return x_1


class QC_Norm_try3(nn.Module):
    def __init__(self, num_features, init_ang_inc=1, momentum=0.1,training = False):
        super(QC_Norm_try3, self).__init__()

        self.x_running_rot = Parameter(torch.zeros((num_features)), requires_grad=False)
        # self.ang_inc = Parameter(torch.ones(1)*init_ang_inc)
        self.ang_inc = Parameter(torch.tensor(init_ang_inc,dtype=torch.float32),requires_grad=True)
        self.momentum = momentum

        self.printed = False
        self.x_mean_ancle = 0
        self.x_mean_rote = 0
        self.input = 0
        self.output = 0

    def forward(self, x, training=True):
        if not training:
            if not self.printed:
                # print("self.ang_inc", self.ang_inc)
                self.printed = True
            x_1 = (self.x_running_rot * x)

        else:
            self.printed = False
            x = x.transpose(0, 1)
            x_sum = x.sum(-1).unsqueeze(-1).expand(x.shape)
            x_lack_sum = x_sum + x
            x_mean = x_lack_sum / x.shape[-1]

            ang_inc = (self.ang_inc > 0).float() * self.ang_inc + 1

            y = 0.5 / x_mean
            y = y.transpose(0, 1)
            y = y / ang_inc
            y = y.transpose(0, 1)

            x_moving_rot = (y.sum(-1) / x.shape[-1])

            self.x_running_rot[:] = self.momentum * self.x_running_rot + \
                                    (1 - self.momentum) * x_moving_rot

            x_1 = y * x
            x_1 = x_1.transpose(0, 1)

        return x_1

    def reset_parameters(self):
        self.reset_running_stats()
        self.ang_inc.data.zeros_()


class QC_Norm_Correction_try2(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super(QC_Norm_Correction_try2, self).__init__()
        self.x_running_rot = Parameter(torch.zeros(num_features), requires_grad=False)
        self.momentum = momentum
        self.x_l_0_5 = Parameter(torch.zeros(num_features), requires_grad=False)
        self.x_g_0_5 = Parameter(torch.zeros(num_features), requires_grad=False)

    def forward(self, x, training=True):
        if not training:
            x_1 = self.x_l_0_5 * (self.x_running_rot * (1 - x) + x)
            x_1 += self.x_g_0_5 * (self.x_running_rot * x)
        else:
            x = x.transpose(0, 1)
            x_sum = x.sum(-1)
            x_mean = x_sum / x.shape[-1]

            self.x_l_0_5[:] = ((x_mean <= 0.5).float())
            self.x_g_0_5[:] = ((x_mean > 0.5).float())

            y = self.x_l_0_5 * ((0.5 - x_mean) / (1 - x_mean))
            y += self.x_g_0_5 * (0.5 / x_mean)

            self.x_running_rot[:] = self.momentum * self.x_running_rot + \
                                    (1 - self.momentum) * y

            x = x.transpose(0, 1)
            x_1 = self.x_l_0_5 * (y * (1 - x) + x)
            x_1 += self.x_g_0_5 * (y * x)

        return x_1
