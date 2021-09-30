from .utils import *

class batch_adj(nn.Module):
    def __init__(self, num_features, init_ang_inc=1, momentum=0.1,training = False):
        super(batch_adj, self).__init__()

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


class indiv_adj(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super(indiv_adj, self).__init__()
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
