from ..qf_net.n_lyr import *
from ..qf_net.p_lyr import *
from ..qf_net.u_lyr import *
from ..qf_net.v_lyr import *
from ..qf_net.b_lyr import *

## Define the NN architecture
class Net(nn.Module):
    """
    class Net is to simply build up a network consisting of quantum layers ,using mat to represent the quantum gate.

    Args:
         img_size: the width /height of input image (width = height)
         layers: a 2-dimensions list. for example,[['u',4]['p',2]] means that the first layer is u-layer with 4 output qubit, and the second layer is p-layer with 2 output qubit
         training: whether  training
         binary: whether the input data should be binarized
         given_ang: initial angle for N-layer if used

    """
    def __init__(self,img_size,layers,training,binary,given_ang=[],train_ang=False,debug="False"):
        super(Net, self).__init__()

        self.in_size = img_size*img_size
        self.training = training
        self.layer = len(layers)
        self.layers = layers
        self.binary = binary
        loop_in_size = self.in_size
        self.debug = debug
        for idx in range(self.layer):
            fc_name = "fc"+str(idx)
            if layers[idx][0]=='u':
                setattr(self, fc_name, U_LYR(loop_in_size, layers[idx][1], bias=False))
            elif layers[idx][0]=='c':
                setattr(self, fc_name, B_LYR(loop_in_size, layers[idx][1], bias=False))
            elif layers[idx][0]=='f':
                setattr(self, fc_name, nn.Linear(loop_in_size, layers[idx][1]))
            elif layers[idx][0]=='p':
                setattr(self, fc_name, P_LYR(loop_in_size, layers[idx][1], bias=False))
            elif layers[idx][0]=='p2a':
                setattr(self, fc_name, Prob2amp())
            elif layers[idx][0]=='v':
                setattr(self, fc_name, VQC_Net(loop_in_size, layers[idx][1]))
            elif layers[idx][0]=='v10':
                setattr(self, fc_name, VQC_Net(loop_in_size, layers[idx][1],'vqc_10'))
            elif layers[idx][0]=='v5':
                setattr(self, fc_name, VQC_Net(loop_in_size, layers[idx][1],'vqc_5'))
            elif layers[idx][0]=='n':
                setattr(self, fc_name, N_LYR(num_features=layers[idx][1], init_ang_inc=given_ang[idx], training=train_ang))
            else:
                print("Not support layer name!")
                sys.exit(0)
            loop_in_size = layers[idx][1]


    def forward(self, x, training=1):
        x = x.view(-1, self.in_size)
        for layer_idx in range(self.layer):
            if self.binary and layer_idx==0:
                x = (binarize(x - 0.5) + 1) / 2
            x = getattr(self, "fc" + str(layer_idx))(x,training)

        if self.layers[-1][1] == 1:
            x = torch.cat((x, 1 - x), -1)

        return x


