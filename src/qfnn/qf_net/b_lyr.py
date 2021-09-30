from .utils import *

class B_LYR(nn.Linear):

    def forward(self, input):
        binary_weight = binarize(self.weight)
        if self.bias is None:
            output = F.linear(input, binary_weight)
            # print(input,binary_weight, input.shape[-1])
            output = torch.div(output, input.shape[-1])
            # output = torch.pow(output, 2)

            return output
        else:
            print("Not Implement")
            sys.exit(0)

    def reset_parameters(self):
        # Glorot initialization
        in_features, out_features = self.weight.size()
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv

