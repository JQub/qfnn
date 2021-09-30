from .base import *

# just for temp
class P_LYR_Circ(BaseCircuit):
    def __init__(self, n_qubits, n_repeats):
        """
        param n_qubits: input qubits of each unit
        param n_repeats: repeat times of each unit
        """
        self.n_qubits = n_qubits
        self.n_repeats = n_repeats

        if self.n_qubits != 2 or self.n_repeats != 2:
            print(
                'PLayerCircuit: The input size or output size is not 2. Now thet p-layer only support 2 inputs and 2 outputs!')
            sys.exit(0)

    def add_out_qubits(self, circuit):
        out_qubits = self.add_qubits(circuit, "p_layer_qbits", self.n_qubits)
        return out_qubits

    def forward(self, circuit, weight, in_qubits, out_qubits):
        for i in range(self.n_repeats):
            # mul weight
            if weight[i].sum() < 0:
                weight[i] = weight[i] * -1
            idx = 0
            for idx in range(weight[i].flatten().size()[0]):
                if weight[i][idx] == -1:
                    circuit.x(in_qubits[idx])
            # TODO: Potential bug for P-LYR
            # sum and pow2
            circuit.h(out_qubits[i])
            circuit.cz(in_qubits[0], out_qubits[i])
            circuit.x(out_qubits[i])
            circuit.cz(in_qubits[1], out_qubits[i])
            circuit.x(out_qubits[i])
            circuit.h(out_qubits[i])
            circuit.x(out_qubits[i])
            # recover
            for idx in range(weight[i].flatten().size()[0]):
                if weight[i][idx] == -1:
                    circuit.x(in_qubits[idx])
            circuit.barrier(in_qubits, out_qubits)