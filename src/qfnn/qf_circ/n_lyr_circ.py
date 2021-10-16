from .base import *


class N_LYR_Circ(BaseCircuit):
    """
    N_LYR_Circ is a class, which includes functions to build N-layer circuit(for batch normalization)

    Args:
         n_qubits: input qubits of each unit
         n_repeats: repeat times of each unit

    """
    def __init__(self, n_qubits, n_repeats=1):

        self.n_qubits = n_qubits
        self.n_repeats = n_repeats

    def add_norm_qubits(self, circuit):
        norm_qubits = self.add_qubits(circuit, "norm_qbits", self.n_qubits)
        return norm_qubits

    def add_out_qubits(self, circuit):
        out_qubits = self.add_qubits(circuit, "norm_output_qbits", self.n_qubits)
        return out_qubits

    def forward(self, circuit, input_qubits, norm_qubits, out_qubits, norm_flag, norm_para):
        """
        Function forward is to add the circuit of N-layer.

        Args:
             circuit: The  circuit that you add the unit at the end
             input_qubits: The register of input qubits
             norm_qubits: The register of qubits that do angle-encoding.
             out_qubits: The register of output qubits
             norm_flag: the direction of batch normalization.
             norm_para: the parameter of batch normalization  encoded in the angle of qubits.

        """
        for i in range(self.n_qubits):
            norm_init_rad = float(norm_para[i].sqrt().arcsin() * 2)
            circuit.ry(norm_init_rad, norm_qubits[i])
            if norm_flag[i]:
                circuit.cx(input_qubits[i], out_qubits[i])
                circuit.x(input_qubits[i])
                circuit.ccx(input_qubits[i], norm_qubits[i], out_qubits[i])
            else:
                circuit.ccx(input_qubits[i], norm_qubits[i], out_qubits[i])
