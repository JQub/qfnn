from ..qf_circ.base import *
from qiskit.extensions import UnitaryGate

class UMatrixCircuit(BaseCircuit):
    """
    UMatrixCircuit is a class, which encodes unitary matrix into qubits using UnitaryGate

    Args:
         n_qubits: input qubits of each unit
         n_repeats: repeat times of each unit

    """
    def __init__(self,n_qubits,n_repeats):
        self.n_qubits = n_qubits
        self.n_repeats = n_repeats

    def add_input_qubits(self,circuit):
        inps = BaseCircuit.add_input_qubits(self,circuit,"u_matrix_input")
        return inps

    def forward(self,circuit,input_qubits,data_matrix,ids = None):
        for i in range(self.n_repeats):
            if ids == None:
                circuit.append(UnitaryGate(data_matrix, label="Input"), input_qubits[i][0:self.n_qubits])
            else:
                circuit.append(UnitaryGate(data_matrix[ids[i]], label="Input"), input_qubits[i][0:self.n_qubits])
