from .base import *
from ..qf_map.u_lyr_map import *

class U_LYR_Circ(LinnerCircuit):

    def __init__(self, n_qubits, n_repeats):
        """
        param n_qubits: input qubits of each unit
        param n_repeats: repeat times of each unit
        """
        LinnerCircuit.__init__(self, n_qubits, n_repeats)

    @classmethod
    def extract_from_weight(self, weights):
        # Find Z control gates according to weights
        w = (weights.detach().cpu().numpy())
        total_len = len(w)
        target_num = np.count_nonzero(w == -1)
        if target_num > total_len / 2:
            w = w * -1
        target_num = np.count_nonzero(w == -1)
        digits = int(math.log(total_len, 2))
        flag = "0" + str(digits) + "b"
        max_num = int(math.pow(2, digits))
        sign = {}
        for i in range(max_num):
            sign[format(i, flag)] = +1

        quantum_gates = Mapping_U_LYR(sign, target_num, digits)

        # Build the mapping from weight to final negative num
        fin_sign = list(sign.values())
        fin_weig = [int(x) for x in list(w)]
        sign_neg_index = []
        try:
            beg_pos = 0
            while True:
                find_pos = fin_sign.index(-1, beg_pos)
                # qiskit_position = int(format(find_pos,flag)[::-1],2)
                sign_neg_index.append(find_pos)
                beg_pos = find_pos + 1
        except Exception as exception:
            pass

        weight_neg_index = []
        try:
            beg_pos = 0
            while True:
                find_pos = fin_weig.index(-1, beg_pos)
                weight_neg_index.append(find_pos)
                beg_pos = find_pos + 1
        except Exception as exception:
            pass

        map = {}
        for i in range(len(sign_neg_index)):
            map[sign_neg_index[i]] = weight_neg_index[i]

        ret_index = list([-1 for i in range(len(fin_weig))])

        for k, v in map.items():
            ret_index[k] = v

        for i in range(len(fin_weig)):
            if ret_index[i] != -1:
                continue
            for j in range(len(fin_weig)):
                if j not in ret_index:
                    ret_index[i] = j
                    break
        return quantum_gates, ret_index


# if __name__ == "__main__":
#     import torch
#     from qiskit import QuantumRegister, QuantumCircuit
#     weight_1 = torch.tensor([[1., 1., 1., 1., 1., 1., 1., -1., 1., -1., 1., -1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., -1., 1., -1., 1., -1., 1., 1., 1., 1.],
#                              [-1., -1., -1., -1., -1., -1., -1., -1., -1., 1., -1., 1., -1., -1., -1., -1.,-1., -1., -1., -1., -1., -1., -1., -1., -1., 1., -1., 1., -1., -1., -1., -1.]])
#     #define your input and output number
#     u_layer = U_LYR_Circ(5,2)
#     #init circuit
#     opt_circ = QuantumCircuit()
#     #add input qubit to your circuit
#     inps = u_layer.add_input_qubits(opt_circ)
#     #add auxiliary qubit to your circuit
#     aux =u_layer.add_aux(opt_circ)
#     #add output qubit to your circuit
#     u_layer_out_qubits = u_layer.add_out_qubits(opt_circ)
#     quantum_matrix = torch.eye(32)
#     #add ulayer to your circuit
#     u_layer.forward(opt_circ,weight_1,inps,u_layer_out_qubits,quantum_matrix,aux)
#     print(opt_circ)