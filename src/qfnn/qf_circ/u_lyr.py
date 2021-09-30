from .base import *
import numpy as np
################ Weiwen on 06-02-2021 ################
# QuantumFlow Weight Generation for U-Layer
######################################################
class U_LYR(LinnerCircuit):

    def __init__(self, n_qubits, n_repeats):
        """
      param n_qubits: input qubits of each unit
      param n_repeats: repeat times of each unit
        """
        LinnerCircuit.__init__(self, n_qubits, n_repeats)

    @classmethod
    def find_start(self, affect_count_table, target_num):
        for k in list(affect_count_table.keys())[::-1]:
            if target_num <= k:
                return k

    @classmethod
    def recursive_change(self, direction, start_point, target_num, sign, affect_count_table, quantum_gates):

        if start_point == target_num:
            # print("recursive_change: STOP")
            return

        gap = int(math.fabs(start_point - target_num))
        step = self.find_start(affect_count_table, gap)
        self.change_sign(sign, affect_count_table[step])
        quantum_gates.append(affect_count_table[step])

        if direction == "r":
            # print("recursive_change: From",start_point,"Right(-):",step)
            start_point = start_point - step
            direction = "l"
            self.recursive_change(direction, start_point, target_num, sign, affect_count_table, quantum_gates)

        else:
            # print("recursive_change: From",start_point,"Left(+):",step)
            start_point = start_point + step
            direction = "r"
            self.recursive_change(direction, start_point, target_num, sign, affect_count_table, quantum_gates)

    @classmethod
    def guarntee_upper_bound_algorithm(self, sign, target_num, total_len, digits):
        flag = "0" + str(digits) + "b"
        pre_num = 0
        affect_count_table = {}
        quantum_gates = []
        for i in range(digits):
            cur_num = pre_num + pow(2, i)
            pre_num = cur_num
            binstr_cur_num = format(cur_num, flag)
            affect_count_table[int(pow(2, binstr_cur_num.count("0")))] = binstr_cur_num

        if target_num in affect_count_table.keys():
            quantum_gates.append(affect_count_table[target_num])
            self.change_sign(sign, affect_count_table[target_num])

        else:
            direction = "r"
            start_point = self.find_start(affect_count_table, target_num)
            quantum_gates.append(affect_count_table[start_point])
            self.change_sign(sign, affect_count_table[start_point])
            self.recursive_change(direction, start_point, target_num, sign, affect_count_table, quantum_gates)

        return quantum_gates

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

        quantum_gates = self.guarntee_upper_bound_algorithm(sign, target_num, total_len, digits)

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