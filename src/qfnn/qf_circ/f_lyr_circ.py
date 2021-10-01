from .base import *

class F_LYR_Circ(LinnerCircuit):

    def __init__(self, n_qubits, n_repeats):
        """
      param n_qubits: input qubits of each unit
      param n_repeats: repeat times of each unit
        """
        LinnerCircuit.__init__(self, n_qubits, n_repeats)

    @classmethod
    def AinB(self, A, B):
        idx_a = []
        for i in range(len(A)):
            if A[i] == "1":
                idx_a.append(i)
        flag = True
        for j in idx_a:
            if B[j] == "0":
                flag = False
                break
        return flag

    @classmethod
    def extract_from_weight(self, weights):
        # Find Z control gates according to weights
        w = (weights.detach().cpu().numpy())
        total_len = len(w)
        digits = int(math.log(total_len, 2))
        flag = "0" + str(digits) + "b"
        max_num = int(math.pow(2, digits))
        sign = {}
        for i in range(max_num):
            sign[format(i, flag)] = +1
        sign_expect = {}
        for i in range(max_num):
            sign_expect[format(i, flag)] = int(w[i])

        order_list = []
        for i in range(digits + 1):
            for key in sign.keys():
                if key.count("1") == i:
                    order_list.append(key)

        gates = []
        sign_cur = copy.deepcopy(sign_expect)
        for idx in range(len(order_list)):
            key = order_list[idx]
            if sign_cur[key] == -1:
                gates.append(key)
                for cor_idx in range(idx, len((order_list))):
                    if self.AinB(key, order_list[cor_idx]):
                        sign_cur[order_list[cor_idx]] = (-1) * sign_cur[order_list[cor_idx]]
        return gates, None


