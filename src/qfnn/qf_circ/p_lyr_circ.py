from math import log2,sqrt,asin
from .base import *
from .gates import ExtendGate 
# just for temp
class P_LYR_Circ(LinnerCircuit):
    """
    P_LYR_Circ is a class, which includes functions to build p-layer circuit

    Args:
         n_qubits: input qubits of each unit
         n_repeats: repeat times of each unit

    """
    def __init__(self, n_qubits, n_repeats):

        self.n_qubits = n_qubits
        self.n_repeats = n_repeats
        self.n_log = log2(self.n_qubits)

        if self.n_qubits == 1 :
            print('PLayerCircuit: The input size does not support 1 !')
            sys.exit(0)
        if abs(self.n_log - round(self.n_log))>1e-2:
            print('PLayerCircuit: The input size must be 2^n!')


    def add_out_qubits(self, circuit,name = "p_layer_qbits"):
        out_qubits = self.add_qubits(circuit, name, self.n_repeats)
        return out_qubits

    def add_aux(self, circuit,name ="aux_qbit"):
        """
        Function add  add_aux is to add a group of qubits as input qubit .

        Args:
             circuit: The  circuit that you add the unit at the end

        Returns:
             qubits: The register of qubits

        """
        if self.n_qubits < 4:
            return None
        # TODO: 09/30, Potential bug.
        else:
            # q_num = round(self.n_log) + 1
            # aux_num = q_num -1
            aux = self.add_qubits(circuit,name , round(self.n_log)*(1+self.n_repeats))
        # else:
        #     print('The input size is too big. Qubits should be less than 4.')
        #     sys.exit(0)
        return aux


    def forward(self, circuit, weight, in_qubits, out_qubits,aux = []):
        """
        Function add forward is to add the circuit of batch normalization.

        Args:
             circuit: The  circuit that you add the unit at the end
             weight: A list of binary weight.
             in_qubits: The register of input qubits
             out_qubit: The register of output qubits
             aux: aux qubits

        """
        for i in range(self.n_repeats):
            # mul weight
            if weight[i].sum() < 0:
                weight[i] = weight[i] * -1
            idx = 0
            for idx in range(weight[i].flatten().size()[0]):
                if weight[i][idx] == -1:
                    circuit.x(in_qubits[idx])
            circuit.barrier()

            # sum and pow2
            if self.n_qubits == 2:
                circuit.h(out_qubits[i])
                circuit.x(out_qubits[i])
                circuit.cz(in_qubits[0], out_qubits[i])
                circuit.cz(in_qubits[1], out_qubits[i])
                circuit.h(out_qubits[i])
                circuit.x(out_qubits[i])
            else:
                #add weight
                operate_qubits = []
                aux_qubits = []
                encoding_num = round(self.n_log)
                operate_qubits.append(in_qubits[self.n_qubits-1])
                for k in range(encoding_num):
                    operate_qubits.append(aux[k+encoding_num*i])
                for k in range(encoding_num):
                    aux_qubits.append(aux[encoding_num+encoding_num*i+k])
                circuit.h(operate_qubits[1:])
                for j in range(self.n_qubits):
                    operate_qubits[0]=in_qubits[self.n_qubits-1-j]
                    state = "{0:b}".format(self.n_qubits+j).zfill(encoding_num)
                    state = state[::-1]
                    ExtendGate.neg_weight_gate(circuit,operate_qubits,aux_qubits,state)
                    circuit.barrier()

                #sum2
                circuit.h(operate_qubits[1:])
                circuit.x(operate_qubits[1:])


                qbits = operate_qubits[1:]
                if encoding_num == 2:
                    circuit.ccx(qbits[0], qbits[1], out_qubits[i])
                else:
                    operate_qubits2 = []
                    aux_qubits2 = []
                    for k in range(encoding_num):
                        operate_qubits2.append(qbits[k])
                        if k < encoding_num - 2:
                            aux_qubits2.append(aux_qubits[k])
                    operate_qubits2.append(out_qubits[i])
                    ExtendGate.cnx(circuit, operate_qubits2, aux_qubits2, encoding_num+1)

            # recover
            for idx in range(weight[i].flatten().size()[0]):
                if weight[i][idx] == -1:
                    circuit.x(in_qubits[idx])
            circuit.barrier()



class P_Neuron_Circ(P_LYR_Circ):
    """
    P_Neuron_Circ is a class, which includes functions to build P-Neuron circuit. P-Layer consists of serverl P-Neuron.A neuron only repeats onece.

    Args:
         n_qubits: input qubits of each unit

    """
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.n_repeats = 1

        P_LYR_Circ.__init__(self,n_qubits,self.n_repeats)


    def forward(self, circuit, weight, in_qubits, out_qubits,aux = [],ang=[]):
        """
        Function forward is to add the circuit of batch normalization.

        Args:
             circuit: The  circuit that you add the unit at the end
             weight: A list of binary weight.
             in_qubits: The register of input qubits
             out_qubit: The register of output qubits
             ang: the angle list of angle encoding
             aux: aux qubits

        """
        for i in range (self.n_qubits):
            if len(ang)>i:
                angle = float(2*math.asin(math.sqrt(float(ang[i]))))
                circuit.ry(angle, in_qubits[i])

        for i in range(self.n_repeats):
            # mul weight
            if weight[i].sum() < 0:
                weight[i] = weight[i] * -1
            idx = 0
            for idx in range(weight[i].flatten().size()[0]):
                if weight[i][idx] == -1:
                    circuit.x(in_qubits[idx])
            circuit.barrier()
            # TODO: Potential bug for P-LYR

            # sum and pow2
            if self.n_qubits == 2:
                circuit.h(out_qubits[i])
                circuit.x(out_qubits[i])
                circuit.cz(in_qubits[0], out_qubits[i])
                circuit.cz(in_qubits[1], out_qubits[i])
                circuit.h(out_qubits[i])
                circuit.x(out_qubits[i])
            else:
                #add weight
                operate_qubits = []
                aux_qubits = []
                encoding_num = round(self.n_log)
                operate_qubits.append(in_qubits[self.n_qubits-1])
                for k in range(encoding_num):
                    operate_qubits.append(aux[k])
                for k in range(encoding_num):
                    aux_qubits.append(aux[encoding_num+k])
                circuit.h(operate_qubits[1:])
                for j in range(self.n_qubits):
                    operate_qubits[0]=in_qubits[self.n_qubits-1-j]
                    state = "{0:b}".format(self.n_qubits+j).zfill(encoding_num)
                    state = state[::-1]
                    ExtendGate.neg_weight_gate(circuit,operate_qubits,aux_qubits,state)
                    circuit.barrier()

                #sum2
                circuit.h(operate_qubits[1:])
                circuit.x(operate_qubits[1:])


                qbits = operate_qubits[1:]
                if encoding_num == 2:
                    circuit.ccx(qbits[0], qbits[1], out_qubits[i])
                else:
                    operate_qubits2 = []
                    aux_qubits2 = []
                    for k in range(encoding_num):
                        operate_qubits2.append(qbits[k])
                        if k < encoding_num - 2:
                            aux_qubits2.append(aux_qubits[k])
                    operate_qubits2.append(out_qubits[i])
                    ExtendGate.cnx(circuit, operate_qubits2, aux_qubits2, encoding_num+1)
