# -*- encoding: utf-8 -*-
'''
Filename         :lib_circuit_base.py
Description      :This document is used for fundamental class of quantum circuit
Time             :2021/09/26 13:58:23
Author           :Weiwen Jiang & Zhirui Hu
Version          :1.0
'''


import sys
import abc
from qiskit import QuantumRegister
from qiskit.extensions import UnitaryGate
from .gates import *
import copy
import numpy as np
import math

class BaseCircuit(metaclass= abc.ABCMeta):
    """BaseCircuit is a class, which includes fundamental functions of a circuit module.

    Args:
         n_qubits: input qubits of each unit
         n_repeats: repeat times of each unit

    """
    def __init__(self,n_qubits,n_repeats):
        self.n_qubits = n_qubits
        self.n_repeats = n_repeats

    def add_qubits(self,circuit,name,number):
        """
        Function add_qubits is to add a group of qubits to a circuit. [Test at 09/29]

        Args:
            circuit: The circuit that you add the unit at the end
            name: The name of the group
            number:  The number of qubits in the group.

        Returns:
            qubits: The register of qubits

        """
        qubits = QuantumRegister(number,name)
        circuit.add_register(qubits)
        return qubits


    def add_input_qubits(self,circuit,name):
        """
        Function add_input_qubits is to add a group of qubits as input qubit .

        Args:
             circuit: The  circuit that you add the unit at the end
             name: The name of the group

        Returns:
             qubits: The register of qubits

        """
        inps = []
        for i in range(self.n_repeats):
            inp = QuantumRegister(self.n_qubits,name+str(i)+"_qbit")
            circuit.add_register(inp)
            inps.append(inp)
        return inps

    ############# Weiwen&Zhirui on 2021/09/26 ############
    # Function: add_qubits
    # Note: Add a circuit unit of the clss at the end.
    # Parameters:
    #     circuit: The  circuit that you add the unit at the end
    ######################################################
    @abc.abstractclassmethod
    def forward(self,circuit):
        pass


class LinnerCircuit(BaseCircuit):
    def __init__(self, n_qubits, n_repeats):
        """
      param n_qubits: input qubits of each unit
      param n_repeats: repeat times of each unit
        """
        self.n_qubits = n_qubits
        self.n_repeats = n_repeats
        if self.n_qubits > 4:
            print('The input size is too big. Qubits should be less than 4.')
            sys.exit(0)

    def add_aux(self, circuit):
        if self.n_qubits < 3:
            aux = self.add_qubits(circuit, "aux_qbit", 1)
        # TODO: 09/30, Potential bug.
        elif self.n_qubits >= 3:
            aux = self.add_qubits(circuit, "aux_qbit", 2)
        else:
            print('The input size is too big. Qubits should be less than 4.')
            sys.exit(0)
        return aux

    def add_input_qubits(self, circuit):
        inputs = BaseCircuit.add_input_qubits(self, circuit, "u_layer")
        return inputs

    def add_out_qubits(self, circuit):
        out_qubits = self.add_qubits(circuit, "u_layer_output_qbit", self.n_repeats)
        return out_qubits

    @abc.abstractclassmethod
    def extract_from_weight(weight):
        pass

    def add_weight(self, circuit, weight, in_qubits, data_matrix=None, aux=[]):
        for i in range(self.n_repeats):
            n_q_gates, n_idx = self.extract_from_weight(weight[i])
            if data_matrix != None and n_idx != None:
                circuit.append(UnitaryGate(data_matrix[n_idx], label="Input"), in_qubits[i][0:self.n_qubits])
            qbits = in_qubits[i]
            for gate in n_q_gates:
                z_count = gate.count("1")
                # z_pos = get_index_list(gate,"1")
                z_pos = self.get_index_list(gate[::-1], "1")

                if z_count == 1:
                    circuit.z(qbits[z_pos[0]])
                elif z_count == 2:
                    circuit.cz(qbits[z_pos[0]], qbits[z_pos[1]])
                else:
                    operate_qubits = []
                    aux_qubits = []
                    for k in range(z_count):
                        operate_qubits.append(qbits[z_pos[i]])
                        if k < z_count - 2:
                            aux_qubits.append(aux[i])
                    ExtendGate.cnz(circuit, operate_qubits, aux_qubits, z_count)
                # elif z_count == 3:
                #     ExtendGate.ccz(circuit, qbits[z_pos[0]], qbits[z_pos[1]], qbits[z_pos[2]], aux[0])
                # elif z_count == 4:
                #     ExtendGate.cccz(circuit, qbits[z_pos[0]], qbits[z_pos[1]], qbits[z_pos[2]], qbits[z_pos[3]], aux[0],
                #                     aux[1])
                # else:
                #     print("Not support yet!")
                #     sys.exit(0)
        circuit.barrier()

    def sum2(self, circuit, in_qubits, out_qubit, aux=[]):
        for i in range(self.n_repeats):
            circuit.h(in_qubits[i])
            circuit.x(in_qubits[i])
        circuit.barrier()
        for i in range(self.n_repeats):
            qbits = in_qubits[i]
            if self.n_qubits == 1:
                circuit.cx(qbits[0], out_qubit[i])
            elif self.n_qubits == 2:
                circuit.ccx(qbits[0], qbits[1], out_qubit[i])
            else:
                operate_qubits = []
                aux_qubits = []
                for k in range(self.n_qubits):
                    operate_qubits.append(qbits[i])
                    if k < self.n_qubits - 2:
                        aux_qubits.append(aux[i])
                operate_qubits.append(out_qubit[i])
                ExtendGate.cnz(circuit, operate_qubits, aux_qubits, self.n_qubits+1)
            #
            # elif self.n_qubits == 3:
            #     ExtendGate.cccx(circuit, qbits[0], qbits[1], qbits[2], out_qubit[i], aux[0], aux[1])
            # elif self.n_qubits == 4:
            #     ExtendGate.ccccx(circuit, qbits[0], qbits[1], qbits[2], qbits[3], out_qubit[i], aux[0], aux[1])

    def forward(self, circuit, weight, in_qubits, out_qubit, data_matrix=None, aux=[]):
        self.add_weight(circuit, weight, in_qubits, data_matrix, aux)
        self.sum2(circuit, in_qubits, out_qubit, aux)

    @classmethod
    def get_index_list(self, input, target):
        index_list = []
        try:
            beg_pos = 0
            while True:
                find_pos = input.index(target, beg_pos)
                index_list.append(find_pos)
                beg_pos = find_pos + 1
        except Exception as exception:
            pass
        return index_list

