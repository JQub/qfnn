# -*- encoding: utf-8 -*-
'''
Filename         :lib_circuit_base.py
Description      :This document is used for fundamental class of quantum circuit
Time             :2021/09/26 13:58:23
Author           :Weiwen Jiang & Zhirui Hu
Version          :1.0
'''


import sys
import numpy as np
import numpy as np
from qiskit.tools.monitor import job_monitor
from qiskit import QuantumRegister
from qiskit.extensions import  UnitaryGate
from qiskit import Aer, execute,IBMQ,transpile
import math
from qiskit import BasicAer
import copy
import abc

class BaseCircuit(metaclass= abc.ABCMeta):
    """BaseCircuit is a class, which includes fundamental functions of a circuit module.

    Args:
         name (n_qubits): input qubits of each unit
         name (n_repeats): repeat times of each unit

    """
    def __init__(self,n_qubits,n_repeats):
        self.n_qubits = n_qubits
        self.n_repeats = n_repeats

    def add_qubits(self,circuit,name,number):
        """
        Function add_qubits is to add a group of qubits to a circuit. [Test at 09/29]

        Args:
            name (circuit): The circuit that you add the unit at the end
            name (name): The name of the group
            name (number):  The number of qubits in the group.

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
             name (circuit): The  circuit that you add the unit at the end
             name (name): The name of the group

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
