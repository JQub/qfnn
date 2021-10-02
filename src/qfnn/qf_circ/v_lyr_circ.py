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

from .gates import *
from .base import *

class V_LYR_Circ(BaseCircuit):    
    def __init__(self, n_qubits,n_repeats):
        """
        param n_qubits: input qubits of each unit
        param n_repeats: repeat times of each unit
        """    
        self.n_qubits = n_qubits
        self.n_repeats = n_repeats



    def add_input_qubits(self,circuit):
        inps = BaseCircuit.add_input_qubits(self,circuit,"vqc_input")
        return inps



    #define the circuit
    def vqc_10(self,circuit,input_qubits,thetas):
        # print(input_qubits)
        #head ry part 
        for i in range(0,self.n_qubits):
            circuit.ry(thetas[i], input_qubits[i])
        circuit.barrier(input_qubits)
        
        #cz part
        for i in range(self.n_qubits-1):
            circuit.cz(input_qubits[self.n_qubits-2-i],input_qubits[self.n_qubits-1-i])
        circuit.cz(input_qubits[0],input_qubits[self.n_qubits-1])
        circuit.barrier(input_qubits)

        #tail ry part
        for i in range(0,self.n_qubits):
            circuit.ry(thetas[i+self.n_qubits], input_qubits[i])


    def vqc_5(self,circuit,input_qubits,thetas):
        for i in range(0,self.n_qubits):
            circuit.rx(thetas[i],input_qubits[i])
        for i in range(0,self.n_qubits):
            circuit.rz(thetas[self.n_qubits+i],input_qubits[i])
        
        circuit.barrier(input_qubits)
        cnt = 0
        for i in range(self.n_qubits-1,-1,-1):
            for j in range(self.n_qubits-1,-1,-1):
                if j == i:
                    continue
                else:
                    circuit.crz(thetas[2*self.n_qubits + cnt],input_qubits[i],input_qubits[j])
                    cnt = cnt +1
            circuit.barrier(input_qubits)
        for i in range(0,self.n_qubits):
            circuit.rx(thetas[5*self.n_qubits+i],input_qubits[i])
        for i in range(0,self.n_qubits):
            circuit.rz(thetas[6*self.n_qubits+i],input_qubits[i])

    def get_parameter_number(self,vqc_name):
        if vqc_name == 'v10':
            return int(2*self.n_qubits)
        elif vqc_name == 'v5':
            return int(7*self.n_qubits)

    
    def forward(self,circuit,input_qubits,vqc_name,thetas):
        if vqc_name == 'v10':
            for i in range(self.n_repeats):
                self.vqc_10(circuit,input_qubits[i],thetas)
        elif vqc_name == 'v5':
            for i in range(self.n_repeats):
                self.vqc_5(circuit,input_qubits[i],thetas)
