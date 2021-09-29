
import sys
import numpy as np
import numpy as np
from qiskit.tools.monitor import job_monitor
from qiskit import QuantumRegister,ClassicalRegister
from qiskit.extensions import  UnitaryGate
from qiskit import Aer, execute,IBMQ,transpile
import math
from qiskit import BasicAer
import copy

from .lib_gate import *
from .lib_circuit_base import *
def auth_output():
    print("welcome to Weiwen Jiang's Quantumflow")
################ Weiwen on 12-30-2020 ################
# Function: fire_ibmq from Listing 6
# Note: used for execute quantum circuit using 
#       simulation or ibm quantum processor
# Parameters: (1) quantum circuit; 
#             (2) number of shots;
#             (3) simulation or quantum processor;
#             (4) backend name if quantum processor.
######################################################
def fire_ibmq(circuit,shots,Simulation = True,backend_name='ibmq_essex'):     
    if not Simulation:
        provider = IBMQ.get_provider('ibm-q-academic')
        backend = provider.get_backend(backend_name)
    else:
        backend = Aer.get_backend('qasm_simulator')
    # circuit.save_statevector()
    job_ibm_q = execute(circuit, backend, shots=shots)
    if not Simulation:
        job_monitor(job_ibm_q)
    result_ibm_q = job_ibm_q.result()

    counts = result_ibm_q.get_counts()
    
    return counts


def get_unitary(circuit,IBMQ=None):   
    if IBMQ == None:
        backend = BasicAer.get_backend('unitary_simulator') 
    else:
        provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
        backend = provider.backend.ibmq_vigo
              
    job = backend.run(transpile(circuit, backend))
    
    unitary = job.result().get_unitary(circuit, decimals=9) # Execute the circuit
    return unitary

def add_measure(circuit,qubits,name):
    lenth = len(qubits)
    c_reg = ClassicalRegister(lenth,name)
    circuit.add_register(c_reg)
    for i in range(lenth):
        circuit.measure(qubits[i],c_reg[i])


################ Weiwen on 12-30-2020 ################
# Function: analyze from Listing 6
# Note: used for analyze the count on states to  
#       formulate the probability for each qubit
# Parameters: (1) counts returned by fire_ibmq; 
######################################################
def analyze(counts):
    mycount = {}
    for i in range(2):
        mycount[i] = 0
    for k,v in counts.items():
        bits = len(k) 
        for i in range(bits):            
            if k[bits-1-i] == "1":
                if i in mycount.keys():
                    mycount[i] += v
                else:
                    mycount[i] = v
    return mycount,bits

class NormerlizeCircuit(BaseCircuit):
    def __init__(self,n_qubits,n_repeats=1):
        """
        param n_qubits: input qubits of each unit
        param n_repeats: repeat times of each unit
        """    
        self.n_qubits = n_qubits
        self.n_repeats = n_repeats

    def add_norm_qubits(self,circuit):
        norm_qubits = self.add_qubits(circuit,"norm_qbits",self.n_qubits)
        return norm_qubits

    def add_out_qubits(self,circuit):
        out_qubits =self.add_qubits(circuit,"norm_output_qbits",self.n_qubits)
        return out_qubits
    
    def forward(self,circuit,input_qubits,norm_qubits,out_qubits,norm_flag,norm_para):
        for i in range(self.n_qubits):
            norm_init_rad = float(norm_para[i].sqrt().arcsin()*2)
            circuit.ry(norm_init_rad,norm_qubits[i])
            if norm_flag[i]:
                circuit.cx(input_qubits[i],out_qubits[i])
                circuit.x(input_qubits[i])
                circuit.ccx(input_qubits[i],norm_qubits[i],out_qubits[i])
            else:
                circuit.ccx(input_qubits[i],norm_qubits[i],out_qubits[i])

class UMatrixCircuit(BaseCircuit):    
    def __init__(self,n_qubits,n_repeats):
        """
        param n_qubits: input qubits of each unit
        param n_repeats: repeat times of each unit
        """    
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
