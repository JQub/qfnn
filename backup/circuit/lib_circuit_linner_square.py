
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

from .lib_circuit_base import *
from .lib_gate import *

class LinnerCircuit(BaseCircuit):
    def __init__(self,n_qubits,n_repeats):
        """
      param n_qubits: input qubits of each unit
      param n_repeats: repeat times of each unit
        """    
        self.n_qubits = n_qubits
        self.n_repeats = n_repeats
        if self.n_qubits > 4:
            print('The input size is too big. Qubits should be less than 4.')
            sys.exit(0)
        
    
    def add_aux(self,circuit):
        if self.n_qubits < 3:
            aux = self.add_qubits(circuit,"aux_qbit",1)
        # TODO: 09/30, Potential bug.
        elif self.n_qubits >= 3:
            aux = self.add_qubits(circuit,"aux_qbit",2)
        else:
            print('The input size is too big. Qubits should be less than 4.')
            sys.exit(0)
        return aux

    def add_input_qubits(self,circuit):
        inputs = BaseCircuit.add_input_qubits(self,circuit,"u_layer")
        return inputs
    
    def add_out_qubits(self,circuit):
        out_qubits = self.add_qubits(circuit,"u_layer_output_qbit",self.n_repeats)
        return out_qubits
    
    @abc.abstractclassmethod
    def extract_from_weight(weight):
        pass

    def add_weight(self,circuit,weight,in_qubits, data_matrix = None,aux = []):
        for i in range(self.n_repeats):
            n_q_gates,n_idx = self.extract_from_weight(weight[i])
            if data_matrix != None and n_idx != None:
                circuit.append(UnitaryGate(data_matrix[n_idx], label="Input"), in_qubits[i][0:self.n_qubits])
            qbits = in_qubits[i]
            for gate in n_q_gates:
                z_count = gate.count("1")
                # z_pos = get_index_list(gate,"1")
                z_pos = self.get_index_list(gate[::-1],"1")
                if z_count==1:
                    circuit.z(qbits[z_pos[0]])
                elif z_count==2:
                    circuit.cz(qbits[z_pos[0]],qbits[z_pos[1]])
                elif z_count==3:
                    ExtendGate.ccz(circuit,qbits[z_pos[0]],qbits[z_pos[1]],qbits[z_pos[2]],aux[0])
                elif z_count==4:
                    ExtendGate.cccz(circuit,qbits[z_pos[0]],qbits[z_pos[1]],qbits[z_pos[2]],qbits[z_pos[3]],aux[0],aux[1])
                else:
                    print("Not support yet!")
                    sys.exit(0)
        circuit.barrier()
    
    def sum2(self,circuit,in_qubits,out_qubit,aux = []):
        for i in range(self.n_repeats):
            circuit.h(in_qubits[i])
            circuit.x(in_qubits[i])
        circuit.barrier()
        for i in range(self.n_repeats):
            qbits = in_qubits[i]
            if self.n_qubits==1:
                circuit.cx(qbits[0],out_qubit[i])
            elif self.n_qubits==2:
                circuit.ccx(qbits[0],qbits[1],out_qubit[i])
            elif self.n_qubits==3:
                ExtendGate.cccx(circuit,qbits[0],qbits[1],qbits[2],out_qubit[i],aux[0],aux[1])
            elif self.n_qubits==4:
                ExtendGate.ccccx(circuit,qbits[0],qbits[1],qbits[2],qbits[3],out_qubit[i],aux[0],aux[1])

    def forward(self,circuit,weight,in_qubits,out_qubit, data_matrix = None,aux = []):
        self.add_weight(circuit,weight,in_qubits,data_matrix ,aux)
        self.sum2(circuit,in_qubits,out_qubit,aux)



    @classmethod
    def get_index_list(self,input,target):
        index_list = []
        try:
            beg_pos = 0
            while True:
                find_pos = input.index(target,beg_pos)
                index_list.append(find_pos)
                beg_pos = find_pos+1
        except Exception as exception:        
            pass    
        return index_list

    @classmethod   
    def change_sign(self,sign,bin):
        affect_num = [bin]
        one_positions = []
        try:
            beg_pos = 0
            while True:
                find_pos = bin.index("1",beg_pos)
                one_positions.append(find_pos)
                beg_pos = find_pos+1
        except Exception as exception:
            # print("Not Found")
            pass
        for k,v in sign.items():
            change = True
            for pos in one_positions:
                if k[pos]=="0":                
                    change = False
                    break
            if change:
                sign[k] = -1*v
    

################ Weiwen on 06-02-2021 ################
# QuantumFlow Weight Generation for U-Layer
######################################################    
class ULayerCircuit(LinnerCircuit):

    def __init__(self,n_qubits,n_repeats):
        """
      param n_qubits: input qubits of each unit
      param n_repeats: repeat times of each unit
        """  
        LinnerCircuit.__init__(self,n_qubits,n_repeats) 

    @classmethod
    def find_start(self,affect_count_table,target_num):
        for k in list(affect_count_table.keys())[::-1]:
            if target_num<=k:
                return k

    @classmethod
    def recursive_change(self,direction,start_point,target_num,sign,affect_count_table,quantum_gates):
        
        if start_point == target_num:
            # print("recursive_change: STOP")
            return
        
        gap = int(math.fabs(start_point-target_num))    
        step = self.find_start(affect_count_table,gap)
        self.change_sign(sign,affect_count_table[step])
        quantum_gates.append(affect_count_table[step])
        
        if direction=="r": 
            # print("recursive_change: From",start_point,"Right(-):",step)
            start_point = start_point - step
            direction = "l"
            self.recursive_change(direction,start_point,target_num,sign,affect_count_table,quantum_gates)
            
        else:        
            # print("recursive_change: From",start_point,"Left(+):",step)
            start_point = start_point + step
            direction = "r"
            self.recursive_change(direction,start_point,target_num,sign,affect_count_table,quantum_gates)
        
    
    @classmethod
    def guarntee_upper_bound_algorithm(self,sign,target_num,total_len,digits):        
        flag = "0"+str(digits)+"b"
        pre_num = 0
        affect_count_table = {}
        quantum_gates = []
        for i in range(digits):
            cur_num = pre_num + pow(2,i)
            pre_num = cur_num
            binstr_cur_num = format(cur_num,flag) 
            affect_count_table[int(pow(2,binstr_cur_num.count("0")))] = binstr_cur_num   
        
        if target_num in affect_count_table.keys():
            quantum_gates.append(affect_count_table[target_num])
            self.change_sign(sign,affect_count_table[target_num])  
      
        else:
            direction = "r"
            start_point = self.find_start(affect_count_table,target_num)
            quantum_gates.append(affect_count_table[start_point])
            self.change_sign(sign,affect_count_table[start_point])
            self.recursive_change(direction,start_point,target_num,sign,affect_count_table,quantum_gates)
        
        return quantum_gates
    @classmethod
    def extract_from_weight(self,weights):    
        # Find Z control gates according to weights
        w = (weights.detach().cpu().numpy())
        total_len = len(w)
        target_num = np.count_nonzero(w == -1)
        if target_num > total_len/2:
            w = w*-1
        target_num = np.count_nonzero(w == -1)    
        digits = int(math.log(total_len,2))
        flag = "0"+str(digits)+"b"
        max_num = int(math.pow(2,digits))
        sign = {}
        for i in range(max_num):        
            sign[format(i,flag)] = +1

        quantum_gates = self.guarntee_upper_bound_algorithm(sign,target_num,total_len,digits)
        
        # Build the mapping from weight to final negative num 
        fin_sign = list(sign.values())
        fin_weig = [int(x) for x in list(w)]
        sign_neg_index = []    
        try:
            beg_pos = 0
            while True:
                find_pos = fin_sign.index(-1,beg_pos)            
                # qiskit_position = int(format(find_pos,flag)[::-1],2)                            
                sign_neg_index.append(find_pos)
                beg_pos = find_pos+1
        except Exception as exception:        
            pass  
    

        weight_neg_index = []
        try:
            beg_pos = 0
            while True:
                find_pos = fin_weig.index(-1,beg_pos)
                weight_neg_index.append(find_pos)
                beg_pos = find_pos+1
        except Exception as exception:        
            pass    
    
        map = {}
        for i in range(len(sign_neg_index)):
            map[sign_neg_index[i]] = weight_neg_index[i]
    
        ret_index = list([-1 for i in range(len(fin_weig))])
        
        
        for k,v in map.items():
            ret_index[k]=v
        
        
        for i in range(len(fin_weig)):
            if ret_index[i]!=-1:
                continue
            for j in range(len(fin_weig)):
                if j not in ret_index:
                    ret_index[i]=j
                    break  
        return quantum_gates,ret_index

class FFNNCircuit(LinnerCircuit):

    def __init__(self,n_qubits,n_repeats):
        """
      param n_qubits: input qubits of each unit
      param n_repeats: repeat times of each unit
        """  
        LinnerCircuit.__init__(self,n_qubits,n_repeats) 

    @classmethod
    def AinB(self,A,B):
        idx_a = []
        for i in range(len(A)):
            if A[i]=="1":
                idx_a.append(i)    
        flag = True
        for j in idx_a:
            if B[j]=="0":
                flag=False
                break
        return flag

    @classmethod
    def extract_from_weight(self,weights):        
        # Find Z control gates according to weights
        w = (weights.detach().cpu().numpy())
        total_len = len(w)            
        digits = int(math.log(total_len,2))
        flag = "0"+str(digits)+"b"
        max_num = int(math.pow(2,digits))
        sign = {}
        for i in range(max_num):        
            sign[format(i,flag)] = +1    
        sign_expect = {}
        for i in range(max_num):
            sign_expect[format(i,flag)] = int(w[i])    
        
        order_list = []
        for i in range(digits+1):
            for key in sign.keys():
                if key.count("1") == i:
                    order_list.append(key)    
        
        gates = []    
        sign_cur = copy.deepcopy(sign_expect)
        for idx in range(len(order_list)):
            key = order_list[idx]
            if sign_cur[key] == -1:
                gates.append(key)
                for cor_idx in range(idx,len((order_list))):
                    if self.AinB(key,order_list[cor_idx]):
                        sign_cur[order_list[cor_idx]] = (-1)*sign_cur[order_list[cor_idx]]    
        return gates,None



# just for temp 
class PLayerCircuit(BaseCircuit):
    def __init__(self,n_qubits,n_repeats):
        """
        param n_qubits: input qubits of each unit
        param n_repeats: repeat times of each unit
        """  
        self.n_qubits = n_qubits
        self.n_repeats = n_repeats

        if self.n_qubits != 2 or self.n_repeats != 2:
            print('PLayerCircuit: The input size or output size is not 2. Now thet p-layer only support 2 inputs and 2 outputs!')
            sys.exit(0)


    def add_out_qubits(self,circuit):
        out_qubits  = self.add_qubits(circuit,"p_layer_qbits",self.n_qubits)
        return out_qubits
    
    def forward(self,circuit,weight,in_qubits,out_qubits):
        for i in range(self.n_repeats):
            #mul weight
            if weight[i].sum()<0:
                weight[i] = weight[i]*-1
            idx = 0
            for idx in range(weight[i].flatten().size()[0]):
                if weight[i][idx]==-1:
                    circuit.x(in_qubits[idx])
            # TODO: Potential bug for P-LYR
            #sum and pow2
            circuit.h(out_qubits[i])
            circuit.cz(in_qubits[0],out_qubits[i])
            circuit.x(out_qubits[i])
            circuit.cz(in_qubits[1],out_qubits[i])
            circuit.x(out_qubits[i])
            circuit.h(out_qubits[i])
            circuit.x(out_qubits[i])
            #recover
            for idx in range(weight[i].flatten().size()[0]):
                if weight[i][idx]==-1:
                    circuit.x(in_qubits[idx])
            circuit.barrier(in_qubits,out_qubits)