
import torch
import os
import sys
import shutil
from .lib_bn import *
from .lib_qf import *

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.tar')
        shutil.copyfile(filename, bestname)

def qf_sum(n_qubits):
    sum_mat = []
    flag = "0"+str(n_qubits)+"b"
    for i in range(0,int(math.pow(2,n_qubits))):
        bit_str = format(i,flag)
        row = []
        for c in bit_str:
            row.append(float(c))
        sum_mat.append(row)
    return sum_mat
 
def set_value(mm,col,row,val):
    index = (torch.LongTensor([col]),torch.LongTensor([row]))#生成索引
    mm = mm.index_put(index ,val)
    return mm   

def tensor_squire(x):
    return torch.pow(x,2)

def tensor_sqrt(x):
    return torch.sqrt(x+1e-6)

def amp2prop(state):
    state = state.double()
    n_qubits = int(math.log2(state.shape[0]))
    sum_mat = torch.tensor(qf_sum(n_qubits),dtype=torch.float64)
    sum_mat = sum_mat.t()
    state = torch.mm(sum_mat,state)
    return state

class Prop2amp():
    def __call__(self, state):
        state = state.double().t()
        n_qubits = state.shape[0]
        mstate = torch.ones(int(math.pow(2,n_qubits)),state.shape[1], dtype=torch.float64)
        sum_mat = torch.tensor(qf_sum(n_qubits),dtype=torch.float64)
        for i in range(sum_mat.shape[0]):
            for j in range(sum_mat.shape[1]):
                if int(sum_mat[i][j]) == 0:
                    val = torch.mul(torch.index_select(mstate,0,torch.tensor([i])),1-torch.index_select(state,0,torch.tensor([j]))).squeeze()
                    mstate = set_value(mstate,i,range(state.shape[1]),val)
                elif int(sum_mat[i][j]) == 1: 
                    val = torch.mul(torch.index_select(mstate,0,torch.tensor([i])),torch.index_select(state,0,torch.tensor([j]))).squeeze()
                    mstate = set_value(mstate,i,range(state.shape[1]),val)
        mstate = tensor_sqrt(mstate)
        return mstate.t()
