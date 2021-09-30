import torch
import torch.nn as nn
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import sys
from .utils import *

################ zhirui on 12-30-2020 ################
# this block is for buding the matrix of variant circuit
######################################################
import math


class VClassicCircuitMatrix:
    def __init__(self, n_qubits):
        # --- parameter definition ---
        self.n_qubits = n_qubits
        # state = state
        # constant matrix
        self.mat_cz = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
        self.mat_identity = torch.tensor([[1, 0], [0, 1]])
        self.mat_swap = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    def get_ry_matrix(self, theta):
        mm = torch.zeros(2, 2, dtype=torch.cdouble)
        mm = set_value(mm, 0, 0, torch.cos(theta / 2))
        mm = set_value(mm, 0, 1, -torch.sin(theta / 2))
        mm = set_value(mm, 1, 0, torch.sin(theta / 2))
        mm = set_value(mm, 1, 1, torch.cos(theta / 2))
        return mm

    def get_rx_matrix(self, theta):
        mm = torch.zeros(2, 2, dtype=torch.cdouble)
        mm = set_value(mm, 0, 0, torch.cos(theta / 2))
        mm = set_value(mm, 0, 1, -1j * torch.sin(theta / 2))
        mm = set_value(mm, 1, 0, -1j * torch.sin(theta / 2))
        mm = set_value(mm, 1, 1, torch.cos(theta / 2))
        return mm

    def get_rz_matrix(self, theta):
        mm = torch.zeros(2, 2, dtype=torch.cdouble)
        mm = set_value(mm, 0, 0, torch.exp(-1j * theta / 2))
        mm = set_value(mm, 1, 1, torch.exp(1j * theta / 2))
        return mm

    def get_ctrl_rx_matrix(self, theta):
        mm = torch.zeros(4, 4, dtype=torch.cdouble)
        mm = set_value(mm, 0, 0, torch.tensor(1, dtype=torch.cdouble))
        mm = set_value(mm, 1, 1, torch.tensor(1, dtype=torch.cdouble))
        mm = set_value(mm, 2, 2, torch.cos(theta / 2))
        mm = set_value(mm, 2, 3, -1j * torch.sin(theta / 2))
        mm = set_value(mm, 3, 2, -1j * torch.sin(theta / 2))
        mm = set_value(mm, 3, 3, torch.cos(theta / 2))
        return mm

    def get_ctrl_ry_matrix(self, theta):
        mm = torch.zeros(4, 4, dtype=torch.cdouble)
        mm = set_value(mm, 0, 0, torch.tensor(1, dtype=torch.cdouble))
        mm = set_value(mm, 1, 1, torch.tensor(1, dtype=torch.cdouble))
        mm = set_value(mm, 2, 2, torch.cos(theta / 2))
        mm = set_value(mm, 2, 3, -torch.sin(theta / 2))
        mm = set_value(mm, 3, 2, torch.sin(theta / 2))
        mm = set_value(mm, 3, 3, torch.cos(theta / 2))
        return mm

    def get_ctrl_rz_matrix(self, theta):
        mm = torch.zeros(4, 4, dtype=torch.cdouble)
        mm = set_value(mm, 0, 0, torch.tensor(1, dtype=torch.cdouble))
        mm = set_value(mm, 1, 1, torch.tensor(1, dtype=torch.cdouble))
        mm = set_value(mm, 2, 2, torch.exp(-1j * theta / 2))
        mm = set_value(mm, 3, 3, torch.exp(1j * theta / 2))
        return mm

    # swap (index) and (index-1)
    def qf_swap_dec(self, state, index):
        # generate the matrix
        temp_mat = torch.ones(1, dtype=torch.cdouble)
        for i in range(0, index - 1):
            temp_mat = torch.kron(self.mat_identity, temp_mat)
        temp_mat = torch.kron(self.mat_swap, temp_mat)
        for i in range(0, self.n_qubits - 1 - index):
            temp_mat = torch.kron(self.mat_identity, temp_mat)
        # change state
        state = torch.mm(temp_mat, state)
        return state

    def qf_ry(self, state, theta, index):

        temp_mat = torch.ones(1, dtype=torch.cdouble)
        if isinstance(index, int):
            for i in range(0, self.n_qubits):
                if i == index:
                    mm = self.get_ry_matrix(theta)
                    temp_mat = torch.kron(mm, temp_mat)
                else:
                    temp_mat = torch.kron(self.mat_identity, temp_mat)
        else:
            for i in range(0, self.n_qubits):
                if i in index:
                    select_theta = torch.index_select(theta, 0, torch.tensor([i]))
                    select_mm = self.get_ry_matrix(select_theta)
                    temp_mat = torch.kron(select_mm, temp_mat)

                else:
                    temp_mat = torch.kron(self.mat_identity, temp_mat)
                    # change state
        state = torch.mm(temp_mat, state)
        return state

    def qf_rx(self, state, theta, index):
        temp_mat = torch.ones(1, dtype=torch.cdouble)
        if isinstance(index, int):
            for i in range(0, self.n_qubits):
                if i == index:
                    mm = self.get_rx_matrix(theta)
                    temp_mat = torch.kron(mm, temp_mat)
                else:
                    temp_mat = torch.kron(self.mat_identity, temp_mat)
        else:
            for i in range(0, self.n_qubits):
                if i in index:
                    select_theta = torch.index_select(theta, 0, torch.tensor([i]))
                    select_mm = self.get_rx_matrix(select_theta)
                    temp_mat = torch.kron(select_mm, temp_mat)

                else:
                    temp_mat = torch.kron(self.mat_identity, temp_mat)
        state = torch.mm(temp_mat, state)
        return state

    def qf_rz(self, state, theta, index):
        temp_mat = torch.ones(1, dtype=torch.cdouble)
        if isinstance(index, int):
            for i in range(0, self.n_qubits):
                if i == index:
                    mm = self.get_rz_matrix(theta)
                    temp_mat = torch.kron(mm, temp_mat)
                else:
                    temp_mat = torch.kron(self.mat_identity, temp_mat)
        else:
            for i in range(0, self.n_qubits):
                if i in index:
                    select_theta = torch.index_select(theta, 0, torch.tensor([i]))
                    select_mm = self.get_rz_matrix(select_theta)
                    temp_mat = torch.kron(select_mm, temp_mat)

                else:
                    temp_mat = torch.kron(self.mat_identity, temp_mat)
        state = torch.mm(temp_mat, state)
        return state

    def qf_cz(self, state, index1, index2):

        # generate the matrix

        # swap the bottom one next to the up one
        for i in range(index2, index1 + 1, -1):
            state = self.qf_swap_dec(state, i)

        # generate cz matrix
        temp_mat = torch.ones(1, dtype=torch.cdouble)
        for i in range(0, index1):
            temp_mat = torch.kron(self.mat_identity, temp_mat)

        temp_mat = torch.kron(self.mat_cz, temp_mat)

        for i in range(0, self.n_qubits - 2 - index1):
            temp_mat = torch.kron(self.mat_identity, temp_mat)

        # change state
        state = torch.mm(temp_mat, state)

        # swap back
        for i in range(index1 + 2, index2 + 1):
            state = self.qf_swap_dec(state, i)

        return state

    def qf_c_r(self, state, theta, name, index_c, index_r):

        if name == 'x':
            self.mat_r = self.get_ctrl_rx_matrix(theta)
        elif name == 'y':
            self.mat_r = self.get_ctrl_ry_matrix(theta)
        elif name == 'z':
            self.mat_r = self.get_ctrl_rz_matrix(theta)

        if index_c > index_r:
            # swap the bottom one next to the up one
            for i in range(index_r + 1, index_c + 1):
                state = self.qf_swap_dec(state, i)

            # generate cz matrix
            temp_mat = torch.ones(1, dtype=torch.cdouble)

            for i in range(0, index_c - 1):
                temp_mat = torch.kron(self.mat_identity, temp_mat)

            temp_mat = torch.kron(self.mat_r, temp_mat)

            for i in range(0, self.n_qubits - 1 - index_c):
                temp_mat = torch.kron(self.mat_identity, temp_mat)

            # change state
            state = torch.mm(temp_mat, state)

            # swap back
            for i in range(index_c, index_r, -1):
                state = self.qf_swap_dec(state, i)
        else:
            # swap the bottom one next to the up one
            for i in range(index_r, index_c + 1, -1):
                state = self.qf_swap_dec(state, i)

            # generate cz matrix
            temp_mat = torch.ones(1, dtype=torch.cdouble)
            for i in range(0, index_c):
                temp_mat = torch.kron(self.mat_identity, temp_mat)

            temp_mat = torch.kron(self.mat_r, temp_mat)

            for i in range(0, self.n_qubits - 2 - index_c):
                temp_mat = torch.kron(self.mat_identity, temp_mat)

            # change state
            state = torch.mm(temp_mat, state)

            # swap back
            for i in range(index_c + 2, index_r + 1):
                state = self.qf_swap_dec(state, i)
        return state

    # the code is similar to the VQuantumCircuit::vqc_10
    def vqc_10(self, state, thetas):

        # head ry part
        state = self.qf_ry(state, thetas[0:self.n_qubits], range(0, self.n_qubits))

        # cz part
        for i in range(self.n_qubits - 1):
            state = self.qf_cz(state, self.n_qubits - 2 - i, self.n_qubits - 1 - i)
        state = self.qf_cz(state, 0, self.n_qubits - 1)

        # tail ry part
        state = self.qf_ry(state, thetas[self.n_qubits:2 * self.n_qubits], range(0, self.n_qubits))
        return state

    def vqc_5(self, state, thetas):
        state = self.qf_rx(state, thetas[0:self.n_qubits], range(0, self.n_qubits))
        state = self.qf_rz(state, thetas[self.n_qubits:2 * self.n_qubits], range(0, self.n_qubits))
        cnt = 0
        for i in range(self.n_qubits - 1, -1, -1):
            for j in range(self.n_qubits - 1, -1, -1):
                if j == i:
                    continue
                else:
                    state = self.qf_c_r(state, thetas[2 * self.n_qubits + cnt], 'z', i, j)
                    cnt = cnt + 1
        state = self.qf_rx(state, thetas[5 * self.n_qubits:6 * self.n_qubits], range(0, self.n_qubits))
        state = self.qf_rz(state, thetas[6 * self.n_qubits:7 * self.n_qubits], range(0, self.n_qubits))
        return state

    def get_parameter_number(self, vqc_name):
        if vqc_name == 'vqc_10':
            return int(2 * self.n_qubits)
        elif vqc_name == 'vqc_5':
            return int(7 * self.n_qubits)


# vqc 必须要显式进行p2a

class VQC_Net(nn.Module):
    def __init__(self, input_num, output_num, vqc_name='vqc_10'):
        super(VQC_Net, self).__init__()

        # init parameter
        self.num_qubit = int(math.log2(input_num))
        self.vcm = VClassicCircuitMatrix(self.num_qubit)
        self.output_num = output_num
        self.vqc_name = vqc_name
        self.theta = Parameter(
            torch.tensor(np.random.randn(self.vcm.get_parameter_number(vqc_name)) * np.pi, dtype=torch.cdouble,
                         requires_grad=True))  # [np.pi/3,np.pi/4,np.pi/3,np.pi/9,np.pi,np.pi/4,np.pi/10,np.pi/2]

        # init  VClassicCircuitMatrix

    def forward(self, x):

        x = x.t().to(torch.cdouble)
        if x.shape[0] != int(math.pow(2, self.num_qubit)):
            print("Not support VQC input size :", x.shape)
            sys.exit(0)

        if self.output_num > int(math.pow(2, self.num_qubit)):
            print("Not support VQC output size!")
            sys.exit(0)

        if self.vqc_name == 'vqc_10':
            x = self.vcm.vqc_10(x, self.theta)
        elif self.vqc_name == 'vqc_5':
            x = self.vcm.vqc_5(x, self.theta)
        else:
            print("Not support VQC name!")
            sys.exit(0)

        if self.output_num <= self.num_qubit:
            x = amp2prop(x)

        x = torch.index_select(x, 0, torch.tensor(range(self.output_num)))

        return x.t().float()
