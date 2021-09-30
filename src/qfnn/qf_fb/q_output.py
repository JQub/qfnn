from qiskit.tools.monitor import job_monitor
from qiskit import QuantumRegister,ClassicalRegister
from qiskit import Aer, execute,IBMQ,transpile
from qiskit import BasicAer

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
def fire_ibmq(circuit, shots, Simulation=True, backend_name='ibmq_essex'):
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


def get_unitary(circuit, IBMQ=None):
    if IBMQ == None:
        backend = BasicAer.get_backend('unitary_simulator')
    else:
        provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
        backend = provider.backend.ibmq_vigo

    job = backend.run(transpile(circuit, backend))

    unitary = job.result().get_unitary(circuit, decimals=9)  # Execute the circuit
    return unitary


def add_measure(circuit, qubits, name):
    lenth = len(qubits)
    c_reg = ClassicalRegister(lenth, name)
    circuit.add_register(c_reg)
    for i in range(lenth):
        circuit.measure(qubits[i], c_reg[i])


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
    for k, v in counts.items():
        bits = len(k)
        for i in range(bits):
            if k[bits - 1 - i] == "1":
                if i in mycount.keys():
                    mycount[i] += v
                else:
                    mycount[i] = v
    return mycount, bits