import copy
import numpy as np
import math
import random
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
import matplotlib as mpl
import matplotlib.pyplot as mp
from matplotlib.patches import Circle

EPSILON = 0.1

class AOD_Atom:
    def __init__(self, atom_id, initial_position):
        self.atom_id = atom_id
        self.initial_position = initial_position 
        self.row = None
        self.col = None

    @property
    def position(self):
        """Return the atom's topological position based on its associated row and column."""
        if self.row and self.col:
            return (self.col.position, self.row.position)
        return self.initial_position

    def __repr__(self):
        if self.row == None or self.col == None:
            return f"Atom(id: {self.atom_id}, initial position: {self.initial_position})"
        else:
            return f"Atom(id: {self.atom_id}, position: {self.position})"

class Row:
    def __init__(self, position):
        self._position = position
        self.atom_objects = [] 
        self.moved = 0

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = round(value, 4)

    def __repr__(self):
        return f"[{' '.join(map(str, [atom.atom_id for atom in self.atom_objects]))}] at position {self.position}"

class Column:
    def __init__(self, position):
        self._position = position
        self.atom_objects = []
        self.moved = 0

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = round(value, 4)

    def __repr__(self):
        return f"[{' '.join(map(str, [atom.atom_id for atom in self.atom_objects]))}] at position {self.position}"
    
class AODGrid:
    #grid_type = 0 if SLM, 1 if AOD (allows movement)
    def __init__(self, grid_dims, array_dims):
        self.rows = []
        self.columns = []
        self.atom_by_id = {}
        
        #Below computes initial position for rows cols s.t. they are equidistant from each other
        row_space = array_dims[1] // grid_dims[0] 
        col_space = array_dims[0] // grid_dims[1]  

        #loop to compute row positions
        for row in range(grid_dims[0]):
            y_position = (row_space // 2) + row * row_space
            
            if row == grid_dims[0] - 1:
                y_position = min(y_position, array_dims[1] - (row_space // 2))
            
            self.rows.append(Row(y_position))
 
        #loop to compute col positions
        for col in range(grid_dims[1]):
            x_position = (col_space // 2) + col * col_space

            if col == grid_dims[1] - 1:
                x_position = min(x_position, array_dims[0] - (col_space // 2))

            self.columns.append(Column(x_position))
            
    def add_atom(self, atom):
        """Add an AOD_Atom to the grid and the dictionary."""
        self.atom_by_id[atom.atom_id] = atom
        
    def __repr__(self):
        rows_repr = self.rows
        columns_repr = self.columns
        return f"Rows:\n{rows_repr}\nColumns:\n{columns_repr}"

class NA_Architecture:
    class PositionException(Exception):
        pass

    def __init__(self, grid_dims, array_dims, qubit_topology, radius, qasm_str):
        self.grid_dims = grid_dims 
        self.array_dims = array_dims 
        self.min_sep_dist = 0.49 
        self.slm = qubit_topology
        self.radius = radius
        self.qasm_str = qasm_str
        self.atom_locs = {i: position for i, position in enumerate(qubit_topology)}
        self.aod = None
        self.mobile_qubits = None
        self.graph_count = 0
        self.swap_counts = 0
    
    def execute_frontier(self, frontier):
        pass
    
    def get_qubit_position(self, qubit_ind):
        """Retrieve the qubit's position based on its location (AOD or SLM)."""
        qubit_reference = self.atom_locs[qubit_ind]
        
        if isinstance(qubit_reference, AOD_Atom):
            return qubit_reference.position
        else:
            return qubit_reference 
        
    def get_gates(self, qasm):
        num_qubits = None
        list_of_gates = []
        # For each row in the QASM file
        for row in qasm.splitlines():
            #get number of qubits out of the qreg file
            if "OPENQASM" in row or "include" in row:
                continue
                
            if 'qreg' in row:
                num_qubits = int(row.split('[')[1].split(']')[0])
                continue

            # Get list of all the qubits as ints involved in the operation
            qubits = [int(row.split('[')[q+1].split(']')[0]) for q in range(row.count('['))]

            list_of_gates.append(qubits)
        return list_of_gates
 
    def categorize_by_qubit_id(self, gates):
        result_dict = {}
        for gate in gates:
            for qubit_id in gate:
                if qubit_id not in result_dict:
                    result_dict[qubit_id] = []
                result_dict[qubit_id].append(gate)

        return result_dict


    def check_interference(self, gate1, gate2):
        for qubit in gate1:
            for qubit2 in gate2:
                if np.linalg.norm(np.array(self.get_qubit_position(qubit)) - np.array(self.get_qubit_position(qubit2))) < 2.5 * self.radius:
                    return True
        return False
    
    def process_interference(self, curr_layer, gate_lists):
        index = 0
        while index < len(curr_layer):
            gate_of_interest = curr_layer[index]
            if len(gate_of_interest) == 2:
                # Use a flag to track if the gate_of_interest should be incremented
                increment_gate = True
                for other_index in range(len(curr_layer) - 1, index, -1):
                    other_gate = curr_layer[other_index]
                    if len(other_gate) == 2:
                        if self.check_interference(gate_of_interest, other_gate):
                            for qubit in other_gate:
                                gate_lists[qubit].insert(0, other_gate)
                            curr_layer.pop(other_index)
                            increment_gate = False
                # If no interference was detected for gate_of_interest, move to the next gate
                if increment_gate:
                    index += 1
            else:
                index += 1
    

    def compile_circuit(self):
        gate_lists = self.categorize_by_qubit_id(self.get_gates(self.qasm_str))
        all_layers = []
        num_qubits = len(self.atom_locs.keys())
        swap_trap_count = 0
        execute_count = 0
        while any(gate_lists[qubit_id] for qubit_id in gate_lists):
            curr_layer = []
            qubit_list = list(gate_lists.keys())
            processed_qubits = set()
            gates_to_pop = []

            for qubit_id in qubit_list:
                if qubit_id in processed_qubits:
                    continue  # Skip this iteration if qubit_id is already processed

                if gate_lists[qubit_id]:
                    if len(gate_lists[qubit_id][0]) == 1:
                        gates_to_pop.append(qubit_id)

                    elif len(gate_lists[qubit_id][0]) == 2:
                        other_qubit_id = gate_lists[qubit_id][0][0] if gate_lists[qubit_id][0][0] != qubit_id else gate_lists[qubit_id][0][1]

                        if gate_lists[other_qubit_id] and gate_lists[qubit_id][0] == gate_lists[other_qubit_id][0]:
                            gates_to_pop.append(qubit_id)
                            gates_to_pop.append(other_qubit_id)

                            processed_qubits.add(qubit_id)
                            processed_qubits.add(other_qubit_id)

            added_to_curr_layer = set()

            for qubit_id in gates_to_pop:
                gate = gate_lists[qubit_id].pop(0)

                # If the gate hasn't been added to curr_layer yet, append it
                if tuple(gate) not in added_to_curr_layer:
                    curr_layer.append(gate)
                    added_to_curr_layer.add(tuple(gate))

            # Shuffle the list of keys randomly to ensure interfering gates not biased towards high/low qubit indices
            keys_list = list(gate_lists.keys())
            random.shuffle(curr_layer)

            # Check for interference. Return gates that interfere back to gate_lists.
            self.process_interference(curr_layer, gate_lists)

            self.execute_frontier(curr_layer)
            
            #For each CZ, if qubits are out of range, find distance between them in terms of number of Rydberg-radius-units.
            #So if one qubit is over 1 unit away, 1 swap there+1 back needed; if over 2 units, 2 there + 2 back, etc.
            #Note: Each SWAP takes 3 CZ gates.
            #NOTE: THESE SWAPS HAPPEN OUTSIDE OF LAYER EXECUTION. GREATEST # SWAPS PER LAYER IN TIME 
            #(so if layer has 2,1,3 swaps, add time to execute 3 SWAPs, which would be 18 CZ gates total (since 3+3 SWAPS needed)).
#             for gate in curr_layer:
#                 if len(gate) == 2:
#                     qubit = gate[0]
#                     qubit2 = gate[1]
#                     q_dist = np.linalg.norm(np.array(self.get_qubit_position(qubit)) - np.array(self.get_qubit_position(qubit2)))
#                     #If dist between qubits > radius:
#                     if q_dist > self.radius:
#                         self.swap_counts += math.floor(q_dist/self.radius) * 2 #still x2 here since these gates are simply out of bounds for whatever reason.

            all_layers.append(curr_layer)
            execute_count += len(curr_layer)

        #Get gate counts
        cz_count = 0
        u_count = 0
        for f in all_layers:
            for g in f:
                if len(g) == 1:
                    u_count += 1
                else:
                    cz_count += 1
        return all_layers, self.swap_counts, cz_count+self.swap_counts*3, u_count #want to include move_dist
