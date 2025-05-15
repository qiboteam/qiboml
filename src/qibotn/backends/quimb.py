from collections import Counter

import quimb.tensor as qtn
from qibo.backends import NumpyBackend
from qibo.config import raise_error
from qibo.result import QuantumState

from qibotn.backends.abstract import QibotnBackend
from qibotn.result import TensorNetworkResult

class QuimbBackend(QibotnBackend, NumpyBackend):

    def __init__(self):
        super().__init__()

        self.name = "qibotn"
        self.platform = "quimb"

        self.configure_tn_simulation()
        self.setup_backend_specifics()

    def configure_tn_simulation(
        self,
        ansatz: str = "MPS",
        max_bond_dimension: int = 10,
    ):
        """
        Configure tensor network simulation.

        Args:
            ansatz : str, optional
                The tensor network ansatz to use. Currently, only "MPS" is supported. Default is "MPS".
            max_bond_dimension : int, optional
                The maximum bond dimension for the MPS ansatz. Default is 10.

        Notes:
            - The ansatz determines the tensor network structure used for simulation. Currently, only "MPS" is supported.
            - The `max_bond_dimension` parameter controls the maximum allowed bond dimension for the MPS ansatz.
        """
        self.ansatz = ansatz
        self.max_bond_dimension = max_bond_dimension

    def setup_backend_specifics(self, qimb_backend="numpy"):
        """Setup backend specifics.
        Args:
            qimb_backend: str
                The backend to use for the quimb tensor network simulation.
        """
        self.backend = qimb_backend

    def execute_circuit(
        self,
        circuit,
        initial_state=None,
        nshots=None,
        return_array=False,
        **prob_kwargs,
    ):
        """
        Execute a quantum circuit using the specified tensor network ansatz and initial state.

        Args:
            circuit : QuantumCircuit
                The quantum circuit to be executed.
            initial_state : array-like, optional
                The initial state of the quantum system. Only supported for Matrix Product States (MPS) ansatz.
            nshots : int, optional
                The number of shots for sampling the circuit. If None, no sampling is performed, and the full statevector is used.
            return_array : bool, optional
                If True, returns the statevector as a dense array. Default is False.
            **prob_kwargs : dict, optional
                Additional keyword arguments for probability computation (currently unused).

        Returns:
            TensorNetworkResult
                An object containing the results of the circuit execution, including:
                - nqubits: Number of qubits in the circuit.
                - backend: The backend used for execution.
                - measures: The measurement frequencies if nshots is specified, otherwise None.
                - measured_probabilities: A dictionary of computational basis states and their probabilities.
                - prob_type: The type of probability computation used (currently "default").
                - statevector: The final statevector as a dense array if return_array is True, otherwise None.

        Raises:
            ValueError
                If an initial state is provided but the ansatz is not "MPS".

        Notes:
            - The ansatz determines the tensor network structure used for simulation. Currently, only "MPS" is supported.
            - If `initial_state` is provided, it must be compatible with the MPS ansatz.
            - The `nshots` parameter enables sampling from the circuit's output distribution. If not specified, the full statevector is computed.
        """

        if initial_state is not None and self.ansatz == "MPS":
            initial_state = qtn.tensor_1d.MatrixProductState.from_dense(
                initial_state, 2
            )  # 2 is the physical dimension
        elif initial_state is not None:
            raise_error(
                ValueError, "Initial state not None supported only for MPS ansatz."
            )

        circ_ansatz = (
            qtn.circuit.CircuitMPS if self.ansatz == "MPS" else qtn.circuit.Circuit
        )
        circ_quimb = circ_ansatz.from_openqasm2_str(
            circuit.to_qasm(), psi0=initial_state
        )

        frequencies = Counter(circ_quimb.sample(nshots)) if nshots is not None else None
        main_frequencies = {state: count for state, count in frequencies.most_common(n=100)}
        computational_states = [state for state in main_frequencies.keys()]
        amplitudes = {state: circ_quimb.amplitude(state) for state in computational_states}
        measured_probabilities = {state: abs(amplitude) ** 2 for state, amplitude in amplitudes.items()}
            
        statevector = circ_quimb.to_dense() if return_array else None
        return TensorNetworkResult(
            nqubits=circuit.nqubits,
            backend=self,
            measures=frequencies,
            measured_probabilities=measured_probabilities,
            prob_type="default",
            statevector=statevector,
        )
