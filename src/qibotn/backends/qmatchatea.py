"""Implementation of Quantum Matcha Tea backend."""

import re
from dataclasses import dataclass

import numpy as np
import qiskit
import qmatchatea
import qtealeaves
from qibo.backends import NumpyBackend
from qibo.config import raise_error

from qibotn.backends.abstract import QibotnBackend
from qibotn.result import TensorNetworkResult


@dataclass
class QMatchaTeaBackend(QibotnBackend, NumpyBackend):

    def __init__(self):
        super().__init__()

        self.name = "qibotn"
        self.platform = "qmatchatea"

        # Set default configurations
        self.configure_tn_simulation()
        self._setup_backend_specifics()

    def configure_tn_simulation(
        self,
        ansatz: str = "MPS",
        max_bond_dimension: int = 10,
        cut_ratio: float = 1e-9,
        trunc_tracking_mode: str = "C",
        svd_control: str = "A",
        ini_bond_dimension: int = 1,
    ):
        """Configure TN simulation given Quantum Matcha Tea interface.

        Args:
            ansatz (str): tensor network ansatz. It can be  tree tensor network "TTN"
                or Matrix Product States "MPS" (default).
            max_bond_dimension : int, optional Maximum bond dimension of the problem. Default to 10.
            cut_ratio : float, optional
                Cut ratio for singular values. If :math:`\\lambda_n/\\lambda_1 <` cut_ratio then
                :math:`\\lambda_n` is neglected. Default to 1e-9.
            trunc_tracking_mode : str, optional
                Modus for storing truncation, 'M' for maximum, 'C' for
                cumulated (default).
            svd_ctrl : character, optional
                Control for the SVD algorithm. Available:
                - "A" : automatic. Some heuristic is run to choose the best mode for the algorithm.
                - "V" : gesvd. Safe but slow method.
                - "D" : gesdd. Fast iterative method. It might fail. Resort to gesvd if it fails
                - "E" : eigenvalue decomposition method. Faster on GPU. Available only when
                    contracting the singular value to left or right
                - "X" : sparse eigenvalue decomposition method. Used when you reach the maximum
                    bond dimension.
                - "R" : random svd method. Used when you reach the maximum bond dimension.
                Default to 'A'.
            ini_bond_dimension: int, optional
                Initial bond dimension of the simulation. It is used if the initial state is random.
                Default to 1.
        """

        self.convergence_params = qmatchatea.QCConvergenceParameters(
            max_bond_dimension=max_bond_dimension,
            cut_ratio=cut_ratio,
            trunc_tracking_mode=trunc_tracking_mode,
            svd_ctrl=svd_control,
            ini_bond_dimension=ini_bond_dimension,
        )
        self.ansatz = ansatz

    def _setup_backend_specifics(self):
        """Configure quimb backend object."""

        qmatchatea_device = (
            "cpu" if "CPU" in self.device else "gpu" if "GPU" in self.device else None
        )
        qmatchatea_precision = (
            "C"
            if self.precision == "single"
            else "Z" if self.precision == "double" else "A"
        )

        # TODO: once MPI is available for Python, integrate it here
        self.qmatchatea_backend = qmatchatea.QCBackend(
            backend="PY",  # The only alternative is Fortran, but we use Python here
            precision=qmatchatea_precision,
            device=qmatchatea_device,
            ansatz=self.ansatz,
        )

    def execute_circuit(
        self,
        circuit,
        initial_state=None,
        nshots=None,
        prob_type=None,
        return_array=False,
        **prob_kwargs,
    ):
        """Execute a Qibo quantum circuit using tensor network simulation.

        This method returns a ``TensorNetworkResult`` object, which provides:
          - Reconstruction of the system state (if the system size is < 20).
          - Frequencies (if the number of shots is specified).
          - Probabilities computed using various methods.

        The following probability computation methods are available, as implemented
        in Quantum Matcha Tea:
          - **"E" (Even):** Probabilities are computed by evenly descending the probability tree,
            pruning branches (states) with probabilities below a threshold.
          - **"G" (Greedy):** Probabilities are computed by following the most probable states
            in descending order until reaching a given coverage (sum of probabilities).
          - **"U" (Unbiased):** An optimal probability measure that is unbiased and designed
            for best performance. See https://arxiv.org/abs/2401.10330 for details.

        Args:
            circuit: A Qibo circuit to execute.
            initial_state: The initial state of the system (default is the vacuum state
                for tensor network simulations).
            nshots: The number of shots for shot-noise simulation (optional).
            prob_type: The probability computation method. Must be one of:
                - "E" (Even)
                - "G" (Greedy)
                - "U" (Unbiased) [default].
            prob_kwargs: Additional parameters required for probability computation:
                - For "U", requires ``num_samples``.
                - For "E" and "G", requires ``prob_threshold``.

        Returns:
            TensorNetworkResult: An object with methods to reconstruct the state,
            compute probabilities, and generate frequencies.
        """

        # TODO: verify if the QCIO mechanism of matcha is supported by Fortran only
        # as written in the docstrings or by Python too (see ``io_info`` argument of
        # ``qmatchatea.interface.run_simulation`` function)
        if initial_state is not None:
            raise_error(
                NotImplementedError,
                f"Backend {self} currently does not support initial state.",
            )

        if prob_type == None:
            prob_type = "U"
            prob_kwargs = {"num_samples": 500}

        # TODO: check
        circuit = self._qibocirc_to_qiskitcirc(circuit)
        run_qk_params = qmatchatea.preprocessing.qk_transpilation_params(False)

        # Initialize the TNObservable object
        observables = qtealeaves.observables.TNObservables()

        # Shots
        if nshots is not None:
            observables += qtealeaves.observables.TNObsProjective(num_shots=nshots)

        # Probabilities
        observables += qtealeaves.observables.TNObsProbabilities(
            prob_type=prob_type,
            **prob_kwargs,
        )

        # State
        observables += qtealeaves.observables.TNState2File(name="temp", formatting="U")

        results = qmatchatea.run_simulation(
            circ=circuit,
            convergence_parameters=self.convergence_params,
            transpilation_parameters=run_qk_params,
            backend=self.qmatchatea_backend,
            observables=observables,
        )

        if circuit.num_qubits < 20 and return_array:
            statevector = results.statevector
        else:
            statevector = None

        return TensorNetworkResult(
            nqubits=circuit.num_qubits,
            backend=self,
            measures=results.measures,
            measured_probabilities=results.measure_probabilities,
            prob_type="default",
            statevector=statevector,
        )

    def expectation(self, circuit, observable):
        """Compute the expectation value of a Qibo-friendly ``observable`` on
        the Tensor Network constructed from a Qibo ``circuit``.

        This method takes a Qibo-style symbolic Hamiltonian (e.g., `X(0)*Z(1) + 2.0*Y(2)*Z(0)`)
        as the observable, converts it into a Quantum Matcha Tea (qmatchatea) observable
        (using `TNObsTensorProduct` and `TNObsWeightedSum`), and computes its expectation
        value using the provided circuit.

        Args:
            circuit: A Qibo quantum circuit object on which the expectation value
                is computed. The circuit should be compatible with the qmatchatea
                Tensor Network backend.
            observable: The observable whose expectation value we want to compute.
                This must be provided in the symbolic Hamiltonian form supported by Qibo
                (e.g., `X(0)*Y(1)` or `Z(0)*Z(1) + 1.5*Y(2)`).

        Returns:
            qibotn.TensorNetworkResult class, providing methods to retrieve
            probabilities, frequencies and state always according to the chosen
            simulation setup.
        """

        # From Qibo to Qiskit
        circuit = self._qibocirc_to_qiskitcirc(circuit)
        run_qk_params = qmatchatea.preprocessing.qk_transpilation_params(False)

        operators = qmatchatea.QCOperators()
        observables = qtealeaves.observables.TNObservables()
        # Add custom observable
        observables += self._qiboobs_to_qmatchaobs(hamiltonian=observable)

        results = qmatchatea.run_simulation(
            circ=circuit,
            convergence_parameters=self.convergence_params,
            transpilation_parameters=run_qk_params,
            backend=self.qmatchatea_backend,
            observables=observables,
            operators=operators,
        )

        return np.real(results.observables["custom_hamiltonian"])

    def _qibocirc_to_qiskitcirc(self, qibo_circuit) -> qiskit.QuantumCircuit:
        """Convert a Qibo Circuit into a Qiskit Circuit."""
        # Convert the circuit to QASM 2.0 to qiskit
        qasm_circuit = qibo_circuit.to_qasm()
        qiskit_circuit = qiskit.QuantumCircuit.from_qasm_str(qasm_circuit)

        # Transpile the circuit to adapt it to the linear structure of the MPS,
        # with the constraint of having only the gates basis_gates
        qiskit_circuit = qmatchatea.preprocessing.preprocess(
            qiskit_circuit,
            qk_params=qmatchatea.preprocessing.qk_transpilation_params(),
        )
        return qiskit_circuit

    def _qiboobs_to_qmatchaobs(self, hamiltonian, observable_name="custom_hamiltonian"):
        """
        Convert a Qibo SymbolicHamiltonian into a qmatchatea TNObsWeightedSum observable.

        The SymbolicHamiltonian is expected to have a collection of terms, where each term has:
        - `coefficient`: A numeric value.
        - `factors`: A list of factors, each a string such as "X2" or "Z0", representing an operator
                    and the qubit it acts on.

        Args:
            hamiltonian (qibo.SymbolicHamiltonian): The symbolic Hamiltonian containing the terms.
            observable_name (str): The name for the resulting TNObsWeightedSum observable.

        Returns:
            TNObsWeightedSum: An observable suitable for use with qmatchatea.
        """
        coeff_list = []
        tensor_product_obs = None

        # Regex to split an operator factor (e.g., "X2" -> operator "X", qubit 2)
        factor_pattern = re.compile(r"([^\d]+)(\d+)")

        # Iterate over each term in the symbolic Hamiltonian
        for i, term in enumerate(hamiltonian.terms):
            # Store the term's coefficient
            coeff_list.append(term.coefficient)

            operator_names = []
            acting_on_qubits = []

            # Process each factor in the term
            for factor in term.factors:
                # Assume each factor is a string like "Y2" or "Z0"
                match = factor_pattern.match(str(factor))
                if match:
                    operator_name = match.group(1)
                    qubit_index = int(match.group(2))
                    operator_names.append(operator_name)
                    acting_on_qubits.append([qubit_index])
                else:
                    raise ValueError(
                        f"Factor '{str(factor)}' does not match the expected format."
                    )

            # Create a TNObsTensorProduct for this term.
            term_tensor_prod = qtealeaves.observables.TNObsTensorProduct(
                name=f"term_{i}",
                operators=operator_names,
                sites=acting_on_qubits,
            )

            # Combine tensor products from each term
            if tensor_product_obs is None:
                tensor_product_obs = term_tensor_prod
            else:
                tensor_product_obs += term_tensor_prod

        # Combine all terms into a weighted sum observable
        obs_sum = qtealeaves.observables.TNObsWeightedSum(
            name=observable_name,
            tp_operators=tensor_product_obs,
            coeffs=coeff_list,
            use_itpo=False,
        )
        return obs_sum
