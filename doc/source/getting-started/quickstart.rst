Quick start
===========

In this section, we provide examples on how to use Qibotn to execute tensor network
simulation of quantum circuit. First, we show how to use the Cutensornet and Quimb
backends, while in a second moment we show a complete example of usage of the Quantum
Matcha Tea Backend.

Setting the backend with Cutensornet and Quimb
""""""""""""""""""""""""""""""""""""""""""""""

Among the backends provided by Qibotn, we have cutensornet (using cuQuantum library)
and qutensornet (using Quimb library) for tensor network based simulations.
At present, cutensornet backend works only for GPUs whereas qutensornet for CPUs.
These backend can be set using the following command line.

To use cuQuantum library, cutensornet can be specified as follows::

   qibo.set_backend(
      backend="qibotn", platform="cutensornet", runcard=computation_settings
   )

Similarly, to use Quimb library, qutensornet can be set as follows::

   qibo.set_backend(
       backend="qibotn", platform="qutensornet", runcard=computation_settings
   )

Setting the runcard
"""""""""""""""""""

The basic structure of the runcard is as follows::

   computation_settings = {
       "MPI_enabled": False,
       "MPS_enabled": False,
       "NCCL_enabled": False,
       "expectation_enabled": {
           "pauli_string_pattern": "IXZ",
       },
   }


**MPI_enabled:** Setting this option *True* results in parallel execution of circuit using MPI (Message Passing Interface). At present, only works for cutensornet platform.

**MPS_enabled:** This option is set *True* for Matrix Product State (MPS) based calculations where as general tensor network structure is used for *False* value.

**NCCL_enabled:** This is set *True* for cutensoret interface for further acceleration while using Nvidia Collective Communication Library (NCCL).

**expectation_enabled:** This option is set *True* while calculating expecation value of the circuit. Observable whose expectation value is to be calculated is passed as a string in the dict format as {"pauli_string_pattern": "observable"}. When the option is set *False*, the dense vector state of the circuit is calculated.


Basic example
"""""""""""""

The following is a basic example to execute a two qubit circuit and print the final state in dense vector form using quimb backend::

   import qibo
   from qibo import Circuit, gates

   # Set the runcard
   computation_settings = {
       "MPI_enabled": False,
       "MPS_enabled": False,
       "NCCL_enabled": False,
       "expectation_enabled": False,
   }


   # Set the quimb backend
   qibo.set_backend(
       backend="qibotn", platform="qutensornet", runcard=computation_settings
   )


   # Construct the circuit with two qubits
   c = Circuit(2)

   # Apply Hadamard gates on first and second qubit
   c.add(gates.H(0))
   c.add(gates.H(1))

   # Execute the circuit and obtain the final state
   result = c()

   # Print the final state
   print(result.state())


Using the Quantum Matcha Tea backend
""""""""""""""""""""""""""""""""""""

In the following we show an example of how the Quantum Matcha Tea backend can be
used to execute a quantum circuit::

    # We need Qibo to setup the circuit and the backend
    from qibo import Circuit, gates
    from qibo.models.encodings import ghz_state
    from qibo.backends import construct_backend

    # We need Quantum Matcha Tea to customize the tensor network simulation
    from qmatchatea import QCConvergenceParameters

    # Set the number of qubits
    nqubits = 40

    # Construct a circuit preparing a Quantum Fourier Transform
    circuit = ghz_state(nqubits)

    # Construct the backend
    backend = construct_backend(backend="qibotn", platform="qmatchatea")

    # Customize the low-level backend preferences according to Qibo's formalism
    backend.set_device("/CPU:1")
    backend.set_precision("double")

    # Customize the tensor network simulation itself
    backend.configure_tn_simulation(
        ansatz = "MPS",
        convergence_params = QCConvergenceParameters(max_bond_dimension=50, cut_ratio=1e-6)
    )

    # Execute the tensor network simulation
    outcome = backend.execute_circuit(
        circuit = circuit,
        nshots=1024,
    )

    # Print some results
    print(outcome.probabilities())
    # Should print something like: {'0000000000000000000000000000000000000000': 0.5000000000000001, '1111111111111111111111111111111111111111': 0.5000000000000001}
    print(outcome.frequencies())
    # Should print something like: {'0000000000000000000000000000000000000000': 488, '1111111111111111111111111111111111111111': 536}


By default, the simulator is choosing a specific method to compute the probabilities,
and for further information we recommend the user to refer to our High-Level-API
docstrings: :doc:`/api-reference/qibotn.backends`.
