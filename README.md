# Qibotn

The tensor network translation module for Qibo to support large-scale simulation of quantum circuits and acceleration.

## Supported Computation

Tensor Network Types:

- Tensornet (TN)
- Matrix Product States (MPS)

Tensor Network contractions to:

- dense vectors
- expecation values of given Pauli string

The supported HPC configurations are:

- single-node CPU
- single-node GPU or GPUs
- multi-node multi-GPU with Message Passing Interface (MPI)
- multi-node multi-GPU with NVIDIA Collective Communications Library (NCCL)

Currently, the supported tensor network libraries are:

- [cuQuantum](https://github.com/NVIDIA/cuQuantum), an NVIDIA SDK of optimized libraries and tools for accelerating quantum computing workflows.
- [quimb](https://quimb.readthedocs.io/en/latest/), an easy but fast python library for ‘quantum information many-body’ calculations, focusing primarily on tensor networks.

## Installation

To get started:

```sh
pip install qibotn
```

to install the tools and dependencies. A few extras are provided, check `pyproject.toml` in
case you need them.

<!-- TODO: describe extras, after Poetry adoption and its groups -->

## Contribute

To contribute, please install using poetry:

```sh
git clone https://github.com/qiboteam/qibotn.git
cd qibotn
poetry install
```

## Sample Codes

### Single-Node Example

The code below shows an example of how to activate the Cuquantum TensorNetwork backend of Qibo.

```py
import numpy as np
from qibo import Circuit, gates
import qibo

# Below shows how to set the computation_settings
# Note that for MPS_enabled and expectation_enabled parameters the accepted inputs are boolean or a dictionary with the format shown below.
# If computation_settings is not specified, the default setting is used in which all booleans will be False.
# This will trigger the dense vector computation of the tensornet.

computation_settings = {
    "MPI_enabled": False,
    "MPS_enabled": {
        "qr_method": False,
        "svd_method": {
            "partition": "UV",
            "abs_cutoff": 1e-12,
        },
    },
    "NCCL_enabled": False,
    "expectation_enabled": False,
}


qibo.set_backend(
    backend="qibotn", platform="cutensornet", runcard=computation_settings
)  # cuQuantum
# qibo.set_backend(backend="qibotn", platform="qutensornet", runcard=computation_settings) #quimb


# Construct the circuit
c = Circuit(2)
# Add some gates
c.add(gates.H(0))
c.add(gates.H(1))

# Execute the circuit and obtain the final state
result = c()

print(result.state())
```

Other examples of setting the computation_settings

```py
# Expectation computation with specific Pauli String pattern
computation_settings = {
    "MPI_enabled": False,
    "MPS_enabled": False,
    "NCCL_enabled": False,
    "expectation_enabled": {
        "pauli_string_pattern": "IXZ",
    },
}

# Dense vector computation using multi node through MPI
computation_settings = {
    "MPI_enabled": True,
    "MPS_enabled": False,
    "NCCL_enabled": False,
    "expectation_enabled": False,
}
```

### Multi-Node Example

Multi-node is enabled by setting either the MPI or NCCL enabled flag to True in the computation settings. Below shows the script to launch on 2 nodes with 2 GPUs each. $node_list contains the IP of the nodes assigned.

```sh
mpirun -n 4 -hostfile $node_list python test.py
```
