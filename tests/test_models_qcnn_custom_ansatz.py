import pytest

# pytest.skip("Soon to be moved to `qiboml`.", allow_module_level=True)

import numpy as np

from qibo import gates, set_backend
from qibo.models import Circuit
#from qibo.models.qcnn import QuantumCNN
from qiboml.models.qcnn import QuantumCNN


def test_two_qubit_ansatz():
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.RX(0, 0))
    circuit.add(gates.CNOT(1, 0))
    QuantumCNN('QCNNCOMPLEX',4, 2, 2, twoqubitansatz=circuit)


def test_two_qubit_ansatz_training():
    # test qibojit case (copy initial state as quick-fix for in-place update)
    # set_backend("qibojit")
    set_backend("numpy")

    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.RX(0, 0))
    circuit.add(gates.CNOT(1, 0))
    test_qcnn = QuantumCNN('QCNNCOMPLEX',4, 2, 2, twoqubitansatz=circuit)

    data = np.zeros([2, 16])
    for i in range(2):
        data_i = np.random.rand(16)
        data[i] = data_i / np.linalg.norm(data_i)
    labels = [[1], [-1]]

    totalNParams = test_qcnn.nparams_layer * 2
    init_theta = [
        0 for i in range(totalNParams + 1)
    ]  # totalNParams+1 to account for bias parameter.

    test_qcnn.minimize(
        init_theta, data=data, labels=labels, nshots=10000, method="Powell"
    )

    # test Predictions function
    predictions = []
    for n in range(len(data)):
        predictions.append(test_qcnn.predict(data[n], nshots=10000)[0])

    # test Accuracy function
    predictions.append(1)
    labels = np.array([[1], [-1], [1]])
    test_qcnn.Accuracy(labels, predictions)
