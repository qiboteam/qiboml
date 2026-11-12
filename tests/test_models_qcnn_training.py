import pytest

# pytest.skip("Soon to be moved to `qiboml`.", allow_module_level=True)

import random

import numpy as np

from qibo import set_backend
from qiboml.models.qcnn import QuantumCNN


def test_qcnn_training():

    set_backend("numpy")

    # generate 2 random states and labels for pytest
    data = np.zeros([2, 16])
    for i in range(2):
        data_i = np.random.rand(16)
        data[i] = data_i / np.linalg.norm(data_i)
    labels = [[1], [-1]]

    # test qcnn training
    testbias = np.zeros(1)
    testangles = [random.uniform(0, 2 * np.pi) for i in range(21 * 2)]
    init_theta = np.concatenate((testbias, testangles))
    test_qcnn = QuantumCNN('QCNNCOMPLEX',nqubits=4, nlayers=1, nclasses=2, params=init_theta)
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
