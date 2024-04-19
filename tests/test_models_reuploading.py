from qiboml.models.reuploading.fourier import FourierReuploading
from qiboml.models.reuploading.u3 import ReuploadingU3

NQUBITS = 3
NLAYERS = 2
DATADIM = (3,)


def test_reuloading_u3():
    model = ReuploadingU3(nqubits=NQUBITS, nlayers=NLAYERS, data_dimensionality=DATADIM)
    init_angles = model.circuit.get_parameters()
    model.inject_data((0.4, 0.5, 0.6))
    new_angles = model.circuit.get_parameters()
    assert init_angles != new_angles


def test_reuloading_fourier():
    model = FourierReuploading(
        nqubits=NQUBITS, nlayers=NLAYERS, data_dimensionality=DATADIM
    )
    init_angles = model.circuit.get_parameters()
    model.inject_data((0.4, 0.5, 0.6))
    new_angles = model.circuit.get_parameters()
    assert init_angles != new_angles
