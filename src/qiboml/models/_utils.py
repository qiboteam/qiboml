from qiboml import ndarray
from qiboml.models.abstract import QuantumCircuitLayer
from qiboml.models.encoding_decoding import QuantumDecodingLayer, QuantumEncodingLayer


def _run_layers(x: ndarray, layers: list[QuantumCircuitLayer], parameters):
    # index = 0
    inputs = x
    parameter_encoding = False
    for layer in layers[:-1]:
        if layer.has_parameters and not issubclass(
            layer.__class__, QuantumEncodingLayer
        ):
            parameter_encoding = True
            # layer.parameters = parameters[index]
            # index += 1
        x = layer.forward(x)
    params = [param for param in inputs] if parameter_encoding else []
    # breakpoint()
    x.set_parameters(params + [par for parameter in parameters for par in parameter])
    return layers[-1].forward(x)
    # return x
