import qibo
from qibo import hamiltonians, Circuit, gates

from qiboml.differentiation import psr
from qiboml.ansatze import reuploading_circuit

qibo.set_backend("tensorflow")

h = hamiltonians.Z(2)
backend = h.backend

c = reuploading_circuit(nqubits=2, nlayers=2)
nparams = len(c.get_parameters())
print(nparams)
p = backend.tf.Variable([1., 0.4, 2., 0.1, 0.9, 0.2, 0.1, 0.1], dtype=backend.tf.float64)
nepochs = 200

for k in range(nepochs):
     with backend.tf.GradientTape() as tape:
          c.set_parameters(p)
          exp = psr.expectation_on_backend(
               observable=h,
               circuit=c,
               nshots=5000,
               backend="numpy"
          )
     grads = tape.gradient(exp, p)
     p.assign_sub(0.05 * grads)
     if k%10 == 0:
          print(f"Epoch {k+1}/{nepochs}\t Exp: {exp}")




