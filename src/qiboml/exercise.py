import qibo
from qibo import hamiltonians, Circuit, gates

from differentiation import psr

qibo.set_backend("tensorflow")

h = hamiltonians.Z(1)
backend = h.backend

c = Circuit(1)
c.add(gates.RZ(0, 0))
c.add(gates.RY(0, 0))
c.add(gates.M(0))

p = backend.tf.Variable([0.4,0.6])

with backend.tf.GradientTape() as tape:
     c.set_parameters(p)
     exp = psr.expectation_with_tf(
          observable=h,
          circuit=c,
          nshots=1000
     )

grads = tape.gradient(exp, p)
print("Exp:", exp)
print("Grads:", grads)

