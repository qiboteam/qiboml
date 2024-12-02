from dataclasses import dataclass
from abc import ABC, abstractmethod
from qibo import Circuit, gates
from numpy import ndarray

import tensorflow as tf
import keras
import numpy as np


def circuit():
    c = Circuit(2)
    c.add(gates.RX(0, theta=0))
    return c


@tf.custom_gradient
def custom_operation_1(c, x):
    print(f"Parametro: {x}")
    c.set_parameters([x])
    z = c().probabilities()
    print(f"Output: {z}")

    def grad_fn(upstream):
        dz_dx = 6
        print(f"Derivata: {upstream * dz_dx}")
        return upstream * dz_dx

    return z, grad_fn


@tf.custom_gradient
def custom_operation_3(x):
    print(f"Parametro: {x}")

    def forward():
        c = circuit()
        c.set_parameters([x])
        z = c().probabilities()
        return z

    # z = tf.py_function(func=forward, inp=[c], Tout=tf.float32)
    z = forward()
    print(f"Output: {z}")

    def grad_fn(upstream):
        dz_dx = 6
        print(f"Derivata: {upstream * dz_dx}")
        return upstream * dz_dx

    return z, grad_fn


@tf.custom_gradient
def custom_operation_2(x):
    print(f"Parametro: {x}")
    c = Circuit(2)
    breakpoint()
    c.add(gates.RX(0, theta=0))
    c.set_parameters([x])
    z = c().probabilities()
    print(f"Output: {z}")

    def grad_fn(upstream):
        dz_dx = 6
        print(f"Derivata: {upstream * dz_dx}")
        return upstream * dz_dx

    return z, grad_fn


# ===================
# ESPERIMENTO 1
# ===================
# In customgrad1 ho verificato che
# la funzione @tf.custom_gradient funzionasse
# dentro ad una classe e con la chiamata __call__ della
# classe

"""

class CustomGrad1:

    def __init__(self):
        pass

    def __call__(self, x):

        @tf.custom_gradient
        def custom_op(x):

            def grad_fn(upstream):
                # upstream sono tutte le derivate dopo
                # "2" è la derivata di questa operazione
                return upstream * 2

            return x * 2, grad_fn

        return custom_op(x)


x = tf.constant([[2.0]])  # Batch di dimensione 1x1
y = tf.constant([[1.0]])  # Batch di dimensione 1x1
params = tf.constant(3.0, dtype=tf.float32)


obj = CustomGrad1()

# Definizione di un input tensor
x = tf.Variable(3.0)  # Un tensore variabile per il calcolo del gradiente

# Calcolo del gradiente con upstream diverso da 0
with tf.GradientTape() as tape:
    y = obj(x)  # Operazione personalizzata (y = x * 2)
    z = 3 * y + 5  # Una funzione che include y (z = 3 * y + 5)

# Calcolo del gradiente di z rispetto a x
# dz/dx = dz/dy dy/dx = 3 * 2 = 6
grad = tape.gradient(z, x)

# Output
print("Output y:", y.numpy())  # Valore di y (x * 2)
print("Output z:", z.numpy())  # Valore di z (3 * y + 5)
print("Gradiente:", grad.numpy())  # Gradiente finale

"""

# ===================
# ESPERIMENTO 2
# ===================
# In customgrad2 ho definito tre funzioni:
# y = f(x) = (y1, y2) = (x^2, x^3)
# z = g(y) = g(f(x))
# z1 = 3 * y1 + 5
# z2 = 2 * y2 + 6
# z3 = 3 * y1 + 5 * y2
# Dato che la funzione f prende in ingresso un solo valore x
# ovvero uno scalare, il custom gradient dovrà essere uno scalare del tipo:
# dz/dx = dz/df * df/dx
# Poichè z dipende sia da y1, che da y2 la derivata rispetto a x sarà:
# z = g(y) = g(y1(x), y2(x)) = dg/dy1 * dy1/dx + dg/dy2 * dy2/dx
# Per questa ragione l'output di grad_fn è la somma: grad_y1 + grad_y2


class CustomGrad2:

    def __init__(self):
        pass

    def __call__(self, x):
        @tf.custom_gradient
        def custom_op(x):

            y1 = x**2
            y2 = x**3

            def grad_fn(upstream_y1, upstream_y2):
                print(f"Upstream {upstream_y1}, {upstream_y2}")
                grad_y1 = upstream_y1 * 2 * x
                grad_y2 = upstream_y2 * 3 * x**2

                return grad_y1 + grad_y2

            return (y1, y2), grad_fn

        return custom_op(x)


"""

# Creiamo un oggetto della classe
obj = CustomGrad2()
x = tf.Variable(3.0)

with tf.GradientTape(persistent=True) as tape:
    # Usare obj(x) per calcolare y1 e y2
    y1, y2 = obj(x)
    z1 = 3 * y1 + 5
    z2 = 2 * y2 + 6
    z3 = 3 * y1 + 5 * y2

# Calcolo dei gradienti di z1 e z2 rispetto a x
grad1 = tape.gradient(z1, x)
grad2 = tape.gradient(z2, x)
grad3 = tape.gradient(z3, x)

print("Output y1:", y1.numpy())
print("Output z1:", z1.numpy())
print("Output y2:", y2.numpy())
print("Output z2:", z2.numpy())
print("Output z3:", z3.numpy())
print("Gradiente z1 rispetto a x1:", grad1.numpy())
print("Gradiente z2 rispetto a x2:", grad2.numpy())
print("Gradiente z3 rispetto a x3:", grad3.numpy())
"""

# ===================
# ESPERIMENTO 3
# ===================
# Adesso voglio che z sia una funzione vettoriale, dunque:
# z = (z1, z2) = (3*y1+5, 3*y2+5)
# y = f(x) = (y1, y2) = (x^2, x^3)

"""
class CustomGrad3:

    def __init__(self):
        pass

    def __call__(self, x):
        @tf.custom_gradient
        def custom_op(x):

            y1 = x**2
            y2 = x**3

            def grad_fn(upstream_y1, upstream_y2):
                print(f"Upstream {upstream_y1}, {upstream_y2}")
                grad_y1 = upstream_y1 * 2 * x
                grad_y2 = upstream_y2 * 3 * x**2

                return grad_y1 + grad_y2

            return (y1, y2), grad_fn

        return custom_op(x)


def fz(y1, y2):
    z1 = 3 * y1 + 4 * y2
    z2 = 1 * y1 + 2 * y2
    return z1, z2


# Creiamo un oggetto della classe
obj = CustomGrad3()
x = tf.Variable(2.0)

with tf.GradientTape(persistent=True) as tape:
    # Usare obj(x) per calcolare y1 e y2
    y1, y2 = obj(x)
    z1 = 3 * y1 + 4 * y2
    z2 = 1 * y1 + 2 * y2
    z = fz(y1, y2)
    print(f"Z {z}")

# Calcolo dei gradienti di z1 e z2 rispetto a x
grad1 = tape.gradient(z[0], x)
grad2 = tape.gradient(z[1], x)

print("Gradiente z[0] rispetto a x:", grad1)
print("Gradiente z[1] rispetto a x:", grad2)

"""


# ===================
# ESPERIMENTO 4
# ===================
# Adesso voglio che z sia una funzione vettoriale e che anche x sia un vettore:
# z = (z1, z2) = (3*y1+5, 3*y2+5)
# y = f(x) = f(x1, x2) = (y1, y2, y3) = (x1+x2, x1^2+x2^2, x1^3+x2^3)

"""

class CustomGrad4:

    def __init__(self):
        pass

    def __call__(self, x1, x2):
        @tf.custom_gradient
        def custom_op(x1, x2):
            y1 = x1 + x2
            y2 = x1**2 + x2**2
            y3 = x1**3 + x2**3

            def grad_fn(upstream_y1, upstream_y2, upstream_y3):
                print(f"Gradienti {upstream_y1}, {upstream_y2}, {upstream_y3}")
                grad_y11 = upstream_y1
                grad_y12 = upstream_y1

                grad_y21 = upstream_y2 * 2 * x1
                grad_y22 = upstream_y2 * 2 * x2

                grad_y31 = upstream_y3 * 3 * x1**2
                grad_y32 = upstream_y3 * 3 * x2**2

                return grad_y11 + grad_y21 + grad_y31, grad_y12 + grad_y22 + grad_y32

            # Ritorna gli output e la funzione gradiente
            return (y1, y2, y3), grad_fn

        return custom_op(x1, x2)


# Creiamo un oggetto della classe

obj = CustomGrad4()
x1 = tf.Variable(1.0)
x2 = tf.Variable(2.0)

with tf.GradientTape(persistent=True) as tape:
    y1, y2, y3 = obj(x1, x2)
    z1 = y1 + y2 + y3
    z2 = 2 * y1 + 3 * y2 + 4 * y3
    z3 = 5 * y1 + 6 * y2 + 7 * y3

# Calcolo dei gradienti di z1 e z2 rispetto a x
grad11 = tape.gradient(z1, x1)
grad12 = tape.gradient(z1, x2)

grad21 = tape.gradient(z2, x1)
grad22 = tape.gradient(z2, x2)

grad31 = tape.gradient(z3, x1)
grad32 = tape.gradient(z3, x2)

# Output
print("Gradiente z1 rispetto a x1:", grad11.numpy())
print("Gradiente z2 rispetto a x1:", grad21.numpy())
print("Gradiente z3 rispetto a x1:", grad31.numpy())

print("Gradiente z1 rispetto a x2:", grad12.numpy())
print("Gradiente z2 rispetto a x2:", grad22.numpy())
print("Gradiente z3 rispetto a x2:", grad32.numpy())


"""

# ===================
# ESPERIMENTO 5
# ===================
# Adesso voglio che z sia una funzione vettoriale e che anche x sia un vettore e non due oggetti
# diversi x1, x2:
# z = (z1, z2) = (3*y1+5, 3*y2+5)
# y = f(x) = f(x1, x2) = (y1, y2, y3) = (x1+x2, x1^2+x2^2, x1^3+x2^3)

# Non capisco come mai ma se uso rotazioni RZ
# ottengo errore
'''

class CustomGrad5:

    def __init__(self):
        pass

    def __call__(self, x, w):
        @tf.custom_gradient
        def custom_op(x, w):
            y1 = w * x**2
            y2 = w * x**3

            def grad_fn(upstream_y1, upstream_y2):
                """
                upstream_y1: gradiente che arriva dall'output y1
                upstream_y2: gradiente che arriva dall'output y2
                """
                print(f"Upstream y1: {upstream_y1}")
                print(f"Upstream y2: {upstream_y2}")

                # Gradiente rispetto a x
                grad_x = upstream_y1 * 2 * x * w + upstream_y2 * 3 * x**2 * w

                # Gradiente rispetto a w
                grad_w = upstream_y1 * x**2 + upstream_y2 * x**3

                return grad_x, grad_w

            return (y1, y2), grad_fn

        return custom_op(x, w)


# Istanzia la classe e prepara i tensori
obj = CustomGrad5()
x = tf.constant([[0.0, 1.0]], dtype=tf.float32)
w = tf.constant([[2.0, 1.0]], dtype=tf.float32)

with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    tape.watch(w)
    y1, y2 = obj(x, w)  # Applica l'operazione personalizzata

# Calcola i gradienti
grad_x = tape.gradient([y1, y2], x)  # Gradiente rispetto a x
grad_w = tape.gradient([y1, y2], w)  # Gradiente rispetto a w

# Output
print(f"Output y1: {y1.numpy()}")
print(f"Output y2: {y2.numpy()}")
print(f"Gradienti rispetto a x: {grad_x.numpy()}")
print(f"Gradienti rispetto a w: {grad_w.numpy()}")

'''

'''
class CustomGrad6:

    def __init__(self):
        pass

    def __call__(self, x, w):
        @tf.custom_gradient
        def custom_op(x, w):
            # Definiamo una nuova operazione per y
            y = tf.concat(
                [w * tf.sin(x), w * tf.exp(-x)], axis=-1
            )  # Operazione personalizzata

            def grad_fn(upstream):
                """
                upstream: gradiente che arriva da y (shape (1, 4))
                """
                print(f"Upstream gradient: {upstream}")

                # Dividiamo il gradiente upstream in due parti corrispondenti a ciascuna operazione
                upstream1, upstream2 = tf.split(upstream, num_or_size_splits=2, axis=-1)

                # Gradiente rispetto a x
                grad_x1 = upstream1 * w * tf.cos(x)  # Derivata di sin(x)
                grad_x2 = -upstream2 * w * tf.exp(-x)  # Derivata di exp(-x)
                grad_x = grad_x1 + grad_x2

                # Gradiente rispetto a w
                grad_w1 = upstream1 * tf.sin(x)  # Derivata rispetto a w di w * sin(x)
                grad_w2 = upstream2 * tf.exp(-x)  # Derivata rispetto a w di w * exp(-x)
                grad_w = grad_w1 + grad_w2

                return grad_x, grad_w

            return y, grad_fn

        return custom_op(x, w)


# Istanza della classe e tensori di input
obj = CustomGrad6()
x = tf.constant([[0.0, 1.0]], dtype=tf.float32)
w = tf.constant([[2.0, 1.0]], dtype=tf.float32)

with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    tape.watch(w)
    y = obj(x, w)  # Applica l'operazione personalizzata

# Calcola i gradienti
grad_x = tape.gradient(y, x)  # Gradiente rispetto a x
grad_w = tape.gradient(y, w)  # Gradiente rispetto a w

# Output
print(f"Output y: {y.numpy()}")
print(f"Gradiente rispetto a x: {grad_x.numpy()}")
print(f"Gradiente rispetto a w: {grad_w.numpy()}")
'''

'''

class CustomGradGeneral:

    def __init__(self, operation):
        self.operation = operation

    def __call__(self, x, w):
        @tf.custom_gradient
        def custom_op(x, w):
            y = self.operation(x, w)

            def grad_fn(upstream):
                """
                upstream: gradiente che arriva da y, ha la stessa shape di y.
                """
                print(f"Upstream gradient: {upstream.numpy()}")

                # Gradiente rispetto a x
                grad_x = tf.gradients(ys=y, xs=x, grad_ys=upstream)[0]
                # Gradiente rispetto a w
                grad_w = tf.gradients(ys=y, xs=w, grad_ys=upstream)[0]

                return grad_x, grad_w

            return y, grad_fn

        return custom_op(x, w)


# Esempio di operazione: una combinazione di sin(x) e exp(-x)
def example_operation(x, w):
    # Operazione dinamica per y
    return tf.concat([w * tf.sin(x), w * tf.exp(-x)], axis=-1)


# Creazione di un oggetto della classe
obj = CustomGradGeneral(operation=example_operation)

# Esempio di input con forma arbitraria
x = tf.constant([[0.0, 1.0], [2.0, 3.0]], dtype=tf.float32)  # Forma (2, 2)
w = tf.constant([[2.0, 1.0], [0.5, 0.2]], dtype=tf.float32)  # Stessa forma di x

# Calcolo con il GradientTape
with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    tape.watch(w)
    y = obj(x, w)  # Applica l'operazione personalizzata

# Calcola i gradienti rispetto a x e w
grad_x = tape.gradient(y, x)
grad_w = tape.gradient(y, w)

# Output
print(f"Input x: {x.numpy()}")
print(f"Input w: {w.numpy()}")
print(f"Output y: {y.numpy()}")
print(f"Gradiente rispetto a x: {grad_x.numpy()}")
print(f"Gradiente rispetto a w: {grad_w.numpy()}")

'''


# Classe Base per l'encoding quantistico
@dataclass
class QuantumEncoding(ABC):
    nqubits: int
    qubits: list[int] = None
    _circuit: Circuit = None

    def __post_init__(self):
        if self.qubits is None:
            self.qubits = list(range(self.nqubits))
        self._circuit = Circuit(self.nqubits)

    @abstractmethod
    def __call__(self, x: tf.Tensor) -> Circuit:
        pass

    @property
    def circuit(self):
        return self._circuit


@dataclass
class BinaryEncoding(QuantumEncoding):
    def __call__(self, x: tf.Tensor) -> Circuit:
        if x.shape[-1] != len(self.qubits):
            raise RuntimeError(
                f"Invalid input dimension {x.shape[-1]}, but the allocated qubits are {self.qubits}."
            )
        circuit = self.circuit.copy()

        for i, q in enumerate(self.qubits):
            if x[0][i] == 1:
                circuit.add(gates.X(q))

        return circuit


# Funzioni Mock per encoding e decoding
def encoding(x):
    c = Circuit(2)
    c.add(gates.RX(0, theta=x[0]))
    c.add(gates.RY(1, theta=x[1]))
    c.add(gates.RX(0, theta=x[2]))
    c.add(gates.RY(1, theta=x[3]))
    return c


def circuit():
    c = Circuit(3)
    c.add(gates.RY(0, theta=0.2))
    c.add(gates.RY(1, theta=0.3))
    return c


def decoding(c):
    return c.probabilities()


class CustomGrad6:
    def __init__(self):
        self.encoding = BinaryEncoding(3, [0, 1, 2])

    def __call__(self, x):
        @tf.custom_gradient
        def custom_op(x):
            c = self.encoding(x)
            c = c.copy()
            c += circuit()
            result = tf.constant(decoding(c()))

            # breakpoint()

            def grad_fn(upstream):
                print(f"Upstream {upstream}")
                grad_x = upstream
                return grad_x

            return result, grad_fn

        return custom_op(x)


obj = CustomGrad6()
# x must be a float
x = tf.constant([[0, 1, 0]], dtype=tf.float32)

with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    result = obj(x)

grad = tape.gradient(result, x)

# Output
print("Output:", result.numpy())
print("Gradiente:", grad)


"""
@dataclass(eq=False)
class CustomGrad(keras.Model):
    encoding: None
    circuit: None

    def __post_init__(self):
        super().__init__()

    def call(self, x):

        @tf.custom_gradient
        def custom_gradient(x):
            def forward(x):
                complete_circuit = self.encoding(x) + self.circuit()
                z = tf.convert_to_tensor(complete_circuit().probabilities())
                breakpoint()
                return z

            def grad_fn(upstream):
                dz_dx = 6
                return upstream * dz_dx

            return forward(x), grad_fn

        return custom_gradient(x)


# model = CustomGrad(encoding, circuit)
# optimizer = keras.optimizers.Adam()
# loss_f = keras.losses.MeanSquaredError()
# model.compile(loss=loss_f, optimizer=optimizer)
# model.fit(
#    x,
#    y,
#    epochs=5,
# )

"""
