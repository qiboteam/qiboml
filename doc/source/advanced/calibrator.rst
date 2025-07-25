Defining a Calibrator
---------------------

In `Qiboml`, the decoder allows for the execution of 
calibration protocols to check the status of the quantum hardware during training. 
To do that, a `Calibrator` object is implemented, to be customized by appending the `Qibocal` routines to be run. For more information about the available calibration routines please check `Qibocal`'s [documentation](https://qibo.science/qibocal/stable/).

This tutorial shows how to specify the calibration protocols and retrieve
their data and results.
First of all, let's define the backend, the transpiler (fundamental to decompose
the circuit's gates into the native ones) and the circuit.

.. code::

        set_backend(backend = "qibolab", platform = "mock")
        transpiler = get_transpiler()
        nqubits = 2
        epochs = 3

        wire_names=[i for i in range(nqubits)]
        vqe_circ = Circuit(2,)
        vqe_circ.add(gates.RX(0, 3*np.pi/4, trainable=True))
        vqe_circ.add(gates.RX(1, np.pi/4, trainable=True))
        vqe_circ.add(gates.CZ(0,1))

In order to check the status of the  calibration, we could be interested, for instance, in
evaluating the readout fidelity and the quality of the single qubit gates. 
This is achieved in `Qibocal` through the classification experiment and the `allxy` routine.

.. code::

        single_shot_action = Action(

            id = "single_shot",
            operation =  "single_shot_classification",
            parameters={"nshots": 100}

        )
        allxy = Action(
            id = "allxy",
            operation =  "allxy",
            parameters={"nshots": 100}

        )

The `id` is a user-chosen name to identify the specific protocol execution, whereas 
the `operation` and `parameters` are the name of the protocol in `Qibocal` and
its parameter, respectively.

Now we can collect our protocols into the `Runcard` object and pass it to the
`Calibrator` object.

.. code::

        runcard = Runcard(
            actions=[single_shot_action, allxy],
            targets = ["0", "1"],
        )
        calibrator = Calibrator(
            runcard=runcard,
            backend = get_backend(),
            path = Path("report_test"),
            trigger_shots = 10,
        )

Finally, we can pass the `Calibrator` to the `Expectation` decoder as an additional argument.
Every `trigger_shots` time the calibration protocols are
executed, and the results are dumped into the `path`.

.. code::

        dec = Expectation(
            nqubits=nqubits,
            nshots=1024,
            density_matrix=False,
            wire_names=wire_names,
            transpiler=transpiler,
            calibrator=calibrator,
        )
        model = QuantumModel(
          circuit_structure=vqe_circ,
          decoding=dec,
          differentiation=PSR()
        )
        optimizer = torch.optim.Adam(
          model.parameters(),
          lr=0.1
        )

        for epoch in range(epochs):
            optimizer.zero_grad()
            cost = model()
            cost.backward()
            optimizer.step()

If we want to access the data and results of the calibration protocols we can
just call the `data` and `results` methods specifying the protocol execution
with its id and progressive number.

.. code::


        data = calibrator.data("allxy", execution_time = 0)
        results = calibrator.results("sgle_shot", execution_time = 0)
