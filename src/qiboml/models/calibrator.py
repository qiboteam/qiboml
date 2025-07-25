import datetime
from dataclasses import dataclass, field
from pathlib import Path

from qibo.backends import Backend, NumpyBackend, _check_backend

try:
    from qibocal.auto.history import History
    from qibocal.auto.mode import AUTOCALIBRATION
    from qibocal.auto.runcard import Runcard
    from qibocal.auto.task import Data, Results
    from qibocal.protocols import single_shot_classification
    from qibocal.update import QubitId

    @dataclass
    class Calibrator:

        runcard: Runcard
        """Qibocal runcard with the calibration protocol to be executed."""
        path: Path
        """Folder to dump the protocol's data and results."""
        backend: Backend
        """Qibo backend."""
        trigger_shots: int = 100
        """Number of shots to trigger :meth:`qiboml.models.utils.Calibrator.execute_experiments`"""
        _history: list[History] = field(default_factory=list)
        _counter: int = 0

        @property
        def history(self):
            return self._history

        def __call__(self):
            self._counter += 1
            if self._counter % self.trigger_shots == 0:
                self.execute_experiments()

        def execute_experiments(self):
            """Execute the experiments in the runcard."""
            platform = self.backend.platform
            assert platform is not None, "Invalid None platform"
            output_folder = self.path / datetime.datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S"
            )
            self._history.append(
                self.runcard.run(
                    output=output_folder,
                    platform=platform,
                    mode=AUTOCALIBRATION,
                    update=False,
                )
            )

        def data(self, id: str, execution_number: int):
            """Return the data of the protocol with the specific `id` at its
            `execution_number` execution.
            """
            return self._history[execution_number][id][0].data

        def results(self, id: str, execution_number: int):
            """Return the results of the protocol with the specific `id` at its
            `execution_number` execution.
            """
            return self._history[execution_number][id][0].results

except ImportError:
    Calibrator = None
