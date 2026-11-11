from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qweather.quantum.base import BaseQuantumEncoder


class AngleEncoder(BaseQuantumEncoder):
    """
    Direct Angle Feature Map encoding classical variables into qubit rotation angles.
    Maps feature x_i to RY(x_i) rotation on qubit i.
    """

    def __init__(self, num_qubits: int):
        super().__init__(num_qubits=num_qubits, feature_dim=num_qubits)

    def build_circuit(self) -> tuple[QuantumCircuit, ParameterVector]:
        qc = QuantumCircuit(self.num_qubits, name="AngleEncoder")
        x = ParameterVector("x", self.feature_dim)

        for i in range(self.num_qubits):
            qc.ry(x[i], i)

        return qc, x


class EntangledFeatureMap(BaseQuantumEncoder):
    """
    Variational Quantum Feature Map utilizing non-linear entanglement (CNOTs)
    to map complex non-linear feature interactions into quantum Hilbert space.
    """

    def __init__(self, num_qubits: int, reps: int = 1):
        super().__init__(num_qubits=num_qubits, feature_dim=num_qubits)
        self.reps = reps

    def build_circuit(self) -> tuple[QuantumCircuit, ParameterVector]:
        qc = QuantumCircuit(self.num_qubits, name="EntangledFeatureMap")
        x = ParameterVector("x", self.feature_dim)

        for rep in range(self.reps):
            # Rotation Layer
            for i in range(self.num_qubits):
                qc.ry(x[i], i)

            # Entanglement Layer (Circular CNOT chain)
            if self.num_qubits > 1:
                for i in range(self.num_qubits - 1):
                    qc.cx(i, i + 1)
                qc.cx(self.num_qubits - 1, 0)

            qc.barrier()

        return qc, x