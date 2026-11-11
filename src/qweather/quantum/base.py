from abc import ABC, abstractmethod
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


class BaseQuantumEncoder(ABC):
    """
    Abstract Base Class for all Quantum Feature Maps and Encoders in qweather.
    
    Ensures a strict contract for translating classical vectors into Qiskit QuantumCircuits.
    """

    def __init__(self, num_qubits: int, feature_dim: int):
        if num_qubits <= 0:
            raise ValueError(f"num_qubits must be positive, got {num_qubits}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")

        self._num_qubits = num_qubits
        self._feature_dim = feature_dim

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @abstractmethod
    def build_circuit(self) -> tuple[QuantumCircuit, ParameterVector]:
        """
        Constructs and returns the Qiskit QuantumCircuit along with its ParameterVector.

        Returns:
            tuple[QuantumCircuit, ParameterVector]:
                - The initialized parameteric circuit.
                - The ParameterVector corresponding to input classical features.
        """
        pass