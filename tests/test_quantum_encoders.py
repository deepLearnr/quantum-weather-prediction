import pytest
from qiskit.circuit import ParameterVector
from qweather.quantum import AngleEncoder, EntangledFeatureMap


def test_angle_encoder_circuit_properties():
    num_qubits = 4
    encoder = AngleEncoder(num_qubits=num_qubits)
    circuit, params = encoder.build_circuit()

    assert circuit.num_qubits == num_qubits
    assert len(params) == num_qubits
    assert isinstance(params, ParameterVector)


def test_entangled_feature_map_structure():
    num_qubits = 4
    encoder = EntangledFeatureMap(num_qubits=num_qubits, reps=2)
    circuit, params = encoder.build_circuit()

    assert circuit.num_qubits == num_qubits
    # Verify CNOT gates exist in circuit
    gate_names = [instruction.operation.name for instruction in circuit.data]
    assert "cx" in gate_names
    assert "ry" in gate_names