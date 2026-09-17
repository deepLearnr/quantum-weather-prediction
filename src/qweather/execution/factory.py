from typing import Any
from qiskit_aer import AerSimulator
from qweather.execution.config import ComputeConfig, load_config

class BackendFactory:
    """Factory responsible for provisioning execution backends based on 
    the validated configuration, completely decoupled from Layers 1-5.
    """

    @staticmethod
    def create_backend(config: ComputeConfig | None = None) -> Any:
        """Provisions and returns the execution backend matching the config target."""
        if config is None:
            config = load_config().compute

        target = config.target

        if target == "local_aer":
            # Instantiate the local Qiskit Aer simulator
            # Default shots can be tracked or set depending on execution wrappers
            return AerSimulator()

        elif target == "cloud_gpu":
            raise NotImplementedError(
                "The 'cloud_gpu' backend target is designated for larger-scale "
                "remote training and is not yet implemented."
            )

        elif target == "ibm_qpu":
            raise NotImplementedError(
                f"The 'ibm_qpu' backend target (configured for '{config.ibm_quantum.backend_name}') "
                "is reserved for later physical hardware validation and is not yet implemented."
            )

        else:
            raise ValueError(f"Unknown compute execution target: {target}")
