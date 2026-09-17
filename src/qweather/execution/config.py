from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, Field

class IBMQuantumConfig(BaseModel):
    backend_name: str = "ibm_brisbane"

class ComputeConfig(BaseModel):
    target: Literal["local_aer", "cloud_gpu", "ibm_qpu"] = "local_aer"
    shots: int = Field(default=1000, gt=0, description="Number of circuit execution shots")
    ibm_quantum: IBMQuantumConfig = Field(default_factory=IBMQuantumConfig)

class AppConfig(BaseModel):
    compute: ComputeConfig

def load_config(config_path: str | Path = "configs/default.yaml") -> AppConfig:
    """Load and validate the system configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
      raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
      raw_data = yaml.safe_load(f) or {}

    return AppConfig(**raw_data)