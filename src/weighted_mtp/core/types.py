"""공통 타입 정의"""

from pathlib import Path
from typing import TypeAlias

PathLike: TypeAlias = str | Path
Device: TypeAlias = str  # "cpu" | "cuda" | "mps"
DType: TypeAlias = str  # "float16" | "bfloat16" | "float32"
