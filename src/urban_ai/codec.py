from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .schema import ProgramConfig
from .validation import validate_program

OP_PAD = 0
OP_BOS = 1
OP_ROOT = 2
OP_ADD = 3
OP_CONNECT = 4
OP_EOS = 5
OP_COUNT = 6

MODES = ("road", "rail")
CLASSES = ("major", "secondary", "local", "rail", "subway", "light_rail", "tram")
VERTICAL_MODES = ("surface", "underground", "elevated", "unknown")

_MODE_TO_INDEX = {value: index for index, value in enumerate(MODES)}
_CLASS_TO_INDEX = {value: index for index, value in enumerate(CLASSES)}
_VERTICAL_TO_INDEX = {value: index for index, value in enumerate(VERTICAL_MODES)}

FIELDS = ("op", "x", "y", "id1", "id2", "mode", "class", "width", "vertical", "layer")


@dataclass(frozen=True)
class CommandCodecConfig:
    program: ProgramConfig = ProgramConfig()
    maximum_nodes: int = 512

    @property
    def maximum_width_bin(self) -> int:
        return max(1, int(round(self.program.maximum_width_m / self.program.width_quantum_m)))

    @property
    def layer_count(self) -> int:
        return self.program.layer_max - self.program.layer_min + 1

    def to_dict(self) -> dict[str, Any]:
        return {"program": asdict(self.program), "maximum_nodes": self.maximum_nodes}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CommandCodecConfig":
        return cls(
            program=ProgramConfig(**value.get("program", {})),
            maximum_nodes=int(value.get("maximum_nodes", 512)),
        )


def empty_encoded_command(op: int) -> dict[str, int]:
    return {field: (int(op) if field == "op" else 0) for field in FIELDS}


def encode_program(
    program: dict[str, Any],
    config: CommandCodecConfig | None = None,
) -> dict[str, list[int]]:
    config = config or CommandCodecConfig(
        program=ProgramConfig(**program.get("program_config", {}))
    )
    validate_program(program, config.program)
    commands = [empty_encoded_command(OP_BOS)]
    for command in program.get("commands", []):
        op = command["op"]
        if op == "root":
            encoded = empty_encoded_command(OP_ROOT)
            encoded["x"] = int(command["x_bin"]) + 1
            encoded["y"] = int(command["y_bin"]) + 1
            encoded["mode"] = _MODE_TO_INDEX[command["transport_mode"]] + 1
            encoded["vertical"] = _VERTICAL_TO_INDEX[command["vertical_mode"]] + 1
            encoded["layer"] = int(command["layer_bin"]) + 1
        elif op == "add":
            encoded = empty_encoded_command(OP_ADD)
            encoded["x"] = int(command["x_bin"]) + 1
            encoded["y"] = int(command["y_bin"]) + 1
            encoded["id1"] = int(command["parent"]) + 1
            encoded["mode"] = _MODE_TO_INDEX[command["transport_mode"]] + 1
            encoded["class"] = _CLASS_TO_INDEX[command["class"]] + 1
            encoded["width"] = int(command["width_bin"])
            encoded["vertical"] = _VERTICAL_TO_INDEX[command["vertical_mode"]] + 1
            encoded["layer"] = int(command["layer_bin"]) + 1
        elif op == "connect":
            encoded = empty_encoded_command(OP_CONNECT)
            encoded["id1"] = int(command["from"]) + 1
            encoded["id2"] = int(command["to"]) + 1
            encoded["mode"] = _MODE_TO_INDEX[command["transport_mode"]] + 1
            encoded["class"] = _CLASS_TO_INDEX[command["class"]] + 1
            encoded["width"] = int(command["width_bin"])
            encoded["vertical"] = _VERTICAL_TO_INDEX[command["vertical_mode"]] + 1
            encoded["layer"] = int(command["layer_bin"]) + 1
        else:
            raise ValueError(f"Unsupported graph-program command: {op}")
        commands.append(encoded)
    commands.append(empty_encoded_command(OP_EOS))
    return {field: [command[field] for command in commands] for field in FIELDS}


def command_sequence_length(program: dict[str, Any]) -> int:
    return len(program.get("commands", [])) + 2


def mode_name(index: int) -> str:
    return MODES[int(index)]


def mode_index(name: str) -> int:
    return _MODE_TO_INDEX[name]


def class_name(index: int) -> str:
    return CLASSES[int(index)]


def vertical_name(index: int) -> str:
    return VERTICAL_MODES[int(index)]


def vertical_index(name: str) -> int:
    return _VERTICAL_TO_INDEX[name]


def classes_for_mode(mode: str) -> tuple[int, ...]:
    if mode == "road":
        return tuple(_CLASS_TO_INDEX[value] for value in ("major", "secondary", "local"))
    return tuple(_CLASS_TO_INDEX[value] for value in ("rail", "subway", "light_rail", "tram"))
