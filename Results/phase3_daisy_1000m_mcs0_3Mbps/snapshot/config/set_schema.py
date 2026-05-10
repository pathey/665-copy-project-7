import os
import json
from pathlib import Path
import yaml
import argparse
from typing import Any, Sequence, Union

JSONSchemaArray = Sequence[Union["JSONSchemaNode", str, int, float, bool]]
JSONSchemaNode = Union[dict[str, "JSONSchemaNode"], JSONSchemaArray, str, int, float, bool]
JSONSchema = dict[str, "JSONSchemaNode"]


def get_workspace_root() -> str:
    """Returns workspace root (./) relative to this file (./config/set_schema.py)"""
    return str(Path(__file__).parent.parent.resolve())

def get_parameters_path() -> str:
    """Returns path to parameters.py which should be in ./ relative to main.py"""
    return str(Path(get_workspace_root()) / "parameters.py")

def get_settings_path() -> str:
    """Return path to .vscode/settings.json, creating .vscode if needed"""
    vscode_dir = Path(get_workspace_root()) / ".vscode"
    vscode_dir.mkdir(exist_ok=True)
    return str(vscode_dir / "settings.json")


def wrap_eval_node(node: dict[str, Any]) -> dict[str, Any]:
    """Wraps a base type to allow literal or {eval: string} values for JSON Schema"""
    base_node = node.copy()
    description = base_node.pop("description", None)
    default = base_node.pop("default", None)
    examples = base_node.pop("examples", None)

    wrapped: dict[str, Any] = {
        "anyOf": [
            base_node,
            {
                "type": "object",
                "properties": {"eval": {"type": "string", "title": "eval"}},
                "required": ["eval"],
                "additionalProperties": False,
            },
        ]
    }
    
    if description is not None:
        wrapped["description"] = description
    if default is not None:
        wrapped["default"] = default
    if examples is not None:
        wrapped["examples"] = examples

    return wrapped


def yaml_to_json_schema(root_name: str, schema_yaml: dict[str, Any]) -> JSONSchema:
    """Convert a YAML schema to JSON Schema, wrapping leaf nodes w/ {eval: string}"""
    def convert_node(node: JSONSchemaNode) -> JSONSchemaNode:
        if isinstance(node, dict) and "type" in node:
            return wrap_eval_node(node)

        if isinstance(node, dict):
            props: dict[str, JSONSchemaNode] = {
                key: convert_node(value) for key, value in node.items()
            }
            return {"type": "object", "properties": props}

        raise ValueError(f"Invalid schema node: {node!r}")

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {root_name: convert_node(schema_yaml)},
        "required": [root_name],
    }


def update_vscode_settings(schema_json_path: str, effects: Union[str, list[str]]) -> None:
    """Associate a JSON schema with files in VSCode YAML settings"""
    settings_path = Path(get_settings_path())
    schema_json_path = str(Path(schema_json_path).resolve())

    try:
        settings = json.load(settings_path.open("r", encoding="utf-8")) if settings_path.exists() else {} # type: ignore[reportUnknownVariableType]
    except json.JSONDecodeError:
        print("[set_schema] Invalid JSON in .vscode/settings.json; overwriting")
        settings = {}

    settings["yaml.schemas"] = settings.get("yaml.schemas", {}) # type: ignore[reportUnknownMemberType]
    settings["yaml.schemas"][schema_json_path] = effects if isinstance(effects, list) else [effects]

    json.dump(settings, settings_path.open("w", encoding="utf-8"), indent=2)
    print(f"[set_schema] Updated VSCode settings: {settings_path}")


def generate_types(schema: JSONSchema, root_key: str) -> str:
    """Generate Python type hints from schema using type-hint; objects generate SimpleNamespace classes"""
    lines: list[str] = [
        "from typing import TypedDict",
        "from types import SimpleNamespace",
        "import numpy as np",
        "from numpy.typing import NDArray",
        'MCSTableEntry = TypedDict("MCSTableEntry", {"Modulation": str, "Coding": float, "MinSNR(dB)": int, "MinSNR": float, "DataRate": float})'
    ]
    generated: dict[str, bool] = {}

    def type_for_node(node: dict[str, Any]) -> str:
        if "anyOf" in node:
            for sub in node["anyOf"]:
                if "type-hint" in sub:
                    return sub["type-hint"]
        return node.get("type-hint", "")

    def convert_node(node: dict[str, Any], class_name: str) -> str:
        props = node.get("properties", {})
        class_lines = [f"class {class_name}(SimpleNamespace):"]
        if not props:
            class_lines.append("\tpass")
            return "\n".join(class_lines)

        for key, subnode in props.items():
            if "properties" in subnode:
                nested_name = key.capitalize()
                if nested_name not in generated:
                    lines.append(convert_node(subnode, nested_name))
                    generated[nested_name] = True
                class_lines.append(f"\t{key}: {nested_name}")
            else:
                class_lines.append(f"\t{key}: {type_for_node(subnode)}")

        return "\n".join(class_lines)

    schema_props = schema.get("properties", {})
    if not isinstance(schema_props, dict): return "\n".join(lines)
    root_node = schema_props.get(root_key, {})
    if not isinstance(root_node, dict): return "\n".join(lines)
    root_props = root_node.get("properties", {})
    if isinstance(root_props, dict):
        for key, subnode in root_props.items():
            if not isinstance(subnode, dict):
                continue
            if "properties" in subnode:
                class_name = key.capitalize()
                if class_name not in generated:
                    lines.append(convert_node(subnode, class_name))
                    generated[class_name] = True
                lines.append(f"{key}: {class_name}")
            else:
                lines.append(f"{key}: {type_for_node(subnode)}")

    return "\n".join(lines)


def inject_into_parameters_py(generated_types: str, parameters_path: str) -> None:
    """Insert generated type hints into parameters.py between defined markers."""
    start_marker = "# === AUTO-GENERATED FROM CONFIG START ==="
    end_marker = "# === AUTO-GENERATED FROM CONFIG END ==="
    path = Path(parameters_path)

    if not path.exists():
        path.write_text(f"{start_marker}\n{end_marker}\n\n", encoding="utf-8")
        
    content = path.read_text(encoding="utf-8")

    if start_marker not in content or end_marker not in content:
        content = f"{start_marker}\n{end_marker}\n\n" + content

    new_content = (
        content[: content.find(start_marker)]
        + f"{start_marker}\n{generated_types}\n{end_marker}"
        + content[content.find(end_marker) + len(end_marker) :]
    )

    path.write_text(new_content, encoding="utf-8")
    print(f"[set_schema] Updated module-level TypedDicts in {parameters_path}")


def generate_schema_and_types(schema_yaml_path: Path, schema_json_path: Path) -> None:
    """Generate JSONSchema and Python type hints from YAML schema, update VSCode, and inject types into parameters.py."""
    
    with schema_yaml_path.open("r", encoding="utf-8") as f:
        raw_yaml: dict[str, Any] = yaml.safe_load(f)

    effects: Union[str, list[str]] = raw_yaml.pop("__effects__", "*.yaml")
    if len(raw_yaml) != 1:
        raise ValueError("YAML schema must have a single root key")

    root_key, schema_def = next(iter(raw_yaml.items()))
    json_schema = yaml_to_json_schema(root_key, schema_def)

    with schema_json_path.open("w", encoding="utf-8") as f:
        json.dump(json_schema, f, indent=2)
    print(f"[set_schema] Generated JSON Schema at {schema_json_path}")

    update_vscode_settings(str(schema_json_path), effects)

    generated_types = generate_types(json_schema, root_key)
    inject_into_parameters_py(generated_types, get_parameters_path())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSON schema and module-level type hints")
    parser.add_argument(
        "--schema_yaml",
        default=os.path.join(os.path.dirname(__file__), "schemas/schema.yaml"),
        help="Path to input YAML schema",
    )
    parser.add_argument(
        "--schema_json",
        default=os.path.join(os.path.dirname(__file__), "schemas/schema.json"),
        help="Path to output JSON schema",
    )
    args = parser.parse_args()
    generate_schema_and_types(Path(args.schema_yaml), Path(args.schema_json))
