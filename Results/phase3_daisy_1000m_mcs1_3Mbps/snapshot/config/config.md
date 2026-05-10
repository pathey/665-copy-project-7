# Config System

This system manages configuration using YAML files with support for:

- JSON Schema validation
- Dynamic expression evaluation
- Layered overrides
- VS Code integration for schema-based autocomplete
- pyi stub generation for language server support

## Files

| File           | Purpose                                                                                             |
| -------------- | --------------------------------------------------------------------------------------------------- |
| set_schema.py  | Converts YAML schema to JSON Schema, updates VS Code settings, and generates `.pyi` stubs.        |
| load_config.py | Loads configuration stack, resolves overrides and expressions, injects config into a Python module. |

## Schema Format

The YAML schema must have a single top-level key called Config (excluding `__effects__`). Both 'type' and 'type-hint' are now required to simplify 'set_schema.py'.

Example:

__effects__: "*.yaml"

```
Config:
  PAYLOAD_DATA_RATE:
    type: number
    type-hint: float
    description: in Mbps
    default: 1.5
  NUMBER_OF_NODES:
    type: number
    type-hint: int
    description: Number of all nodes (regular + BS)
    default: {eval: "50 + ${NUMBER_OF_BS}"}
```

### Expression Support

Fields can be primitives, like numbers, strings, arrays, etc. or use expressions using an eval block:

```
WAVELENGTH: 
  eval: 'c/${FREQUENCY}'
```

## Usage

### 1. Generate Schema and Load Config

Run:

python set_schema.py --schema_yaml=config/schemas/schema.yaml --schema_json=config/schemas/schema.json

This will:

- Generate a JSON Schema
- Update `.vscode/settings.json` to enable YAML autocompletion
- Generate `.pyi` type stubs for config

### 2. Access in Code

After running `set_schema.py`, access config values as regular variables:

import parameters

print(parameters.WAVELENGTH)

main.py automatically calls load_config.py when run.

## Override Mechanism

A config can extend another using `__override__` or `__overrides__`.

Example:

__override__: base.yaml

This loads `base.yaml` first, then applies overrides from the current file. All layers are merged and evaluated in order.

## Expression Resolution

- `${var}` placeholders are substituted using previously resolved config.
- `eval` blocks are evaluated using `asteval` with a limited set of globals (can be added to if needed):
  - np (NumPy)
  - simpy
  - pi, c (from scipy.constants)

## Output

- `parameters.py`: Target Python module. Must exist.
- `parameters.pyi`: Auto-generated stub with types and doc comments.
- `parameters-cfg.log`: Log showing source and evaluation history per key.

## New Requirements

pyyaml jsonschema asteval

## Notes

- Running set_schema.py with --required will allow you to easily autocomplete (ctrl+space in vscode) all config values (only works one level deep, to autocomplete a level delete its values and autocomplete)
- The config promotion system injects values into parameters as top-level attributes.
- Config values can be changed using CLI as --KEY=VALUE, but use yaml config for anything other than primitives without dependents
