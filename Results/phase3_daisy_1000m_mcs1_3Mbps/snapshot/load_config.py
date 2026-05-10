from collections.abc import Iterable
import sys
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, ParamSpec, TypeAlias, TypeVar, Union
import yaml
import json
import re
from graphlib import TopologicalSorter
from asteval.asteval import Interpreter
import jsonschema
from types import ModuleType, SimpleNamespace
import numpy as np
from scipy.constants import c, pi
import simpy
from collections import defaultdict
import io

from config import set_schema

from line_profiler import profile # type: ignore[reportAssignmentType]

P = ParamSpec("P")
R = TypeVar("R")

try:
	@profile
	def check_for_line_profiler() -> None:
		pass
except:
	def profile(f: Callable[P, R]) -> Callable[P, R]:
		return f

VAR_PATTERN = re.compile(r"\$\{([a-zA-Z0-9_.]+)\}")

@profile
def merge_nested_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
	"""
	Merge two dictionaries recursively. override's values take precedence.
	"""
	return {
		key: merge_nested_dicts(base[key], value) if isinstance(base.get(key), Mapping) and isinstance(value, Mapping) else value # type: ignore[reportUnknownArgumentType]
		for key, value in {**base, **override}.items()
	}

@profile
def flatten_for_eval(data: dict[str, Any], parent: str = "") -> dict[str, Any]:
	"""
	Flatten nested dictionaries into dotted paths for expression substitution and evaluation.
	"""
	@profile
	def yield_flat_items(d: dict[str, Any], parent: str) -> Iterator[tuple[str, Any]]:
		for key, value in d.items():
			full_key = f"{parent}.{key}" if parent else key
			if isinstance(value, dict) and "eval" not in value:
				yield from yield_flat_items(value, full_key) # type: ignore[reportUnknownArgumentType]
			else:
				yield full_key, value

	return dict(yield_flat_items(data, parent))

@profile
def unflatten_from_eval(flat: dict[str, Any]) -> dict[str, Any]:
	"""
	Reconstruct nested dicts from dotted key paths after evaluation.
	"""
	result: dict[str, Any] = {}
	for dotted_key, value in flat.items():
		parts = dotted_key.split(".")
		node = result
		for part in parts[:-1]:
			node = node.setdefault(part, {})
		node[parts[-1]] = value
	return result


PlainValue: TypeAlias = Union[str, int, float, bool, None, dict[str, Any], list[Any], tuple[Any, ...]]

@profile
def to_plain_python(value: Any) -> PlainValue:
	"""
	Convert to standard Python types so the result is accepted by jsonschema validation.
	"""
	if isinstance(value, np.generic):
		return value.item() # type: ignore[return-value]
	if isinstance(value, np.ndarray):
		return [to_plain_python(x) for x in value]

	if isinstance(value, dict):
		return {k: to_plain_python(v) for k, v in value.items()} # type: ignore[reportUnknownVariableType]
	if isinstance(value, (list, tuple)):
		return type(value)(to_plain_python(x) for x in value) # type: ignore[call-arg]

	return value

@profile
def substitute_for_eval(expr: str, flat_values: dict[str, Any]) -> str:
	"""
	Replace ${var} with evaluable representations.
	"""
	def replace(match: re.Match[str]) -> str:
		key = match.group(1)
		if key not in flat_values:
			raise KeyError(f"Unresolved variable '{key}' in expression '{expr}'")
		value = flat_values[key]

		if isinstance(value, str):
			return repr(value)  # ensures proper quotes for strings
		if isinstance(value, (dict, list, tuple)):
			return repr(to_plain_python(value))
		return str(value)

	return VAR_PATTERN.sub(replace, expr)

@profile
def substitute_for_log(expr: str, flat_values: dict[str, Any]) -> str:
	"""
	Replace ${var} with string representations for logging. Unresolved variables remain as-is.
	"""
	def replace(match: re.Match[str]) -> str:
		key = match.group(1)
		if key not in flat_values:
			return match.group(0)
		value = flat_values[key]
		return str(to_plain_python(value))

	return VAR_PATTERN.sub(replace, expr)

@profile
def extract_dependencies(expr: str) -> set[str]:
	"""
	Return all variable names (in the form ${var_name}) referenced in the expression.
	"""
	return set(VAR_PATTERN.findall(expr))

@profile
def dict_to_namespace(data: Any, leaf_keys: Optional[Iterable[str]] = None, path: tuple[str, ...] = ()) -> SimpleNamespace:
	"""
	Recursively convert a nested dict to SimpleNamespace terminating recursion at leaf_keys.
	"""
	if not isinstance(data, dict):
		return data

	leaf_keys_set = set(leaf_keys or ())
	
	ns_dict: dict[Any, Any] = {
		key: dict_to_namespace(value, leaf_keys_set, path + (str(key),)) # type: ignore[reportUnknownArgumentType]
			if isinstance(value, dict) and ".".join(path + (str(key),)) not in leaf_keys_set # type: ignore[reportUnknownArgumentType]
			else value
		for key, value in data.items() # type: ignore[reportUnknownVariableType]
	}

	return SimpleNamespace(**ns_dict)

@profile
def load_config_stack(start_path: str = "config/cfg/default.yaml", schema_path: str = "config/schemas/schema.json",
					  target_module: Optional[ModuleType] = None, interactive_override: dict[str, Any] = {}) -> SimpleNamespace:
	"""
	Load and merge configuration files recursively, evaluate expressions,
	validate against the JSONSchema, and add variables to target module if provided.
	"""
	cfg_stack: list[tuple[dict[str, Any], Path]] = []
	cfg_path = Path(start_path).resolve()
	parent_path = cfg_path.parent
	while True:
		with cfg_path.open("r", encoding="utf-8") as f:
			cfg: dict[str, Any] = yaml.safe_load(f) or {}
			cfg_stack.append((cfg, cfg_path))

			override_file = cfg.get("__override__") or cfg.get("__overrides__")
			if isinstance(override_file, str):
				cfg_path = (parent_path / override_file).resolve()
			else:
				break

	if Path(schema_path).suffix == ".yaml":
		schema_json = Path(schema_path).with_suffix(".json")
		set_schema.generate_schema_and_types(Path(schema_path), Path(schema_json))
		print(f"{schema_path} converted to JSON schema. Please rerun.")
		sys.exit(1)

	merged_config: dict[str, Any] = {}
	sources: dict[str, str] = {}

	for cfg, cfg_path in reversed(cfg_stack):
		config_body = cfg.get("Config", cfg)

		# Track source file for each key
		for key in flatten_for_eval(config_body):
			sources.setdefault(key, cfg_path.name)

		merged_config = merge_nested_dicts(merged_config, cfg)
	
	merged_config = merge_nested_dicts(merged_config, interactive_override)

	config_body = merged_config.get("Config", merged_config)
	flat_config = flatten_for_eval(config_body)

	# Extract eval fields to create tolopogical sort
	eval_fields: dict[str, str] = {key: value["eval"] for key, value in flat_config.items() if isinstance(value, dict) and "eval" in value}

	for key in eval_fields:
		flat_config[key] = None

	leaf_keys = set(flat_config.keys())
	ancestry: dict[str, list[dict[str, Any]]] = defaultdict(list)

	for key, value in flat_config.items():
		src_file = sources.get(key, "unknown")
		if key in eval_fields:
			ancestry[key].append({
				"source": src_file,
				"raw": eval_fields[key]
			})
		else:
			ancestry[key].append({
				"source": src_file,
				"value": to_plain_python(value)
			})

	dep_graph: dict[str, list[str]] = { key: [dep for dep in extract_dependencies(expr) if dep in eval_fields]
										for key, expr in eval_fields.items() }

	ts = TopologicalSorter(dep_graph)
	
	# Substitute and eval expressions as needed
	ae = Interpreter(err_writer=io.StringIO()) # type: ignore
	ae.symtable.update({"np": np, "pi": pi, "c": c, "simpy": simpy}) # type: ignore[reportUnknownMemberType]

	for key in ts.static_order():
		raw_expr = eval_fields[key]
		substituted_eval = substitute_for_eval(raw_expr, flat_config)
		substituted_log = substitute_for_log(raw_expr, flat_config)
		ae.symtable.update(flat_config) # type: ignore[reportUnknownMemberType]
		value = ae(substituted_eval) # type: ignore[reportUnknownVariableType]
		if value is None: value = substituted_eval
		flat_config[key] = to_plain_python(value)

		ancestry[key].append({
			"substituted": substituted_log,
			"value": flat_config[key]
		})
	
	# Validate and log
	final_config = unflatten_from_eval(flat_config)

	with open(schema_path, "r", encoding="utf-8") as f:
		schema = json.load(f)
		jsonschema.validate({"Config": to_plain_python(final_config)}, schema) # type: ignore[reportUnknownMemberType]

	log_file = Path(start_path).parent / "parameters-cfg.log"
	with open(log_file, "w", encoding="utf-8") as f:
		yaml.dump(dict(ancestry), f, default_flow_style=False, sort_keys=False)

	# Add as globals of target module
	if target_module:
		ns = dict_to_namespace(final_config, leaf_keys=leaf_keys)
		for k, v in vars(ns).items():
			setattr(target_module, k, v)

	return dict_to_namespace(final_config, leaf_keys=leaf_keys)
