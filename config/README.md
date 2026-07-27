# Config composition contract

`config_lib` implements the Hydra/OmegaConf behavior used by this repository
without installing Hydra, Python, or another C++ dependency. `yaml-cpp` is
linked only from `libs/yaml-cpp`.

## Composition

`ConfigLoader::load("setting/main.yaml")` supports:

- recursive `defaults` lists;
- `_self_` at its exact list position, or implicitly last when omitted;
- scalar includes such as `learning` and `v_sim_setting/default`;
- group selections such as `{p_net_setting: default}`;
- `optional group: option`, `override group: option`, and null placeholders;
- package suffixes (`@package`, `@_here_`, `@_group_`, `@_global_`);
- recursive map merge, with scalars, nulls, and lists replaced as whole values;
- cycle and missing-file diagnostics.

All includes and later config-group overrides are resolved from the directory
of the main YAML file. They never depend on the process working directory.

## Interpolation

Interpolation is lazy, so changing a source value through a CLI override also
changes every value that references it. `get<T>()`, `get_raw()`, `root()`, and
`save()` return resolved values.

Supported forms are:

```yaml
copy: ${some.path}
message: http://${server.host}:${server.port}
sibling: ${.name}
parent: ${..root_value}
date: ${now:%Y-%m-%d}
environment: ${oc.env:NAME,default}
selected: ${oc.select:optional.path,default}
```

An interpolation that occupies the complete YAML value preserves its source
type, including lists and maps. Embedded interpolation produces a string.
Missing references, unknown resolvers, and interpolation cycles fail with a
descriptive exception.

## CLI overrides

`override_parser::apply` accepts Hydra-style dot paths:

```cpp
override_parser::apply(cfg, "experiment.seed=123");
override_parser::apply(cfg, "logger.backends=[console,file]");
override_parser::apply(cfg, "p_net_setting=p_net_setting_flagvne");
override_parser::apply(cfg, "+runtime.tags=[fast,true]");
override_parser::apply(cfg, "++runtime.tags=[]");
```

Values preserve booleans, null, signed/base-prefixed integers, decimal or
exponent floats, quoted strings, lists, and maps. Dotted and bracketed list
indices are supported. A normal override requires an existing key, `+` creates
a missing key, and `++` creates or replaces it. Existing null-valued keys count
as existing.

## Deliberate boundary

This is deterministic single-run composition, not the full Hydra runtime. It
does not implement multirun sweeps, launcher/plugin discovery, command-line key
deletion, custom resolver registration, or Hydra's generated output-directory
and job metadata. None of those behaviors are used by files under `setting/`.

## Shipped Virne setting matrix

The C++ tree carries every setting file from the original Python project plus
the two existing FlagVNE-specific options. Supported physical-network group
options are:

- `default`, `p_net_setting`, `p_net_setting_ltc`;
- `p_net_setting_multi_resource`, `p_net_setting_single_resource`;
- `wx100_p_net_setting`, `p_net_setting_flagvne`.

Supported virtual-simulation group options are:

- `default`, `v_sim_setting`, `v_sim_setting_ltc`;
- `v_sim_setting_multi_resource`, `v_sim_setting_single_resource`;
- `v_sim_setting_flagvne`.

The contract test composes `main.yaml` independently with every option and
checks the expected topology and attribute schema. This catches both YAML parse
regressions and config-group resolution regressions.

Run the contract test with:

```bash
ctest --test-dir build -R '^config_hydra_compat$' --output-on-failure
```
