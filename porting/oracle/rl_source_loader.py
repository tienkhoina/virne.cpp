"""Load actual Virne numeric classes while bypassing unrelated package initializers."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType


class SourceLoader:
    def __init__(self, root: Path):
        self.root = root
        self.hashes = {}
        for name in (
            "virne", "virne.utils", "virne.network", "virne.network.topology",
            "virne.core", "virne.core.controller", "virne.solver", "virne.solver.rank",
            "virne.solver.learning", "virne.solver.learning.rl_core",
        ):
            module = ModuleType(name)
            module.__path__ = [str(root.joinpath(*name.split(".")[1:]))]
            module.__package__ = name
            sys.modules[name] = module

    def load(self, relative: str):
        name = "virne." + relative.replace("/", ".").removesuffix(".py")
        path = self.root / relative
        self.hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        if name in sys.modules and getattr(sys.modules[name], "__file__", None):
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def export(package: str, module: ModuleType, names=None):
        destination = sys.modules[package]
        for name in names or [key for key in vars(module) if not key.startswith("_")]:
            setattr(destination, name, getattr(module, name))

    def environment(self):
        for relative in ("utils/network.py", "utils/dataset.py", "utils/setting.py", "utils/class_dict.py"):
            self.export("virne.utils", self.load(relative))
        self.load("utils/config.py")
        importlib.import_module("virne.network.attribute")
        for relative in ("network/topology/topology_generator.py", "network/topology/topological_metric_calculator.py"):
            self.export("virne.network.topology", self.load(relative))
        for relative, name in (("network/base_network.py", "BaseNetwork"),
                               ("network/physical_network.py", "PhysicalNetwork"),
                               ("network/virtual_network.py", "VirtualNetwork")):
            self.export("virne.network", self.load(relative), [name])
        attribute = importlib.import_module("virne.network.attribute")
        self.export("virne.network", attribute, ["BaseAttribute", "AttributeBenchmarkManager", "AttributeBenchmarks"])
        topology = sys.modules["virne.network.topology"]
        self.export("virne.network", topology, ["TopologyGenerator", "TopologicalMetricCalculator", "TopologicalMetrics"])
        self.export("virne.core", self.load("core/solution.py"), ["Solution"])
        self.export("virne.core", self.load("core/counter.py"), ["Counter"])
        self.export("virne.core", self.load("core/controller/controller.py"), ["Controller"])
        self.export("virne.core.controller", sys.modules["virne.core.controller.controller"], ["Controller"])
        # These collaborators are only stored/annotated by instance environments.
        sys.modules["virne.core"].Recorder = object
        sys.modules["virne.core"].Logger = object
        for child, exported in (("recorder", "Recorder"), ("logger", "Logger")):
            module = ModuleType("virne.core." + child)
            setattr(module, exported, object)
            sys.modules[module.__name__] = module
        self.export("virne.network", self.load("network/virtual_network_request_simulator.py"),
                    ["VirtualNetworkRequestSimulator"])
        self.export("virne.core", self.load("core/environment.py"), ["BaseEnvironment"])
        self.load("solver/rank/node_rank.py")
        self.load("solver/learning/obs_handler.py")
        self.load("solver/learning/rl_core/feature_constructor.py")
        self.load("solver/learning/rl_core/reward_calculator.py")
        self.load("solver/learning/rl_core/rl_enviroment_base.py")
        self.load("solver/learning/rl_core/online_rl_environment.py")
        result = self.load("solver/learning/rl_core/instance_rl_environment.py")
        for name, module in tuple(sys.modules.items()):
            if name.startswith("virne.") and getattr(module, "__file__", None):
                path = Path(module.__file__)
                if path.is_relative_to(self.root):
                    self.hashes[path.relative_to(self.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result


if __name__ == "__main__":
    loader = SourceLoader(Path("/virne/virne"))
    module = loader.environment()
    print("Loaded", module.__name__, "from", len(loader.hashes), "original source files")
