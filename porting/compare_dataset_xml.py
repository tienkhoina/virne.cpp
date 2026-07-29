#!/usr/bin/env python3
"""Exact parser/graph/GML differential for the non-Torch dataset XML leaf."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Iterable
from xml.dom import minidom

import compare_dataset_core


EXPECTED_NETWORKX = "3.4.2"


@dataclass(frozen=True)
class Fixture:
    name: str
    path: pathlib.Path
    topology_name: str


def text_hex(value: str) -> str:
    return value.encode("utf-8").hex()


def parse_records(path: pathlib.Path) -> tuple[list[tuple[str, str, str]], list[tuple[str, ...]]]:
    # Python 3.10 minidom treats pathlib.Path as a file object instead of
    # applying os.fspath; the production oracle receives the same path as text.
    document = minidom.parse(str(path))
    nodes = []
    for node in document.getElementsByTagName("node"):
        nodes.append(
            (
                node.attributes["id"].value,
                node.getElementsByTagName("x")[0].firstChild.data,
                node.getElementsByTagName("y")[0].firstChild.data,
            )
        )
    edges = []
    for link in document.getElementsByTagName("link"):
        capacities = link.getElementsByTagName("capacity")
        capacity_st = capacities[0].firstChild.data
        capacity_ts = capacities[1].firstChild.data
        edges.append(
            (
                link.attributes["id"].value,
                link.getElementsByTagName("source")[0].firstChild.data,
                link.getElementsByTagName("target")[0].firstChild.data,
                capacity_st,
                capacity_ts,
                capacity_st,
                capacity_ts,
            )
        )
    return nodes, edges


def serialize_records(records: tuple[list[tuple[str, ...]], list[tuple[str, ...]]]) -> bytes:
    nodes, edges = records
    parts = ["records-v1\n", f"nodes={len(nodes)}\n"]
    parts.extend("node=" + "|".join(text_hex(value) for value in node) + "\n" for node in nodes)
    parts.append(f"edges={len(edges)}\n")
    parts.extend("edge=" + "|".join(text_hex(value) for value in edge) + "\n" for edge in edges)
    return "".join(parts).encode("ascii")


def serialize_batch(items: Iterable[tuple[list[tuple[str, ...]], list[tuple[str, ...]]]]) -> bytes:
    payloads = [serialize_records(item) for item in items]
    output = bytearray(f"batch-v1\ncount={len(payloads)}\n".encode("ascii"))
    for payload in payloads:
        output.extend(f"item-bytes={len(payload)}\n".encode("ascii"))
        output.extend(payload)
    return bytes(output)


def oracle_preprocess(module, name: str, source: pathlib.Path, target: pathlib.Path):
    """Call the original path-string API without pathlib's minidom ambiguity."""
    return module.preprocess_xml(name, str(source), str(target))


NODE_FIELDS = ("label", "x", "y")
EDGE_FIELDS = (
    "label",
    "source_label",
    "target_label",
    "capacity_st",
    "capacity_ts",
    "cost_st",
    "cost_ts",
)


def require_string(value, context: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{context} is not a string: {type(value).__name__}")
    return value


def serialize_graph(graph) -> bytes:
    if graph.is_directed() or graph.is_multigraph():
        raise RuntimeError("preprocess_xml returned the wrong graph kind")
    if list(graph.graph) != ["name"]:
        raise RuntimeError(f"graph attribute order drift: {list(graph.graph)!r}")
    parts = [
        "graph-v1\n",
        "directed=0\n",
        "multigraph=0\n",
        f"name={text_hex(require_string(graph.graph['name'], 'graph name'))}\n",
        f"nodes={graph.number_of_nodes()}\n",
    ]
    expected_nodes = list(range(graph.number_of_nodes()))
    if list(graph.nodes) != expected_nodes:
        raise RuntimeError(f"node order/density drift: {list(graph.nodes)!r}")
    for node, attributes in graph.nodes(data=True):
        if list(attributes) != list(NODE_FIELDS):
            raise RuntimeError(f"node {node} attribute order drift: {list(attributes)!r}")
        values = [require_string(attributes[field], f"node {node}/{field}") for field in NODE_FIELDS]
        parts.append(f"node={node}|" + "|".join(text_hex(value) for value in values) + "\n")
    edge_items = list(graph.edges(data=True))
    parts.append(f"edges={len(edge_items)}\n")
    for source, target, attributes in edge_items:
        if list(attributes) != list(EDGE_FIELDS):
            raise RuntimeError(
                f"edge {(source, target)} attribute order drift: {list(attributes)!r}"
            )
        values = [
            require_string(attributes[field], f"edge {(source, target)}/{field}")
            for field in EDGE_FIELDS
        ]
        parts.append(
            f"edge={source}|{target}|"
            + "|".join(text_hex(value) for value in values)
            + "\n"
        )
    return "".join(parts).encode("ascii")


def parse_response(process: subprocess.CompletedProcess[str], benchmark: bool = False) -> dict[str, str]:
    if process.returncode != 0:
        raise RuntimeError(f"C++ XML harness failed: {process.stderr.strip()}")
    values: dict[str, str] = {}
    for line in process.stdout.splitlines():
        if "=" not in line:
            raise RuntimeError(f"malformed XML harness line: {line!r}")
        key, value = line.split("=", 1)
        if key in values:
            raise RuntimeError(f"duplicate XML harness key: {key}")
        values[key] = value
    version_key = "dataset_xml_benchmark_version" if benchmark else "dataset_xml_harness_version"
    if values.get(version_key) != "1":
        raise RuntimeError(f"invalid XML harness version: {values!r}")
    return values


def run_cpp(harness: pathlib.Path, *arguments: str) -> dict[str, str]:
    process = subprocess.run(
        [str(harness), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return parse_response(process)


def require_cpp_ok(values: dict[str, str], command: str) -> tuple[bytes, bytes | None]:
    required = {"dataset_xml_harness_version", "command", "status", "payload_hex"}
    if "gml_hex" in values:
        required.add("gml_hex")
    if set(values) != required or values["command"] != command or values["status"] != "ok":
        raise RuntimeError(f"invalid C++ {command} success response: {values!r}")
    payload = bytes.fromhex(values["payload_hex"])
    gml = bytes.fromhex(values["gml_hex"]) if "gml_hex" in values else None
    return payload, gml


def require_cpp_error(
    values: dict[str, str],
    command: str,
    code: str,
    operation: str,
    input_index: str = "none",
) -> None:
    required = {
        "dataset_xml_harness_version",
        "command",
        "status",
        "error_code",
        "operation",
        "input_index",
        "path_hex",
        "target_kind",
        "target_hex",
    }
    if set(values) != required:
        raise RuntimeError(f"invalid C++ {command} error response keys: {values!r}")
    actual = (values["command"], values["status"], values["error_code"], values["operation"], values["input_index"])
    expected = (command, "error", code, operation, input_index)
    if actual != expected:
        raise RuntimeError(f"C++ {command} error mismatch: actual={actual}, expected={expected}")


def assert_exact_bytes(name: str, python_bytes: bytes, cpp_bytes: bytes) -> None:
    if python_bytes == cpp_bytes:
        return
    limit = min(len(python_bytes), len(cpp_bytes))
    offset = next((index for index in range(limit) if python_bytes[index] != cpp_bytes[index]), limit)
    raise RuntimeError(
        f"{name} exact byte mismatch: offset={offset}, "
        f"python_len={len(python_bytes)}, cpp_len={len(cpp_bytes)}, "
        f"python_sha256={hashlib.sha256(python_bytes).hexdigest()}, "
        f"cpp_sha256={hashlib.sha256(cpp_bytes).hexdigest()}, "
        f"python_slice={python_bytes[offset:offset + 40]!r}, "
        f"cpp_slice={cpp_bytes[offset:offset + 40]!r}"
    )


def write_fixtures(root: pathlib.Path) -> list[Fixture]:
    minimal = root / "minimal.xml"
    minimal.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<network><networkStructure><nodes>"
        '<node id="A"><coordinates><meta><x>first-x</x></meta>'
        "<x>ignored-x</x><y>1 &amp; 2</y></coordinates></node>"
        '<node id="B"><coordinates><x><![CDATA[3<4]]></x><y>5</y></coordinates></node>'
        "</nodes><links>"
        '<link id="AB"><source>A</source><target>B</target><additionalModules>'
        "<addModule><capacity>10</capacity><cost>900</cost></addModule>"
        "<addModule><capacity>20</capacity><cost>800</cost></addModule>"
        "<addModule><capacity>30</capacity><cost>700</cost></addModule>"
        "</additionalModules></link></links></networkStructure></network>",
        encoding="utf-8",
    )

    duplicates = root / "duplicates.xml"
    duplicates.write_text(
        "<network>"
        '<link id="AB-first"><source>A</source><target>B</target>'
        "<capacity>10</capacity><capacity>20</capacity><cost>900</cost></link>"
        '<node id="A"><x>0</x><y>0</y></node>'
        '<node id="A"><x>1</x><y>1</y></node>'
        '<node id="B"><x>2</x><y>2</y></node>'
        '<link id="BA-last"><source>B</source><target>A</target>'
        "<capacity>30</capacity><capacity>40</capacity><cost>800</cost></link>"
        '<link id="BB"><source>B</source><target>B</target>'
        "<capacity>50</capacity><capacity>60</capacity></link>"
        "</network>",
        encoding="utf-8",
    )

    utf8 = root / "utf8.xml"
    utf8.write_bytes(
        b"\xef\xbb\xbf"
        + '<?xml version="1.0" encoding="UTF-8"?>'
        '<network><node id="Né"><x>漢</x><y>quote &quot; &amp;</y></node></network>'.encode("utf-8")
    )

    latin1 = root / "latin1.xml"
    latin1.write_bytes(
        '<?xml version="1.0" encoding="ISO-8859-1"?>'
        '<network><node id="Né"><x>1</x><y>2</y></node></network>'.encode("latin-1")
    )

    empty_ids = root / "empty_ids.xml"
    empty_ids.write_text(
        '<network><node id=""><x>0</x><y>0</y></node>'
        '<node id="B"><x>1</x><y>1</y></node>'
        '<link id=""><source><!----></source><target>B</target>'
        '<capacity>7</capacity><capacity>8</capacity></link></network>',
        encoding="utf-8",
    )

    first_children = root / "first_children.xml"
    first_children.write_bytes(
        b'<?xml version="1.0" encoding="UTF-8"?>\r\n'
        b'<network><node id="Q&amp;&quot;&#233;">'
        b'<x>\r\n value</x><y><!--comment & literal-->ignored</y></node></network>'
    )

    prefixed = root / "prefixed.xml"
    prefixed.write_text(
        '<s:network xmlns:s="urn:test"><s:node id="ignored">'
        '<s:x>1</s:x><s:y>2</s:y></s:node></s:network>',
        encoding="utf-8",
    )
    return [
        Fixture("minimal", minimal, "Minimal"),
        Fixture("duplicates", duplicates, "Duplicate & reverse"),
        Fixture("utf8", utf8, "Unicode é漢"),
        Fixture("latin1", latin1, "Latin é"),
        Fixture("empty_ids", empty_ids, "Empty IDs"),
        Fixture("first_children", first_children, "Line\n& emoji 😀"),
        Fixture("prefixed", prefixed, "Prefixed namespace"),
    ]


def check_valid_cases(module, harness: pathlib.Path, fixtures: list[Fixture], root: pathlib.Path) -> int:
    count = 0
    for fixture in fixtures:
        expected_records = serialize_records(parse_records(fixture.path))
        actual, gml = require_cpp_ok(run_cpp(harness, "parse", str(fixture.path)), "parse")
        if gml is not None:
            raise RuntimeError("parse unexpectedly returned GML bytes")
        assert_exact_bytes(f"{fixture.name}/records", expected_records, actual)
        count += 1

        python_materialize_target = root / f"{fixture.name}_python_materialize.gml"
        python_graph = oracle_preprocess(
            module, fixture.topology_name, fixture.path, python_materialize_target
        )
        expected_graph = serialize_graph(python_graph)
        actual, gml = require_cpp_ok(
            run_cpp(harness, "materialize", fixture.topology_name, str(fixture.path)),
            "materialize",
        )
        if gml is not None:
            raise RuntimeError("materialize unexpectedly returned GML bytes")
        assert_exact_bytes(f"{fixture.name}/graph", expected_graph, actual)
        count += 1

        python_target = root / f"{fixture.name}_python.gml"
        cpp_target = root / f"{fixture.name}_cpp.gml"
        python_graph = oracle_preprocess(
            module, fixture.topology_name, fixture.path, python_target
        )
        expected_graph = serialize_graph(python_graph)
        expected_gml = python_target.read_bytes()
        response = run_cpp(
            harness,
            "preprocess",
            fixture.topology_name,
            str(fixture.path),
            str(cpp_target),
        )
        actual_graph, actual_gml = require_cpp_ok(response, "preprocess")
        if actual_gml is None:
            raise RuntimeError("preprocess did not return GML bytes")
        assert_exact_bytes(f"{fixture.name}/preprocess-graph", expected_graph, actual_graph)
        assert_exact_bytes(f"{fixture.name}/gml", expected_gml, actual_gml)
        if actual_gml != cpp_target.read_bytes():
            raise RuntimeError(f"{fixture.name}: harness GML differs from target bytes")
        count += 1
    return count


def expect_python_error(callable_value, expected_types: tuple[type[BaseException], ...]) -> None:
    try:
        callable_value()
    except expected_types:
        return
    except Exception as error:
        raise RuntimeError(f"unexpected Python error type {type(error).__name__}: {error}") from error
    raise RuntimeError("expected Python XML error")


def check_errors(module, harness: pathlib.Path, root: pathlib.Path) -> int:
    malformed = root / "malformed.xml"
    malformed.write_text("<network><node></network>", encoding="utf-8")
    missing_x = root / "missing_x.xml"
    missing_x.write_text('<network><node id="A"><y>2</y></node></network>', encoding="utf-8")
    empty_y = root / "empty_y.xml"
    empty_y.write_text('<network><node id="A"><x>1</x><y/></node></network>', encoding="utf-8")
    one_capacity = root / "one_capacity.xml"
    one_capacity.write_text(
        '<network><node id="A"><x>1</x><y>2</y></node>'
        '<link id="AA"><source>A</source><target>A</target>'
        "<capacity>1</capacity></link></network>",
        encoding="utf-8",
    )
    missing_y = root / "missing_y.xml"
    missing_y.write_text(
        '<network><node id="A"><x>1</x></node></network>', encoding="utf-8"
    )
    missing_node_id = root / "missing_node_id.xml"
    missing_node_id.write_text(
        '<network><node><x>1</x><y>2</y></node></network>', encoding="utf-8"
    )
    missing_link_id = root / "missing_link_id.xml"
    missing_link_id.write_text(
        '<network><node id="A"><x>1</x><y>2</y></node>'
        '<link><source>A</source><target>A</target><capacity>1</capacity>'
        '<capacity>2</capacity></link></network>',
        encoding="utf-8",
    )
    missing_source = root / "missing_source.xml"
    missing_source.write_text(
        '<network><node id="A"><x>1</x><y>2</y></node>'
        '<link id="AA"><target>A</target><capacity>1</capacity>'
        '<capacity>2</capacity></link></network>',
        encoding="utf-8",
    )
    missing_target = root / "missing_target.xml"
    missing_target.write_text(
        '<network><node id="A"><x>1</x><y>2</y></node>'
        '<link id="AA"><source>A</source><capacity>1</capacity>'
        '<capacity>2</capacity></link></network>',
        encoding="utf-8",
    )
    empty_source = root / "empty_source.xml"
    empty_source.write_text(
        '<network><node id="A"><x>1</x><y>2</y></node>'
        '<link id="AA"><source/><target>A</target><capacity>1</capacity>'
        '<capacity>2</capacity></link></network>',
        encoding="utf-8",
    )
    invalid_utf8 = root / "invalid_utf8.xml"
    invalid_utf8.write_bytes(
        b'<network><node id="A"><x>\xff</x><y>2</y></node></network>'
    )
    embedded_nul = root / "embedded_nul.xml"
    embedded_nul.write_bytes(
        b'<network><node id="A"><x>1\x00</x><y>2</y></node></network>'
    )
    unknown_entity = root / "unknown_entity.xml"
    unknown_entity.write_text(
        '<network><meta>&bogus;</meta><node id="A"><x>1</x><y>2</y>'
        '</node></network>',
        encoding="utf-8",
    )
    invalid_numeric_entity = root / "invalid_numeric_entity.xml"
    invalid_numeric_entity.write_text(
        '<network><meta>&#0;</meta><node id="A"><x>1</x><y>2</y>'
        '</node></network>',
        encoding="utf-8",
    )
    unknown = root / "unknown.xml"
    unknown.write_text(
        '<network><node id="A"><x>1</x><y>2</y></node>'
        '<link id="AX"><source>A</source><target>X</target>'
        "<capacity>1</capacity><capacity>2</capacity></link></network>",
        encoding="utf-8",
    )
    missing = root / "does_not_exist.xml"

    cases = [
        ("missing", missing, "xml_parse_failure", "parse_xml", (OSError,)),
        ("malformed", malformed, "xml_parse_failure", "parse_xml", (Exception,)),
        ("missing_x", missing_x, "xml_schema_failure", "parse_xml", (IndexError,)),
        ("missing_y", missing_y, "xml_schema_failure", "parse_xml", (IndexError,)),
        ("missing_node_id", missing_node_id, "xml_schema_failure", "parse_xml", (KeyError,)),
        ("missing_link_id", missing_link_id, "xml_schema_failure", "parse_xml", (KeyError,)),
        ("empty_y", empty_y, "xml_schema_failure", "parse_xml", (AttributeError,)),
        ("one_capacity", one_capacity, "xml_schema_failure", "parse_xml", (IndexError,)),
        ("missing_source", missing_source, "xml_schema_failure", "parse_xml", (IndexError,)),
        ("missing_target", missing_target, "xml_schema_failure", "parse_xml", (IndexError,)),
        ("empty_source", empty_source, "xml_schema_failure", "parse_xml", (AttributeError,)),
        ("invalid_utf8", invalid_utf8, "xml_parse_failure", "parse_xml", (Exception,)),
        ("embedded_nul", embedded_nul, "xml_parse_failure", "parse_xml", (Exception,)),
        ("unknown_entity", unknown_entity, "xml_parse_failure", "parse_xml", (Exception,)),
        (
            "invalid_numeric_entity",
            invalid_numeric_entity,
            "xml_parse_failure",
            "parse_xml",
            (Exception,),
        ),
    ]
    count = 0
    for name, path, code, operation, error_types in cases:
        try:
            expect_python_error(lambda path=path: parse_records(path), error_types)
        except RuntimeError as error:
            raise RuntimeError(f"{name}: {error}") from error
        response = run_cpp(harness, "parse", str(path))
        try:
            require_cpp_error(response, "parse", code, operation)
        except RuntimeError as error:
            raise RuntimeError(f"{name}: {error}") from error
        count += 1

    python_target = root / "unknown_python.gml"
    expect_python_error(
        lambda: oracle_preprocess(module, "Unknown", unknown, python_target),
        (ValueError, TypeError),
    )
    response = run_cpp(harness, "materialize", "Unknown", str(unknown))
    require_cpp_error(response, "materialize", "unknown_endpoint", "materialize_graph")
    count += 1

    python_unchanged = root / "python_unchanged.gml"
    cpp_unchanged = root / "cpp_unchanged.gml"
    python_unchanged.write_bytes(b"sentinel")
    cpp_unchanged.write_bytes(b"sentinel")
    expect_python_error(
        lambda: oracle_preprocess(module, "Malformed", malformed, python_unchanged),
        (Exception,),
    )
    response = run_cpp(
        harness, "preprocess", "Malformed", str(malformed), str(cpp_unchanged)
    )
    require_cpp_error(response, "preprocess", "xml_parse_failure", "parse_xml")
    if python_unchanged.read_bytes() != b"sentinel":
        raise RuntimeError("Python parse failure changed target bytes")
    if bytes.fromhex(response["target_hex"]) != b"sentinel" or response["target_kind"] != "file":
        raise RuntimeError("C++ parse failure target side effect mismatch")
    count += 1

    valid = root / "writer_valid.xml"
    valid.write_text('<network><node id="A"><x>1</x><y>2</y></node></network>', encoding="utf-8")
    python_directory = root / "python_target_directory"
    cpp_directory = root / "cpp_target_directory"
    python_directory.mkdir()
    cpp_directory.mkdir()
    expect_python_error(
        lambda: oracle_preprocess(module, "Writer", valid, python_directory),
        (OSError,),
    )
    response = run_cpp(
        harness, "preprocess", "Writer", str(valid), str(cpp_directory)
    )
    require_cpp_error(response, "preprocess", "gml_write_failure", "write_gml")
    if response["target_kind"] != "directory" or not cpp_directory.is_dir():
        raise RuntimeError("C++ writer failure directory side effect mismatch")
    count += 1
    return count


def check_batches(harness: pathlib.Path, fixtures: list[Fixture], root: pathlib.Path) -> int:
    paths = [fixture.path for fixture in fixtures]
    paths = (paths * 4)[:16]
    expected = serialize_batch(parse_records(path) for path in paths)
    count = 0
    for workers in range(9):
        response = run_cpp(
            harness, "parse_batch", str(workers), *(str(path) for path in paths)
        )
        actual, gml = require_cpp_ok(response, "parse_batch")
        if gml is not None:
            raise RuntimeError("parse_batch unexpectedly returned GML")
        assert_exact_bytes(f"batch/w{workers}", expected, actual)
        count += 1

    missing = root / "batch_missing.xml"
    malformed = root / "batch_malformed.xml"
    malformed.write_text("<network><node></network>", encoding="utf-8")
    error_paths = [fixtures[0].path, missing, malformed, fixtures[1].path]
    for workers in (0, 1, 2, 4, 8):
        response = run_cpp(
            harness,
            "parse_batch",
            str(workers),
            *(str(path) for path in error_paths),
        )
        require_cpp_error(
            response,
            "parse_batch",
            "xml_parse_failure",
            "parse_xml",
            input_index="1",
        )
        if pathlib.Path(bytes.fromhex(response["path_hex"]).decode()) != missing:
            raise RuntimeError(f"batch/w{workers} lowest-error path mismatch")
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", type=pathlib.Path, required=True)
    parser.add_argument("--python-source", type=pathlib.Path, required=True)
    parser.add_argument("--brain-xml", type=pathlib.Path, required=True)
    parser.add_argument("--brain-gml", type=pathlib.Path, required=True)
    parser.add_argument("--json-output", type=pathlib.Path)
    args = parser.parse_args()

    import networkx as nx

    if nx.__version__ != EXPECTED_NETWORKX:
        raise RuntimeError(f"NetworkX version drift: {nx.__version__}")
    module = compare_dataset_core.load_oracle(args.python_source)
    brain_xml_bytes = args.brain_xml.read_bytes()
    brain_gml_bytes = args.brain_gml.read_bytes()
    with tempfile.TemporaryDirectory(prefix="virne_dataset_xml_diff_") as text_root:
        root = pathlib.Path(text_root)
        fixtures = write_fixtures(root)
        fixtures.append(Fixture("brain", args.brain_xml, "Brain"))
        compatible_cases = check_valid_cases(module, args.harness, fixtures, root)
        error_cases = check_errors(module, args.harness, root)
        batch_cases = check_batches(args.harness, fixtures, root)

        python_brain_target = root / "brain_fixture_check.gml"
        oracle_preprocess(module, "Brain", args.brain_xml, python_brain_target)
        generated_brain = python_brain_target.read_bytes()
        normalized_checked_brain = brain_gml_bytes.replace(b"\r\n", b"\n")
        assert_exact_bytes("Brain.gml checked fixture", normalized_checked_brain, generated_brain)

    total = compatible_cases + error_cases + batch_cases + 1
    print(f"dataset XML differential: PASS ({total}/{total} cases)")
    print(f"networkx={nx.__version__}")
    print(f"brain_nodes=161 brain_input_links=332 brain_simple_edges=166")
    print(f"brain_xml_sha256={hashlib.sha256(brain_xml_bytes).hexdigest()}")
    print(f"brain_gml_generated_sha256={hashlib.sha256(generated_brain).hexdigest()}")
    if args.json_output:
        artifact = {
            "batch_cases": batch_cases,
            "brain_gml_checked_raw_sha256": hashlib.sha256(brain_gml_bytes).hexdigest(),
            "brain_gml_generated_bytes": len(generated_brain),
            "brain_gml_generated_sha256": hashlib.sha256(generated_brain).hexdigest(),
            "brain_xml_bytes": len(brain_xml_bytes),
            "brain_xml_sha256": hashlib.sha256(brain_xml_bytes).hexdigest(),
            "compatibility_cases": compatible_cases,
            "error_cases": error_cases,
            "harness_sha256": hashlib.sha256(args.harness.read_bytes()).hexdigest(),
            "networkx": nx.__version__,
            "python_source_sha256": compare_dataset_core.SOURCE_SHA256,
            "status": "PASS",
            "total_cases": total,
        }
        args.json_output.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
