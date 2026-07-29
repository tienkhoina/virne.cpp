# Results: non-Torch `virne.utils.dataset` XML/graph/GML — 2026-07-28

State: **XML/GRAPH/GML LEAF COMPLETE**. The same-named Python/Torch seed
facade remains outside this non-ML leaf.

## Locked identity and dependencies

- Original Python leaf: `virne/utils/dataset.py` at commit
  `d1ec1e4a20461fc9bad50612ad5026fd31e693a8`.
- Python source SHA-256:
  `269650EBCC373D7BDF79FA17346BD6F847973F17E60C1AC9BCAE7CFD97BF936F`;
  exact size 9,635 bytes.
- Final Release differential/benchmark harness SHA-256:
  `2D8608B6AB07E9A4147A0974A9247170F5B94A84BC1D1B3C3B6AC3AC75ECC766`.
- Final Release unit SHA-256:
  `A4212AA4D04CBBF6C4B57F51727A156BF2BF3A0B58F329A19C8610FE6309DB31`.
- Production source/header SHA-256:
  `F48D202701BCBCFC1883D33F1C14CB7EC1C0F97AFCC7A48D6DAD49D5EF6DB7EB`
  and
  `44B07CCFA4D22AC661848AEE20E176834DF1E44E5E31B3CA38E006D55BB0B3D5`.
- Differential/benchmark/sweep script SHA-256:
  `25250890D6C61A895C95CB0022712ADDD2EA19BD8166F3927B4E539E9A8132E5`,
  `48325E5F7DE3D67BC63A873E05C2B01B995F6835D4E8E023A08DDE54B7DDCFFF`,
  and
  `1EC1150BE9D755F32DC2F4F91BF50E33B438880E8AD165D6A77FD28526B4EFDF`.
- Differential artifact:
  `porting/results/dataset_xml_differential_2026-07-28.json`, SHA-256
  `60D454F887DC25822DBF14810770DC8E68FE932EB116E9C349CDE646C556BDE5`.
- Canonical timing artifact:
  `porting/results/dataset_xml_benchmark_2026-07-28.json`, SHA-256
  `CD1D12651076AE89B85FF2CC95821CD4B2F4584A8F5B0900FC7BE2E595DE4902`.
- Worker-sweep artifact:
  `porting/results/dataset_xml_worker_sweep_2026-07-28.json`, SHA-256
  `32981F4CFDDC6A44633DD3963F7C7B52881F704DBC2FBF9C6258A1EE288ED7B8`.
- Runtime: CPython 3.10.20, NetworkX 3.4.2, and GCC 11.4.0 on
  `Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.41`, affinity
  CPUs 0–7.

The production target is `vne_utils_dataset_xml`; the isolated verification
targets are `vne_dataset_xml_unit` and `vne_dataset_xml_harness`. No XML
package was added or rebuilt. The implementation reuses the pinned workspace
Boost 1.85 payload and its bundled RapidXML adapter, plus the frozen graph
library and `Threads::Threads`. Direct in-situ RapidXML parsing avoids the
second allocation tree of `boost::property_tree::read_xml`. Use of
`boost/property_tree/detail/rapidxml.hpp` is an explicit Boost-1.85
implementation pin and must be retested if Boost changes.

## Stable public API

The declarations below in `virne/utils/dataset_xml.h` are the integration
contract. Ordinary callers should use this document and
`porting/components/dataset.md`, not reopen the implementation.

```cpp
struct XmlNodeRecord {
    std::string label;
    std::string x;
    std::string y;
};

struct XmlEdgeRecord {
    std::string label;
    std::string source_label;
    std::string target_label;
    std::string capacity_st;
    std::string capacity_ts;
    std::string cost_st;
    std::string cost_ts;
};

struct ParsedXmlTopology {
    std::vector<XmlNodeRecord> nodes;
    std::vector<XmlEdgeRecord> edges;
};

struct XmlTopologyRequest {
    std::string topology_name;
    std::filesystem::path xml_source_path;
    std::filesystem::path gml_target_path;
};

ParsedXmlTopology parse_sndlib_xml(
    const std::filesystem::path& source_path);

std::vector<ParsedXmlTopology> parse_sndlib_xml_batch(
    const std::vector<std::filesystem::path>& source_paths,
    std::size_t workers = 0);

Graph materialize_xml_topology(
    std::string_view topology_name,
    const ParsedXmlTopology& topology);

Graph preprocess_xml(const XmlTopologyRequest& request);
```

The parser preserves minidom descendant document order and literal first-child
`.data` extraction; text, CDATA, comment, and processing-instruction children
all supply data and none is skipped. Missing node/link `id` attributes are
schema failures while explicit empty IDs are valid. It parses all node and edge
records before graph mutation, keeps XML values as strings, and maps duplicate
labels to the last dense node ID. The original capacity-as-cost bug is locked:
the first two capacity values populate both capacity and cost fields, while XML
`cost` elements are ignored.

Materialization returns one frozen undirected simple `Graph`.
Duplicate/reversed edges update the existing edge in input order and self-loops
remain valid. `preprocess_xml` performs parse, materialization, graph-name
assignment, and the dataset-local NetworkX-3.4.2-compatible GML write; it
returns only after a successful write. The generic frozen graph writer remains
unchanged because it is not byte-identical to this Python boundary.

Fixed XML schema fields are direct record members. Graph attributes `name`,
`label`, `x`, `y`, `source_label`, `target_label`, both capacities, and both
costs are resolved once to `AttrId` before their loops. Hot loops use direct
fields, dense `Vertex`, and `AttrMap::set(AttrId, ...)`. The genuinely dynamic
label table performs exactly one source and one target lookup per edge, then
carries native IDs. There is no repeated fixed-key string lookup in a parse,
materialization, worker, or GML loop.

## Typed errors and intentional native boundaries

Stable error mappings are:

- file/XML syntax, invalid XML characters/entities, and unsupported encodings:
  `xml_parse_failure/parse_xml`;
- missing required attributes/descendants or first data children:
  `xml_schema_failure/parse_xml`;
- missing endpoint label: `unknown_endpoint/materialize_graph`;
- other graph failures:
  `graph_materialization_failure/materialize_graph`;
- writer failures: `gml_write_failure/write_gml`.

`std::bad_alloc` propagates unchanged. Platform-native message details remain
diagnostic; the typed code, operation, path/index, output bytes, and side
effects are the stable differential boundary.

The following deliberate C++ restrictions are not compatibility failures:

- the public source boundary is a native filesystem path, not Python's
  arbitrary file-object/path-like/reflection surface;
- accepted declared encodings are UTF-8 and ISO-8859-1; other encodings and
  custom DTD entities are typed parse rejections;
- fixed GML `source` and `target` keys cannot be aliased by input attributes;
- `.gz` and `.bz2` targets are rejected before opening rather than adding a
  compression dependency, so an existing target remains unchanged.

The byte oracle is fresh NetworkX 3.4.2 output on Linux. The checked-out
`Brain.gml` has CRLF-normalized raw SHA-256
`B0651A388A904D6E45559388F56A7BFFE2207D4928450552003C567907783CC3`;
the required generated LF file has a different hash below. This newline
difference is platform representation, not graph-semantic drift.

## Exact differential

The direct-loaded pinned Python leaf passed **57/57 cases** against the final
Release harness: 24 compatibility cases, 18 typed-error cases, 14 batch cases,
and the real Brain fixture. The corpus covers UTF-8 with declaration/BOM,
ISO-8859-1, entities, comments, CDATA, whitespace, descendant order, Unicode
and GML escaping, duplicate labels, duplicate/reversed edges, self-loops, the
capacity-as-cost behavior, malformed/schema inputs, unknown endpoints, writer
failures and target side effects. Batch output order, worker invariance for
requests 0 through 8, and lowest-input-index failure selection are exact.

The real fixture gate records:

- `Brain.xml`: 132,090 bytes, SHA-256
  `2CBB55BBD979DDAE7696E57A3C70867BE51ABC82E20FE8E0BDBDE32CDC966F4B`;
- parsed/materialized result: 161 nodes, 332 XML link records, and 166 simple
  graph edges;
- semantic graph SHA-256
  `7A6F570A13634ABFA9C1CAE79526E2D91726442D61A169B6A0A37D5F29AABC9E`;
- generated LF GML: exactly 45,085 bytes, SHA-256
  `0F30D85C53ECC94461FE7DD00D627F1F91492428266F37528642F29EA2965A3F`.

Graph fields, attribute order/value, returned record order, exact GML bytes,
and the corresponding hashes are gates; no tolerance or semantic-only
normalization is used.

## Canonical Python/C++ timing

The canonical protocol used five warm-ups and 31 measured samples. Python/C++
order and worker requests were interleaved to limit scheduler/thermal bias.
File reads, allocations, public thread creation/join, parse, and
materialization are inside the relevant timer; fixture construction, process
startup, and checksum verification are outside. Every accepted sample first
passed its output-byte and checksum gate.

Times below are median/MAD/p95 milliseconds. `auto` means production
`workers=0`; the single-document and full-preprocess rows are sequential.

| Row | Python median / MAD / p95 ms | C++ median / MAD / p95 ms | Speedup |
|---|---:|---:|---:|
| Brain parse, one document, worker 1 | 37.983588 / 3.761744 / 91.955980 | 1.366041 / 0.159583 / 1.944058 | 27.806x |
| Brain parse, 16 documents, auto | 826.902155 / 27.782063 / 938.322247 | 5.533607 / 0.815866 / 9.735412 | 149.433x |
| Synthetic parse, 16 documents, auto | 1,687.222790 / 43.562446 / 1,783.207707 | 15.703109 / 2.354765 / 19.993751 | 107.445x |
| Brain `preprocess_xml`, inner repetitions 5 | 59.732092 / 3.245571 / 70.815933 | 1.580773 / 0.076870 / 1.883894 | 37.787x |

The artifact contains **28/28 passing rows**: three parse families across
worker requests 1, auto, and 2 through 8, plus full preprocess. Every C++
median beats Python. The literal all-row speedup range is **26.590x to
149.433x**; redundant multi-worker requests for the one-document case are
count-capped to one lane and differ only by measurement noise. Both
`performance_gate` and `worker_policy_gate` are `true`.

## Worker sweep and production policy

`parse_sndlib_xml_batch` is the only worker extension. Documents are
independent and read-only; pre-sized slots preserve input order and the lowest
failing input index. Explicit requests are capped by document count and Linux
affinity. If thread construction fails, every already-started thread is joined
before propagation. `preprocess_xml` has no worker argument because concurrent
target writes would change observable side-effect and error order.

The final sweep used both Brain and synthetic corpora, 2/4/8/16/32/64
documents, worker requests 1, auto, and 2 through 8, five warm-ups, and 31
samples: **108/108 rows PASS**. Auto beat sequential in all 12 family/size
policy rows and stayed within 25% of the best explicit request.

| Corpus / documents | Sequential C++ ms | Auto C++ ms | Auto vs sequential | Best explicit request / ms |
|---|---:|---:|---:|---:|
| Brain / 2 | 2.494858 | 2.059481 | 1.211x | 5 / 1.877746 |
| Brain / 4 | 4.529567 | 3.231338 | 1.402x | 6 / 2.760887 |
| Brain / 8 | 8.660844 | 4.823675 | 1.795x | 3 / 4.785987 |
| Brain / 16 | 17.246456 | 8.141974 | 2.118x | 4 / 6.669392 |
| Brain / 32 | 33.128698 | 12.888706 | 2.570x | 8 / 11.722548 |
| Brain / 64 | 68.374140 | 23.680844 | 2.887x | 8 / 21.671649 |
| Synthetic / 2 | 5.284167 | 3.552579 | 1.487x | 8 / 3.544905 |
| Synthetic / 4 | 9.801331 | 5.386115 | 1.820x | 7 / 5.067051 |
| Synthetic / 8 | 18.871069 | 8.360919 | 2.257x | 8 / 7.958698 |
| Synthetic / 16 | 39.248414 | 14.570607 | 2.694x | 8 / 14.670132 |
| Synthetic / 32 | 69.016528 | 26.553422 | 2.599x | 8 / 25.322138 |
| Synthetic / 64 | 131.921673 | 54.428822 | 2.424x | 7 / 47.071054 |

For requests wider than the document count, the table reports the requested
width stored in the artifact; execution remains count-capped. Automatic mode
reads the first input file size once as `representative bytes`, then applies:

| Documents | Representative bytes | Automatic lanes |
|---:|---:|---:|
| 0 | any | 0 |
| 1 | any | 1 |
| 2 | any | 2 |
| 3–7 | `< 196,608` / `>= 196,608` | 2 / 4 |
| 8 | `< 196,608` / `>= 196,608` | 3 / 4 |
| 9–23 | `< 196,608` / `>= 196,608` | 4 / 8 |
| 24 or more | any | 5 |

The selected lane count is still capped by document count and process
affinity. No path or string-policy lookup occurs inside a parse worker loop.

## Engineering gates

- Isolated Release target, unit, and exact harness: **PASS**.
- Strict production/unit/harness compilation with
  `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Werror`: **PASS**.
- AddressSanitizer with leak detection and UBSan with no recovery: **PASS**.
- ThreadSanitizer with ASLR disabled: **PASS**.
- Repeated Release unit stress, 100 iterations, and concurrent-process stress:
  **PASS**.
- Release verification constrained to one, two, and eight CPUs: **PASS**.
- Full Release build and repository CTest: **PASS 22/22**.
- Frozen graph/CSV/config/yaml-cpp/random integrity: **PASS**; those libraries
  were reused rather than rebuilt or modified.
- `git diff --check`: **PASS** after final documentation.

Result: the non-Torch dataset **XML/graph/GML leaf is COMPLETE on
2026-07-28**. Future integration starts from the stable API above and
`porting/components/dataset.md`; reopening production code is reserved for an
API ambiguity or a measured optimization that the documented contract cannot
resolve.
