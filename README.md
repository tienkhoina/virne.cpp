# Nguyên Tắc Thiết Kế VirneCpp

## Bắt buộc đọc trước khi phát triển tiếp

Mọi thay đổi production phải đọc các tài liệu canonical sau theo đúng
thứ tự trước khi viết code:

1. [`DEPENDENCIES.md`](DEPENDENCIES.md) — version pin, dependency chỉ trong
   `libs/`, memory-layout hack và quy trình nâng cấp.
2. [`graph/API.md`](graph/API.md) — Graph/DiGraph API, ngữ nghĩa NetworkX,
   hot-path contract và stability boundary.
3. [`random/README.md`](random/README.md) — toàn bộ Random API, fixed-width
   contract, state consumption và oracle CPython/NumPy.
4. [`benchmarks/README.md`](benchmarks/README.md) và
   [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) — cách chạy gate và
   baseline đã ghi nhận.
5. [`API_MUST_BUILD.md`](API_MUST_BUILD.md) — public-surface checklist bắt buộc
   compile, test và benchmark.

README này là bản định hướng/quick start. Khi có khác biệt, tài
liệu canonical theo từng subsystem và declaration trong public header là
nguồn sự thật.

Phiên bản và chính sách thư viện cục bộ được cố định tại
[`DEPENDENCIES.md`](DEPENDENCIES.md); các dependency C++ chỉ được đặt trong
`libs/` và link thủ công qua CMake. `libs/` không được Git track; clone
mới phải tái tạo nó bằng archive/version trong
[`DEPENDENCIES.md`](DEPENDENCIES.md) và checksum trong
[`DEPENDENCIES.sha256`](DEPENDENCIES.sha256); tuyệt đối không fallback sang
thư viện OS hay Conda.

Các memory-layout hack được chấp nhận có baseline riêng: Boost 1.85.0 cho
graph và GCC 11.4.0/libstdc++ 11 cho direct Random output. Rủi ro, fallback và
quy trình nâng cấp nằm trong [`DEPENDENCIES.md`](DEPENDENCIES.md),
[`graph/API.md`](graph/API.md) và [`random/README.md`](random/README.md).

## Hiệu năng là ưu tiên số 1

Mọi quyết định thiết kế phải hướng tới giảm chi phí runtime của solver và simulator.

### 1. Bắt buộc resolve một lần trước hot loop

Public API vẫn cho phép tra cứu bằng:

```text
string
hash map
registry
yaml
```

ở boundary/cấu hình. Mọi vòng lặp theo node, edge, neighbor, path,
candidate, source hoặc sample **MUST** resolve tên thành ID đúng một lần
trước khi vào vòng lặp. Trong hot loop **MUST NOT** gọi `attr_id`,
`at("...")`, `find("...")`, `contains("...")`, YAML lookup hay string/hash
lookup tương đương.

Sau khi chạy:

```text
string -> id
```

và chỉ làm việc với ID. Callback/predicate được gọi lặp lại
**MUST** capture ID đã resolve; thuật toán lồng nhau **MUST** truyền ID
xuống helper thay vì resolve lại theo mỗi source/spur/candidate. Chuỗi chỉ
được giữ ở public/config/YAML/GML boundary.

---

### 2. Thuộc tính tĩnh là field

Không:

```cpp
solution["revenue"]
network["num_nodes"]
```

Mà:

```cpp
solution.revenue
network.num_nodes
```

Ưu tiên đọc thẳng bộ nhớ.

---

### 3. Thuộc tính động dùng ID

Không:

```cpp
attrs["cpu"]
```

Trong hot loop dùng:

```cpp
attrs.at(CPU_ID)
```

Mọi thuộc tính phải được ánh xạ sang ID càng sớm càng tốt.

---

### 4. Hot loop không dùng hash map

Thứ tự ưu tiên:

```text
Field
>
Array
>
Vector
>
ID Lookup
>
Hash Map
>
String Lookup
```

Hash map/string vẫn hợp lệ ở public boundary; resolve sang ID trước vòng lặp
nhạy hiệu năng.

---

### 5. Cache các giá trị dùng nhiều

Các giá trị truy cập thường xuyên:

```text
cpu
bw
degree
centrality
...
```

phải được lưu trực tiếp thay vì tính hoặc tra cứu lặp lại.

---

### 6. Port kiến trúc, không port workaround của Python

Giữ:

```text
Thuật toán
Cấu trúc dữ liệu
Interface
Kiến trúc
```

Không cố giữ:

```text
OmegaConf workaround
NetworkX workaround
Dynamic dict workaround
```

nếu chúng làm giảm hiệu năng.

---

## Triết lý

```text
Chuyển mọi thứ động thành tĩnh càng sớm càng tốt.

Trả giá một lần khi khởi tạo.
Không trả giá lần thứ hai trong runtime.
```


# Config

## Mục đích

Load YAML, override từ CLI và truy cập cấu hình theo path.

Chi tiết composition, interpolation, kiểu override và phạm vi tương thích
Hydra được khóa tại [config/README.md](config/README.md).

---

## ConfigLoader::load

```cpp
Config cfg =
    ConfigLoader::load(
        "setting/main.yaml");
```

---

## Config::get<T>

```cpp
cfg.get<int>(
    "experiment.seed");

cfg.get<std::string>(
    "solver.solver_name");
```

---

## override_parser::apply

```cpp
override_parser::apply(
    cfg,
    "experiment.seed=123");
```

Ví dụ:

```cpp
override_parser::apply(
    cfg,
    "solver.solver_name=my_solver");
```

---

## Config::save

```cpp
cfg.save(
    "after.yaml");
```

---

## Config::root

```cpp
cfg.root()
```

Dùng để:

```cpp
dump
debug
inspect
```

---

## CLI Override

Ví dụ:

```bash
./main \
solver.solver_name=my_solver \
experiment.seed=123
```

Runtime output defaults to one rate-limited Python-style progress bar per epoch
plus a compact JSON summary. A redirected stderr receives only the final line
of each epoch; use Docker `exec -t`/`-it` for live in-place updates. Use
`++native.output.report=full` only for differential or debugging,
`++native.output.report=none` to suppress stdout, and
`++native.progress.enabled=false` to disable progress. Config-group files are
selected without the `.yaml` suffix. Native and Python-compatible spellings
are both accepted, for example `p_net_setting=p_net_setting_multi_resource`
or `+p_net_setting=p_net_setting_multi_resource` together with the matching
`v_sim_setting` group. The Python-compatible `+group=option` spelling merges
the option over the existing default group; the native `group=option` spelling
replaces it. Each live progress frame is one buffered write plus one explicit
flush. Its bar and labels automatically compact to `TTY columns - 1`, so a
Docker/PowerShell terminal updates one physical row without auto-wrap;
redirected output contains only the final newline-terminated frame.

Before setup, INFO logging emits `Use <solver> ...` and the full composed,
interpolation-resolved config through the configured Logger backends, matching
Python's `Config:` startup log. `logger.backends=[]` or a level above INFO
suppresses it and skips the unused serialization. The config is serialized
once at the cold boundary; request loops continue to use typed fields and
numeric IDs only. If `experiment.run_id=auto`, the emitted and saved config
contains the concrete generated run ID.

The cold adapter also materializes the same `simulation.*` and
`rl.feature_constructor.*` derived fields as Python. The `Config:` block and
saved `config.yaml` exclude Hydra internals and the C++-only `native` subtree,
so their application tree is directly comparable with Python. Native workers,
progress and output controls are logged as `Native config` and saved separately
as `native_config.yaml`.

Tương đương:

```yaml
solver:
  solver_name: my_solver

experiment:
  seed: 123
```

---

# Random

`PyRandom` và `NumpyRandomState` là hai stream độc lập, tương thích bit/state
với subset CPython 3.10 và NumPy `RandomState` 1.26.4 mà Virne sử dụng. Ví dụ
khởi tạo stream CPython-compatible:

```cpp
PyRandom rng(42);
```

Toàn bộ chữ ký/default, công thức `uniform`, validation, fixed-width domain,
quy tắc state consumption, memory-layout hack và oracle nằm duy nhất tại
[`random/README.md`](random/README.md). Phát triển Random **MUST** đọc tài liệu
canonical đó; README gốc không lặp lại một inventory dễ lệch phiên bản.

---

# Progress

Quy tắc throttle, TTY/non-TTY và benchmark hot path được ghi tại
[progress/README.md](progress/README.md).

## Khởi tạo

```cpp
Progress p(
    1000,
    "Running with nea_rank in epoch 0");
```

## Cập nhật metric

```cpp
p.set_postfix(
{
    {"ac", 0.95},
    {"r2c", 0.85},
    {"reward", 1.23},
    {"loss", 0.42},
    {"inservice", 16}
});
```

## Cập nhật tiến độ

```cpp
p.update(
    current);
```

## Kết thúc

```cpp
p.finish();
```

## Ví dụ

```cpp
Progress p(
    1000,
    "Running with nea_rank in epoch 0");

for (size_t i = 0;
     i < 1000;
     ++i)
{
    p.set_postfix(
    {
        {"ac", 0.95},
        {"r2c", 0.85},
        {"inservice", 16}
    });

    p.update(
        i + 1);
}

p.finish();
```

## Kiểu dữ liệu hỗ trợ

```cpp
int
long long
double
std::string
```
---

# Graph Library Knowledge Base

## Mục tiêu

Thư viện đồ thị C++ lấy cảm hứng từ NetworkX.

Được xây dựng trên Boost Graph Library (BGL) nhưng cung cấp API đơn giản hơn cho:

* Thuật toán đồ thị
* Network Simulation
* Virtual Network Embedding (VNE)
* Reinforcement Learning (RL)
* Các bài toán tối ưu trên đồ thị

Các loại đồ thị hiện tại:

```text
Graph   (undirected)
DiGraph (directed)
```

Hợp đồng API đầy đủ, ngữ nghĩa có hướng, version pin và quy tắc hot-path được
ghi tại [`graph/API.md`](graph/API.md). Bộ đối chiếu NetworkX/benchmark được
ghi tại [`benchmarks/README.md`](benchmarks/README.md).

---

# Cấu trúc tổng thể

```text
graph/
├── attribute.h
├── graph.h
├── graph.cpp
├── graph_types.h
├── sparse_matrix.h
│
├── algorithms/
├── nx/
├── generators/
├── io/
├── random/
├── config/
└── views/
```

# Kiểu dữ liệu

## Vertex

ID của node.

Luôn liên tiếp:

```cpp
0 .. num_nodes()-1
```

---

## Edge

Descriptor của Boost Graph.

Được sử dụng trong các thuật toán nội bộ.

Không nên cache lâu dài sau các thao tác chỉnh sửa đồ thị.

---

## Edge ID

ID ổn định của cạnh, biểu diễn trực tiếp bằng `uint32_t` trong API hiện tại.

```cpp
uint32_t id =
    g.edge_id(edge);
```

Mỗi cạnh được gán một ID duy nhất khi được tạo.

ID không thay đổi trong suốt vòng đời graph.

ID không được tái sử dụng sau khi cạnh bị xóa.

---

## AttrValue

Các kiểu dữ liệu hỗ trợ:

```cpp
int64_t
double
bool
std::string
AttrListPtr
AttrObjectPtr
```

Scalar được giữ inline cho hot path. `AttrListPtr`/`AttrObjectPtr` dành cho
metadata lồng nhau (ví dụ GML/position); thuật toán hiệu năng cao vẫn phải
resolve scalar attribute sang `AttrId` trước vòng lặp.

---

## AttrId

ID nội bộ của thuộc tính.

```cpp
using AttrId =
    uint32_t;
```

Ví dụ:

```text
cpu -> 0
gpu -> 1
bw  -> 2
```

ID được sinh lần đầu khi thuộc tính xuất hiện.

---

## AttributeRegistry

Graph quản lý toàn bộ thuộc tính thông qua registry.

```cpp
AttrId attr_id(
    std::string_view name) const;

std::string_view attr_name(
    AttrId id) const;
```

Ví dụ:

```cpp
AttrId cpu =
    g.attr_id("cpu");

AttrId bw =
    g.attr_id("bw");
```

Mỗi lần gọi API bằng chuỗi vẫn cần lookup/hash. Vì vậy hãy resolve tên một lần
thành `AttrId`, sau đó dùng `at`, `find` hoặc `set` với ID trong hot loop.

---

## AttrMap

Node và edge lưu dữ liệu bằng mảng.

```cpp
AttrId
    ->
slot
```

Ví dụ:

```text
cpu -> slot[0]
gpu -> slot[1]
bw  -> slot[2]
```

Không sử dụng:

```cpp
unordered_map<
    std::string,
    AttrValue>
```

trong runtime.

---

# Graph API

## Graph

### Tạo đồ thị

```cpp
Graph g;
```

`DiGraph` cung cấp cùng tập chữ ký public cho đồ thị có hướng:

```cpp
DiGraph g;

DiEdge uv = g.add_edge(u, v);
bool reverse_exists = g.has_edge(v, u);
```

Có thể dựng trực tiếp từ edge list, edge list kèm attribute, hoặc ma trận kề
dày hình vuông:

```cpp
Graph g_from_edges(edge_list);
DiGraph dg_from_matrix(adjacency, "weight");
```

Constructor ma trận tạo đủ node theo số hàng, bỏ ô bằng `0`, giữ self-loop và
lưu ô khác `0` dưới tên weight đã chọn. `add_edge(u, v)` tự tạo mọi chỉ số node
liên tiếp đến `max(u, v)`; `add_nodes_from` chỉ nhận phần mở rộng liên tiếp.

`neighbors(v)` và `neighbors_fast(v)` của `DiGraph` chỉ duyệt successor/cung
đi ra. `degree(v)` là tổng in-degree và out-degree.

---

### Thêm node

```cpp
Vertex v =
    g.add_node();
```

---

### Thêm cạnh

```cpp
Edge e =
    g.add_edge(
        u,
        v);
```

Gắn attribute ngay khi thêm cạnh:

```cpp
Edge e = g.add_edge(u, v, attrs);
```

---

### Xóa cạnh

```cpp
g.remove_edge(
    u,
    v);
```

---

### Kiểm tra cạnh tồn tại

```cpp
bool ok =
    g.has_edge(
        u,
        v);
```

---

### Lấy cạnh

```cpp
Edge e =
    g.edge(
        u,
        v);
```

---

### Số node

```cpp
size_t n =
    g.num_nodes();

size_t nx_n =
    g.number_of_nodes();
```

---

### Số cạnh

```cpp
size_t m =
    g.num_edges();

size_t nx_m =
    g.number_of_edges();
```

---

### Degree

```cpp
size_t deg =
    g.degree(v);
```

---

### Lấy source / target

```cpp
Vertex u =
    g.source(e);

Vertex v =
    g.target(e);
```

---

### Duyệt node

```cpp
auto [it, end] =
    g.nodes();

for (; it != end; ++it)
{
    Vertex v =
        *it;
}
```

---

### Duyệt cạnh

```cpp
auto [it, end] =
    g.edges();

for (; it != end; ++it)
{
    Edge e =
        *it;
}
```

`g.edges()` là range descriptor dành cho thuật toán thấp tầng. Public
`g.edge_view()` duyệt trực tiếp cặp endpoint `(u, v)` theo thứ tự NetworkX;
`edge_view().descriptors()` là escape hatch khi thực sự cần descriptor.

```cpp
for (auto [u, v] : g.edge_view())
{
}
```

---

### Duyệt neighbor

```cpp
auto [it, end] =
    g.neighbors(v);

for (; it != end; ++it)
{
    Vertex u =
        *it;
}
```

---

### Fast neighbor access

```cpp
const auto& out =
    g.neighbors_fast(v);

for (const auto& edge : out)
{
    Vertex u =
        edge.get_target();
}
```

---

### Edge id

```cpp
uint32_t id =
    g.edge_id(e);
```

---

### Edge lookup theo id

```cpp
Edge e =
    g.edge_by_id(id);
```

---

### Endpoint lookup theo id

```cpp
auto [u, v] =
    g.edge_endpoints(id);
```

---

### Raw boost graph

```cpp
BGLGraph& bg =
    g.raw();
```

`raw()` chỉ là escape hatch đọc cho hot path đã profile. Không mutate qua BGL:
việc đó bỏ qua registry, edge ID, simple-edge invariant và bookkeeping. Boost
1.85.0 cùng layout nội bộ được pin tại [`graph/API.md`](graph/API.md); mọi
boundary nên dùng API checked, còn vòng lặp nóng dùng dense index,
`neighbors_fast()` và `AttrId`.

---

# Attributes

## Node attributes

### Gán

```cpp
g.node_attrs(v)["cpu"] =
    int64_t(100);

g.node_attrs(v)["name"] =
    std::string("A");
```

---

### Đọc

```cpp
int64_t cpu =
    std::get<int64_t>(
        g.node_attrs(v)
            .at("cpu"));
```

---

### Tìm

```cpp
auto value =
    g.node_attrs(v)
        .find("cpu");

if (value != nullptr)
{
}
```

---

### Kiểm tra tồn tại

```cpp
bool ok =
    g.node_attrs(v)
        .contains("cpu");
```

---

## Edge attributes

### Gán

```cpp
g.edge_attrs(e)["weight"] =
    10.0;
```

---

### Đọc

```cpp
double w =
    std::get<double>(
        g.edge_attrs(e)
            .at("weight"));
```

---

### Tìm

```cpp
auto value =
    g.edge_attrs(e)
        .find("weight");
```

---

# Attribute Registry

## Tạo id cho attribute

```cpp
AttrId cpu =
    g.attr_id("cpu");

AttrId gpu =
    g.attr_id("gpu");
```

---

## Lấy tên từ id

```cpp
std::string_view name =
    g.attr_name(cpu);
```

---

## Truy cập bằng AttrId

```cpp
AttrId cpu =
    g.attr_id("cpu");

auto value =
    g.node_attrs(v)
        .find(cpu);
```

---

## Duyệt toàn bộ attribute hiện có

```cpp
for (AttrId id :
     attrs.attribute_ids())
{
}
```

---

# Generators

## Waxman

```cpp
WaxmanConfig cfg;

cfg.num_nodes = 100;
cfg.alpha = 0.4;
cfg.beta = 0.2;
cfg.seed = 42;

Graph g =
    WaxmanGenerator::
        generate(cfg);
```

---

## GML

```cpp
Graph g =
    GmlLoader::load(
        "graph.gml");
```

```cpp
DiGraph g =
    GmlLoader::load_directed(
        "directed.gml");
```

---

# Connectivity

## Kiểm tra liên thông

```cpp
bool ok =
    nx::is_connected(g);
```

---

# BFS

## Chạy BFS

```cpp
auto result =
    bfs(
        g,
        source);
```

---

### Distance

```cpp
result.distance[v];
```

---

### Predecessor

```cpp
result.predecessor[v];
```

---

# Bidirectional BFS

## Đường đi ngắn nhất không trọng số

```cpp
auto result =
    bidirectional_bfs(
        g,
        source,
        target);
```

---

### Found

```cpp
result.found;
```

---

### Distance

```cpp
result.distance;
```

---

### Path

```cpp
result.path;
```

---

# Dijkstra

## Single source

```cpp
auto result =
    dijkstra(
        g,
        source,
        "weight");
```

---

### Distance

```cpp
result.distance[v];
```

---

### Predecessor

```cpp
result.predecessor[v];
```

---

## Có blacklist

```cpp
VertexSet banned_vertices;

EdgeSet banned_edges;

auto result =
    dijkstra(
        g,
        source,
        banned_vertices,
        banned_edges,
        "weight");
```

---

## Path reconstruction

```cpp
auto path =
    build_path(
        result,
        source,
        target);
```

---

## Edge cost

```cpp
double w =
    edge_cost(
        g,
        u,
        v,
        "weight");
```

---

## Path cost

```cpp
double cost =
    path_cost(
        g,
        path,
        "weight");
```

---

## Prefix cost

```cpp
auto prefix =
    path_prefix_costs(
        g,
        path,
        "weight");
```

---

# Bidirectional Dijkstra

```cpp
auto result =
    bidirectional_dijkstra(
        g,
        source,
        target,
        VertexSet{},
        EdgeSet{},
        "weight");
```

---

### Found

```cpp
result.found;
```

---

### Cost

```cpp
result.cost;
```

---

### Path

```cpp
result.path;
```

---

# Floyd Warshall

```cpp
DistanceMatrix dist =
    floyd_warshall(
        g,
        "weight");
```

---

### Truy cập

```cpp
double d =
    dist(i, j);
```

---

# Yen K Shortest Paths

```cpp
auto paths =
    yen_k_shortest_paths(
        g,
        source,
        target,
        k,
        "weight");
```

---

### PathResult

```cpp
paths[i].path;

paths[i].cost;
```

---

# NetworkX-Compatible C++ Subset

Các hàm `nx::*` dưới đây cố định API C++ hiện có và đối chiếu
giá trị với NetworkX 3.4.2 trong phạm vi được hỗ trợ. Chúng không sao
chép các chữ ký Python động, option chỉ có trên Python hay biến thể
kiểu trả về. Chi tiết default và edge-case được đóng băng tại
[`graph/API.md`](graph/API.md).

## shortest_path_length

```cpp
size_t d =
    nx::shortest_path_length(
        g,
        source,
        target);
```

---

## shortest_path

```cpp
auto path =
    nx::shortest_path(
        g,
        source,
        target);
```

---

## all_shortest_paths

```cpp
auto paths =
    nx::all_shortest_paths(
        g,
        source,
        target,
        std::nullopt);
```

Đối số thứ tư là `std::optional<std::string_view>`: `std::nullopt` tương ứng
`weight=None`, chuỗi tương ứng tên edge attribute.

---

## single_source_shortest_path_length

```cpp
auto dist =
    nx::single_source_shortest_path_length(
        g,
        source,
        4.0);
```

`cutoff` là `std::optional<double>` mặc định `std::nullopt`, đúng thứ tự public
của NetworkX.

---

## dijkstra_path

```cpp
auto path =
    nx::dijkstra_path(
        g,
        source,
        target,
        "weight");
```

`weight` có kiểu `std::optional<std::string_view>`, mặc định là `"weight"`;
dùng `std::nullopt` để biểu diễn chính xác `NetworkX weight=None`.

---

## dijkstra_path_length

```cpp
double cost =
    nx::dijkstra_path_length(
        g,
        source,
        target,
        "weight");
```

---

## single_source_dijkstra_path_length

```cpp
auto dist =
    nx::single_source_dijkstra_path_length(
        g,
        source,
        4.0,
        "weight");
```

Chữ ký public giữ thứ tự NetworkX:
`(graph, source, cutoff = std::nullopt, weight = "weight")`. Muốn bỏ cutoff
nhưng vẫn truyền tên weight, dùng `std::nullopt` ở đối số thứ ba; overload ba
đối số nhận `string_view` cũ vẫn được giữ để tương thích mã Virne hiện hữu.

### bfs_nx

Triển khai BFS theo phong cách NetworkX, tối ưu cho các bài toán chỉ cần khoảng cách ngắn nhất.

Khác với BFS tổng quát, `bfs_nx` duyệt theo từng mức (level) và chỉ lưu khoảng cách của các đỉnh đã thăm. Thuật toán mô phỏng trực tiếp cách `networkx.single_source_shortest_path_length()` hoạt động.

**Đặc điểm**

* Duyệt theo từng lớp (level-by-level BFS)
* Không lưu predecessor
* Dừng sớm khi đã thăm toàn bộ đồ thị
* Chỉ trả về khoảng cách của các đỉnh reachable
* Tối ưu cho các phép đo centrality dựa trên khoảng cách

**Hiệu năng**

`bfs` và `bfs_nx` trả về cấu trúc dữ liệu khác nhau nên không dùng một con số
cũ để khẳng định API nào luôn nhanh hơn API kia. Số đo Release hiện hành cho
cả hai, cùng toàn bộ thuật toán Graph/DiGraph, được lưu tại
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

**Sử dụng cho**

* `single_source_shortest_path_length`
* `closeness_centrality`
* Các bài toán chỉ cần khoảng cách ngắn nhất

**Không phù hợp cho**

* Khôi phục đường đi
* `shortest_path`
* Các thuật toán cần predecessor


---

## nx::floyd_warshall

```cpp
auto dist =
    nx::floyd_warshall(
        g,
        "weight");
```

---

## shortest_simple_paths

Lazy, theo chữ ký NetworkX `(graph, source, target, weight=None)`:

```cpp
auto generator =
    nx::shortest_simple_paths(
        g,
        source,
        target,
        std::optional<std::string_view>{"weight"});
```

Convenience API của Virne để materialize `k` path đầu:

```cpp
auto paths =
    nx::shortest_simple_paths(
        g,
        source,
        target,
        k,
        "weight");
```

---

# WeightCache

## Build

```cpp
WeightCache cache(
    g);
```

Với `DiGraph`, dùng snapshot tương ứng:

```cpp
DiWeightCache cache(g);
```

---

## Attribute id

```cpp
AttrId weight =
    cache.attribute_id(
        "weight");
```

---

## Fast edge lookup

```cpp
double w =
    cache.value(
        edge,
        weight);
```

---

## Fast vertex lookup

```cpp
double w =
    cache.value(
        u,
        v,
        weight);
```

# Attributes

## get_node_attributes

```cpp
auto attrs =
    nx::get_node_attributes(
        g,
        "cpu");
```

```cpp
AttrId cpu =
    g.attr_id(
        "cpu");

auto attrs =
    nx::get_node_attributes(
        g,
        cpu);
```

---

## set_node_attributes

```cpp
nx::set_node_attributes(
    g,
    values,
    "cpu");
```

```cpp
AttrId cpu =
    g.attr_id(
        "cpu");

nx::set_node_attributes(
    g,
    values,
    cpu);
```

Entry có node index không tồn tại được bỏ qua giống NetworkX.

---

## get_edge_attributes

```cpp
auto attrs =
    nx::get_edge_attributes(
        g,
        "bw");

for (const auto& [edge, value] : attrs)
{
    Vertex u = edge.first;
    Vertex v = edge.second;
}
```

```cpp
AttrId bw =
    g.attr_id(
        "bw");

auto attrs =
    nx::get_edge_attributes(
        g,
        bw);
```

Cả overload theo tên và `AttrId` đều duyệt key endpoint `(u, v)` theo thứ tự
edge của NetworkX. `attrs.at(edge_id)`, `find(edge_id)` và `contains(edge_id)`
là accessor tương thích cho code đã resolve stable ID; không dùng lookup
string trong hot loop.

---

## set_edge_attributes

```cpp
std::unordered_map<
    EdgeEndpoints,
    AttrValue,
    nx::EdgeEndpointHash> values;

values[{u, v}] = 100.0;

nx::set_edge_attributes(
    g,
    values,
    "bw");
```

```cpp
AttrId bw =
    g.attr_id(
        "bw");

std::unordered_map<uint32_t, AttrValue> values_by_id;
values_by_id[edge_id] = 100.0;

nx::set_edge_attributes(
    g,
    values_by_id,
    bw);
```

Setter endpoint bỏ qua cặp không tồn tại giống NetworkX. Overload
`unordered_map<uint32_t, AttrValue>` và `set_edge_attributes_by_id` dành cho
đường indexed; ID thiếu/stale là lỗi.

## nx.adjacency_matrix

Tạo ma trận kề dạng `SparseMatrix`.

```cpp
SparseMatrix A =
    nx::adjacency_matrix(
        graph);
```

---

## nx.attr_sparse_matrix

Tạo ma trận thưa từ thuộc tính cạnh.

Theo tên thuộc tính:

```cpp
SparseMatrix A =
    nx::attr_sparse_matrix(
        graph,
        "weight");
```

Theo `AttrId`:

```cpp
AttrId weight =
    graph.attr_id(
        "weight");

SparseMatrix A =
    nx::attr_sparse_matrix(
        graph,
        weight);
```

`adjacency_matrix` lỗi trên graph rỗng và dùng `1.0` khi thiếu `weight`.
`attr_sparse_matrix` chỉ dùng default `1.0` cho attribute có tên `"weight"`;
thiếu custom attribute trên bất kỳ edge nào là lỗi. COO/CSR được sắp theo
row/column giống SciPy để kiểm tra thứ tự đầu ra chính xác.

---

## SparseMatrix

Thêm phần tử:

```cpp
A.add(
    row,
    col,
    value);
```

Số phần tử khác 0:

```cpp
size_t nnz =
    A.nnz();
```

Chuyển sang CSR:

```cpp
CSRMatrix csr =
    A.to_csr();
```

---

## CSRMatrix

Số phần tử khác 0:

```cpp
size_t nnz =
    csr.nnz();
```

Truy cập dữ liệu:

```cpp
csr.row_ptr

csr.col_idx

csr.values
```

## nx.degree_centrality

```cpp
auto scores =
    nx::degree_centrality(
        graph);
```

---

## nx.eigenvector_centrality

```cpp
auto scores =
    nx::eigenvector_centrality(
        graph);
```

Tùy chỉnh:

```cpp
auto scores =
    nx::eigenvector_centrality(
        graph,
        100,
        1e-6);
```

---

## nx.closeness_centrality

```cpp
auto scores =
    nx::closeness_centrality(
        graph);
```

---

## nx.betweenness_centrality

```cpp
auto scores =
    nx::betweenness_centrality(
        graph);
```

Theo thuộc tính trọng số cạnh:

```cpp
auto scores =
    nx::betweenness_centrality(
        graph,
        "weight");
```

---

## NodeScores

Các API centrality trả về:

```cpp
using NodeScores =
    std::vector<double>;
```

Truy cập điểm của node:

```cpp
double score =
    scores[node];
```
## write_gml

Lưu đồ thị ra file GML.

```cpp
nx::write_gml(
    graph,
    "graph.gml");
```
## CSV

### Create DataFrame

```cpp
csvio::DataFrame df;

df.columns = {
    "node",
    "degree",
    "score"
};

df.rows = {
    {"0", "12", "0.85"},
    {"1", "8",  "0.44"}
};
```

### Write CSV

```cpp
csvio::write_csv(
    "result.csv",
    df);
```

### Read CSV

```cpp
auto df =
    csvio::read_csv(
        "result.csv");
```

### Print Table

```cpp
csvio::print_table(df);
```

### Metadata

```cpp
std::cout << df.nrows() << '\n';
std::cout << df.ncols() << '\n';
```
