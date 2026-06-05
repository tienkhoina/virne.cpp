# Config

## Mục đích

Load YAML, override từ CLI và truy cập cấu hình theo path.

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

Tương đương:

```yaml
solver:
  solver_name: my_solver

experiment:
  seed: 123
```

---

# Random

## Mục đích

Tương thích hành vi với:

```python
random.Random
```

Đảm bảo Python và C++ sinh cùng kết quả với cùng seed.

---

## Khởi tạo

```cpp
PyRandom rng(42);
```

---

## random

```cpp
double random();
```

Output:

```cpp
[0,1)
```

---

## uniform

```cpp
double uniform(
    double a,
    double b);
```

Output:

```cpp
[a,b]
```

---

## getrandbits32

```cpp
uint32_t getrandbits32();
```

---

## randrange

```cpp
uint64_t randrange(
    uint64_t stop);
```

Output:

```cpp
[0,stop)
```

---

## choice

```cpp
template<typename T>
T& choice(
    std::vector<T>& v);
```

---

## shuffle

```cpp
template<typename T>
void shuffle(
    std::vector<T>& v);
```

---

## API

```cpp
explicit PyRandom(
    uint64_t seed);

double random();

double uniform(
    double a,
    double b);

uint32_t getrandbits32();

uint64_t randrange(
    uint64_t stop);

template<typename T>
T& choice(
    std::vector<T>& v);

template<typename T>
void shuffle(
    std::vector<T>& v);
```

---

# Progress

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

Đồ thị hiện tại:

```text
Undirected Graph
```

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

## EdgeId

ID ổn định của cạnh.

```cpp
using EdgeId =
    uint32_t;
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
```

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
    std::string_view name);

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

Mỗi tên thuộc tính chỉ được hash một lần.

Sau đó toàn bộ hệ thống sử dụng AttrId.

---

## AttrStore

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
```

---

### Số cạnh

```cpp
size_t m =
    g.num_edges();
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

# NetworkX Compatible APIs

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

## single_source_shortest_path_length

```cpp
auto dist =
    nx::single_source_shortest_path_length(
        g,
        source);
```

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
        "weight");
```

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

Trên đồ thị 1000 đỉnh, ~56k cạnh:

```text
BFS thường   ~1543 ms
bfs_nx       ~450 ms
```

Nhanh hơn khoảng **3.4 lần** so với BFS tổng quát.

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

---

## get_edge_attributes

```cpp
auto attrs =
    nx::get_edge_attributes(
        g,
        "bw");
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

---

## set_edge_attributes

```cpp
nx::set_edge_attributes(
    g,
    values,
    "bw");
```

```cpp
AttrId bw =
    g.attr_id(
        "bw");

nx::set_edge_attributes(
    g,
    values,
    bw);
```

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
