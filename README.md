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

---

# Kiến trúc

## graph/

Chứa cấu trúc dữ liệu lõi.

Bao gồm:

```cpp
Graph
Vertex
Edge
AttrMap
```

Không chứa thuật toán.

---

## algorithms/

Chứa thuật toán hiệu năng cao.

Ví dụ:

```cpp
bfs
bidirectional_bfs
dijkstra
bidirectional_dijkstra
floyd_warshall
yen_k_shortest_paths
```

Không chứa API NetworkX.

---

## nx/

Lớp tương thích NetworkX.

Ví dụ:

```cpp
nx::shortest_path
nx::dijkstra_path
nx::is_connected
```

Được phép gọi:

```cpp
algorithms/*
```

Không được cài thuật toán tại đây.

---

## generators/

Sinh topology hoặc đọc topology.

Ví dụ:

```cpp
WaxmanGenerator
GmlLoader
```

---

## config/

Hệ thống cấu hình.

Tương tự:

```text
Hydra
OmegaConf
```

---

## random/

Bộ sinh số ngẫu nhiên tương thích:

```python
random.Random
```

---

# Kiểu dữ liệu

## Vertex

ID của node.

Luôn liên tiếp:

```cpp
0 .. num_nodes()-1
```

---

## Edge

Descriptor của cạnh.

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

## AttrMap

Map lưu thuộc tính node hoặc edge.

Ví dụ:

```cpp
g.node_attrs(v)["cpu"] =
    int64_t(100);

g.edge_attrs(e)["weight"] =
    10.0;
```

---

# Graph API

## Khởi tạo

```cpp
Graph g;
```

---

## Node

### add_node

```cpp
Vertex add_node();
```

Output:

```cpp
Vertex
```

---

## Edge

### add_edge

```cpp
Edge add_edge(
    Vertex u,
    Vertex v);
```

Output:

```cpp
Edge
```

---

## Node Attributes

### node_attrs

```cpp
AttrMap&
node_attrs(
    Vertex v);
```

Ví dụ:

```cpp
g.node_attrs(v)["cpu"];
```

---

## Edge Attributes

### edge_attrs

```cpp
AttrMap&
edge_attrs(
    Edge e);
```

Ví dụ:

```cpp
g.edge_attrs(e)["weight"];
```

---

## Queries

### num_nodes

```cpp
size_t num_nodes() const;
```

---

### num_edges

```cpp
size_t num_edges() const;
```

---

### degree

```cpp
size_t degree(
    Vertex v) const;
```

---

### has_edge

```cpp
bool has_edge(
    Vertex u,
    Vertex v) const;
```

---

### edge

```cpp
Edge edge(
    Vertex u,
    Vertex v) const;
```

Ném exception nếu không tồn tại.

---

### source

```cpp
Vertex source(
    Edge e) const;
```

---

### target

```cpp
Vertex target(
    Edge e) const;
```

---

## Removal

### remove_edge

```cpp
bool remove_edge(
    Vertex u,
    Vertex v);
```

Output:

```cpp
true
false
```

---

## Iterators

### nodes

```cpp
nodes()
```

---

### edges

```cpp
edges()
```

---

### neighbors

```cpp
neighbors(
    Vertex v)
```

---

## Raw Boost Graph

### raw

```cpp
BGLGraph&
raw();

const BGLGraph&
raw() const;
```

Truy cập trực tiếp graph Boost.

---

## Fast Adjacency

### neighbors_fast

```cpp
const RawNeighborList&
neighbors_fast(
    Vertex v) const;
```

Được dùng trong:

```cpp
bfs
bidirectional_bfs
dijkstra
bidirectional_dijkstra
floyd_warshall
```

---

# Trọng số cạnh

Mặc định:

```cpp
"weight"
```

Ví dụ:

```cpp
g.edge_attrs(e)["weight"] =
    5.0;
```

Nếu không tồn tại:

```cpp
weight = 1.0
```

Các kiểu hợp lệ:

```cpp
double
int64_t
```

---

# Generators

## WaxmanGenerator

Header:

```cpp
graph/generators/waxman_generator.h
```

API:

```cpp
Graph generate(
    const WaxmanConfig&);
```

Config:

```cpp
struct WaxmanConfig
{
    size_t num_nodes;

    double alpha;
    double beta;

    uint64_t seed;
};
```

Node attributes:

```cpp
pos_x
pos_y
```

Edge attributes:

```cpp
distance
```

---

## GmlLoader

Header:

```cpp
graph/generators/gml_loader.h
```

API:

```cpp
Graph load(
    const std::string& path);
```

Hỗ trợ:

```gml
node [
    id
]

edge [
    source
    target
]
```

Các field khác bị bỏ qua.

---

# Algorithms

## BFS

Header:

```cpp
algorithms/bfs.h
```

API:

```cpp
BFSResult
bfs(
    const Graph&,
    Vertex source);
```

Output:

```cpp
distance
predecessor
```

Khoảng cách từ source tới mọi node reachable.

---

## Bidirectional BFS

Header:

```cpp
algorithms/bidirectional_bfs.h
```

API:

```cpp
BidirectionalBFSResult
bidirectional_bfs(
    const Graph&,
    Vertex source,
    Vertex target);
```

Output:

```cpp
found
distance
path
```

Tìm một đường đi ngắn nhất theo số cạnh.

---

## Dijkstra

Header:

```cpp
algorithms/dijkstra.h
```

API:

```cpp
DijkstraResult
dijkstra(
    const Graph&,
    Vertex source,
    const std::string& weight_attr =
        "weight");
```

Output:

```cpp
distance
predecessor
```

Tính:

```cpp
source -> mọi node reachable
```

---

## Dijkstra có ràng buộc

```cpp
DijkstraResult
dijkstra(
    const Graph&,
    Vertex source,
    const VertexSet& banned_vertices,
    const EdgeSet& banned_edges,
    const std::string& weight_attr =
        "weight");
```

Dùng trong Yen.

---

## Dijkstra Helpers

### build_path

```cpp
std::vector<Vertex>
build_path(
    const DijkstraResult&,
    Vertex source,
    Vertex target);
```

---

### edge_cost

```cpp
double edge_cost(
    const Graph&,
    Vertex u,
    Vertex v,
    const std::string& weight_attr =
        "weight");
```

---

### path_cost

```cpp
double path_cost(
    const Graph&,
    const std::vector<Vertex>& path,
    const std::string& weight_attr =
        "weight");
```

---

### path_prefix_costs

```cpp
std::vector<double>
path_prefix_costs(
    const Graph&,
    const std::vector<Vertex>& path,
    const std::string& weight_attr =
        "weight");
```

Dùng trong Yen.

---

## Bidirectional Dijkstra

Header:

```cpp
algorithms/bidirectional_dijkstra.h
```

API:

```cpp
BidirectionalPathResult
bidirectional_dijkstra(
    const Graph&,
    Vertex source,
    Vertex target,
    const VertexSet& banned_vertices,
    const EdgeSet& banned_edges,
    const std::string& weight_attr =
        "weight");
```

Output:

```cpp
found
cost
path
```

Được dùng bởi:

```cpp
nx::dijkstra_path
nx::dijkstra_path_length
yen_k_shortest_paths
```

---

## Floyd Warshall

Header:

```cpp
algorithms/floyd_warshall.h
```

API:

```cpp
DistanceMatrix
floyd_warshall(
    const Graph&,
    const std::string& weight_attr =
        "weight");
```

Output:

```cpp
Khoảng cách ngắn nhất giữa mọi cặp đỉnh
```

---

## Yen K Shortest Paths

Header:

```cpp
algorithms/k_shortest_paths.h
```

### PathResult

```cpp
struct PathResult
{
    std::vector<Vertex> path;
    double cost;
};
```

---

### join_paths

```cpp
std::vector<Vertex>
join_paths(
    const std::vector<Vertex>& root,
    const std::vector<Vertex>& spur);
```

---

### yen_k_shortest_paths

```cpp
std::vector<PathResult>
yen_k_shortest_paths(
    const Graph&,
    Vertex source,
    Vertex target,
    size_t k,
    const std::string& weight_attr =
        "weight");
```

Output:

```cpp
K đường đi đơn ngắn nhất
```

sắp xếp theo cost tăng dần.

---

# Connectivity

Header:

```cpp
nx/connectivity.h
```

API:

```cpp
bool is_connected(
    const Graph&);
```

---

# NetworkX Layer

Namespace:

```cpp
nx
```

Mục tiêu:

```cpp
API tương tự NetworkX
```

---

## shortest_path

```cpp
std::vector<Vertex>
shortest_path(
    const Graph&,
    Vertex source,
    Vertex target);
```

Implementation:

```cpp
bidirectional_bfs
```

---

## shortest_path_length

```cpp
size_t
shortest_path_length(
    const Graph&,
    Vertex source,
    Vertex target);
```

Implementation:

```cpp
bidirectional_bfs
```

---

## single_source_shortest_path_length

```cpp
std::unordered_map<
    Vertex,
    size_t>
single_source_shortest_path_length(
    const Graph&,
    Vertex source);
```

Implementation:

```cpp
bfs
```

---

## dijkstra_path

```cpp
std::vector<Vertex>
dijkstra_path(
    const Graph&,
    Vertex source,
    Vertex target,
    const std::string& weight_attr =
        "weight");
```

Implementation:

```cpp
bidirectional_dijkstra
```

---

## dijkstra_path_length

```cpp
double
dijkstra_path_length(
    const Graph&,
    Vertex source,
    Vertex target,
    const std::string& weight_attr =
        "weight");
```

Implementation:

```cpp
bidirectional_dijkstra
```

---

## single_source_dijkstra_path_length

```cpp
std::unordered_map<
    Vertex,
    double>
single_source_dijkstra_path_length(
    const Graph&,
    Vertex source,
    const std::string& weight_attr =
        "weight");
```

Implementation:

```cpp
dijkstra
```

---

## floyd_warshall

```cpp
DistanceMatrix
floyd_warshall(
    const Graph&,
    const std::string& weight_attr =
        "weight");
```

Implementation:

```cpp
algorithms::floyd_warshall
```

---

## shortest_simple_paths

```cpp
std::vector<
    std::vector<Vertex>>
shortest_simple_paths(
    const Graph&,
    Vertex source,
    Vertex target,
    size_t k,
    const std::string& weight_attr =
        "weight");
```

Implementation:

```cpp
yen_k_shortest_paths
```

---



# Quy tắc hiệu năng

Ưu tiên:

```cpp
neighbors_fast(...)
```

Tránh:

```cpp
boost::adjacent_vertices(...)
boost::out_edges(...)
```

Không dùng:

```cpp
Boost Property Map
```

trong các vòng lặp nóng.

Thuật toán phải nằm trong:

```cpp
algorithms/
```

Không cài thuật toán trong:

```cpp
nx/
```

Lớp `nx` chỉ là wrapper tương thích NetworkX.
