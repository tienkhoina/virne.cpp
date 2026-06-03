# Config API

## ConfigLoader::load

Đọc file YAML và tạo đối tượng `Config`.

```cpp
Config cfg =
    ConfigLoader::load(
        "setting/main.yaml");
```

---

## Config::get<T>

Lấy giá trị theo đường dẫn dạng `a.b.c`.

```cpp
std::string solver =
    cfg.get<std::string>(
        "solver.solver_name");

int seed =
    cfg.get<int>(
        "experiment.seed");
```

---

## override_parser::apply

Áp dụng một override theo cú pháp Hydra/OmegaConf.

```cpp
override_parser::apply(
    cfg,
    "experiment.seed=123");

override_parser::apply(
    cfg,
    "solver.solver_name=my_solver");
```

---

## Config::save

Lưu trạng thái hiện tại của cấu hình ra file YAML.

```cpp
cfg.save("after.yaml");
```

---

## Config::root

Trả về node gốc để dump/debug toàn bộ cây cấu hình.

```cpp
std::cout
    << cfg.root()
    << '\n';
```

---

## Chạy từ CLI

Chương trình tự động áp dụng tất cả override được truyền qua command line.

```bash
./main
```

```bash
./main \
experiment.seed=123
```

```bash
./main \
solver.solver_name=my_solver \
experiment.seed=123 \
p_net_setting.topology.num_nodes=500
```

Tương đương:

```yaml
solver:
  solver_name: my_solver

experiment:
  seed: 123

p_net_setting:
  topology:
    num_nodes: 500
```

---

## Ví dụ hoàn chỉnh

```cpp
Config cfg =
    ConfigLoader::load(
        "setting/main.yaml");

override_parser::apply(
    cfg,
    "experiment.seed=123");

std::cout
    << cfg.get<int>(
           "experiment.seed")
    << '\n';

cfg.save("after.yaml");
```
# PyRandom

Bộ sinh số ngẫu nhiên tương thích với Python `random.Random`.

## Include

```cpp
#include "random/py_random.h"
```

## Khởi tạo

```cpp
PyRandom rng(42);
```

## Sinh số thực [0, 1)

```cpp
double x =
    rng.random();
```

## Sinh số thực [a, b]

```cpp
double x =
    rng.uniform(
        0.0,
        1.0);
```

## Sinh 32 bit ngẫu nhiên

```cpp
uint32_t x =
    rng.getrandbits32();
```

## Sinh số nguyên [0, stop)

```cpp
uint64_t x =
    rng.randrange(
        100);
```

## Chọn ngẫu nhiên một phần tử

```cpp
std::vector<int> v =
{
    1,2,3,4
};

int x =
    rng.choice(v);
```

## Trộn ngẫu nhiên

```cpp
std::vector<int> v =
{
    1,2,3,4
};

rng.shuffle(v);
```

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

# Graph Module

Backend: Boost Graph Library (BGL)

## Structure

```text
graph/
├── attribute.h
├── graph_types.h
├── graph.h
├── graph.cpp
├── node_view.h
├── edge_view.h
├── algorithms/
└── io/
```

## Include

```cpp
#include "graph/graph.h"
```

## Create Graph

```cpp
Graph g;
```

## Add Node

```cpp
auto n0 = g.add_node();
auto n1 = g.add_node();
```

# Add Edge

```cpp
auto e = g.add_edge(n0, n1);
```

## Node Attributes

```cpp
g.node_attrs(n0)["cpu"] = int64_t(100);
g.node_attrs(n0)["name"] = std::string("server_0");

auto cpu =
    std::get<int64_t>(
        g.node_attrs(n0)["cpu"]);
```

## Edge Attribute

```cpp
g.edge_attrs(e)["bw"] = 100.0;
g.edge_attrs(e)["fiber"] = true;

auto bw =
    std::get<double>(
        g.edge_attrs(e)["bw"]);
```

## Graph Info

```cpp
g.num_nodes();
g.num_edges();
```

## Raw Boost Graph

```cpp
auto& raw = g.raw();

boost::num_vertices(raw);
boost::num_edges(raw);
```

## Supported Attribute Types

```cpp
int64_t
double
bool
std::string
```

## Curremt API

```cpp
Vertex add_node();

Edge add_edge(
    Vertex u,
    Vertex v);

AttrMap&
node_attrs(Vertex v);

AttrMap&
edge_attrs(Edge e);

size_t num_nodes() const;

size_t num_edges() const;

BGLGraph&
raw();
```

# WaxmanGenerator

Sinh topology Waxman tương thích `networkx.waxman_graph()`.

## Include

```cpp
#include "graph/generators/waxman_generator.h"
```

## Cấu hình

```cpp
WaxmanConfig cfg;

cfg.num_nodes = 100;
cfg.alpha = 0.1;
cfg.beta = 0.4;
cfg.seed = 42;
```

## Sinh graph

```cpp
Graph g =
    WaxmanGenerator::generate(
        cfg);
```

## Node Attributes

```cpp
pos_x
pos_y
```

## Edge Attributes

```cpp
distance
```

## API

```cpp
struct WaxmanConfig
{
    size_t num_nodes;

    double alpha;
    double beta;

    uint64_t seed;
};

class WaxmanGenerator
{
public:

    static Graph generate(
        const WaxmanConfig& cfg);
};
```

# GmlLoader

Đọc topology từ file `.gml`.

## Include

```cpp
#include "graph/generators/gml_loader.h"
```

## Load Graph

```cpp
Graph g =
    GmlLoader::load(
        "topology.gml");
```

## GML được hỗ trợ

```gml
node [
    id 0
]

node [
    id 1
]

edge [
    source 0
    target 1
]
```

## API

```cpp
class GmlLoader
{
public:

    static Graph load(
        const std::string& path);
};
```