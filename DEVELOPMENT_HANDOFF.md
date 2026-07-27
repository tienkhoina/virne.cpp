# Bàn giao phát triển VirneCpp

Tài liệu này chốt trạng thái Graph/DiGraph và Random tại mốc
**2026-07-27 UTC**, đồng thời cung cấp đường dẫn ngắn nhất để một người mới có
thể tiếp tục phát triển mà không phá vỡ tính tương thích, tính xác định hoặc
hiệu năng đã kiểm chứng.

Đây là tài liệu định hướng bàn giao, không phải một bản khai API thứ hai. Khi
có khác biệt, public header quyết định chữ ký chính xác; `graph/API.md` và
`random/README.md` quyết định ngữ nghĩa của từng subsystem.

## 1. Trạng thái tại thời điểm bàn giao

- Public surface Graph/DiGraph trong phạm vi Virne hỗ trợ đã được kiểm kê đầy
  đủ. Không còn cặp overload Graph/DiGraph bị thiếu trong surface đã đóng băng.
- Public surface Random mà Virne hỗ trợ đã được kiểm kê đầy đủ, gồm chữ ký,
  default, validation, thứ tự kết quả và mức tiêu thụ trạng thái RNG.
- Chữ ký public hiện tại được đóng băng. Internal helper nhận ID không phải API
  public mới.
- Graph/Random đã vượt clean Release build, CTest, differential oracle,
  benchmark gate và ASan/UBSan tại mốc bàn giao.
- Không còn lỗi correctness hoặc regression hiệu năng đã biết trong các case
  đang được gate. Điều này không phải lời hứa rằng toàn bộ NetworkX, toàn bộ
  Python `random` hoặc toàn bộ NumPy đã được port.

"Đầy đủ" trong tài liệu này luôn có nghĩa là đầy đủ đối với **supported
surface đã ghi trong repository**, không có nghĩa là sao chép mọi API động của
Python.

## 2. Thứ tự bắt buộc đọc trước khi sửa

1. `README.md` để hiểu triết lý chung và quy tắc hiệu năng.
2. `DEPENDENCIES.md` và `DEPENDENCIES.sha256` để hiểu dependency pin, local
   layout và các memory-layout hack được chấp nhận.
3. `graph/API.md` trước mọi thay đổi liên quan Graph, DiGraph, attribute,
   generator, GML, view hoặc graph algorithm.
4. `random/README.md` trước mọi thay đổi liên quan seed, state, distribution,
   choice, shuffle, permutation hoặc `NdArray`.
5. `benchmarks/README.md` và `benchmarks/RESULTS.md` để hiểu gate, oracle và
   cách diễn giải số đo.
6. `API_MUST_BUILD.md` trước khi kết luận một thay đổi đã hoàn tất.

Không phát triển dựa trên README tóm tắt hoặc một benchmark đơn lẻ. Phải đọc
canonical document của subsystem và declaration thật trong public header.

## 3. Thứ tự ưu tiên của nguồn sự thật

Khi hai nguồn có vẻ không thống nhất, áp dụng thứ tự sau:

1. Public header quyết định tên, overload, argument order, default và return
   type C++.
2. `graph/API.md` hoặc `random/README.md` quyết định ngữ nghĩa, lifetime,
   ordering, validation và state consumption.
3. Differential tests/oracles quyết định parity thực tế với phiên bản Python
   đã pin.
4. Benchmark scripts quyết định release performance gate.
5. `benchmarks/RESULTS.md` là baseline lịch sử; nó không thay thế contract.

Nếu declaration và canonical document khác nhau, không tự chọn một phía. Hãy
coi đó là lỗi cần được làm rõ và đồng bộ cùng test.

## 4. Bất biến thiết kế quan trọng nhất: hot loop chỉ dùng ID

String được phép tồn tại ở public boundary để API dễ đọc và gần NetworkX.
String, YAML key, hash map và registry lookup không được đi vào compute loop.

Luồng bắt buộc là:

```text
public/config name
        |
        v
resolve đúng một lần
        |
        v
Vertex / EdgeId / AttrId
        |
        v
toàn bộ loop, callback và nested helper chỉ dùng ID/index
```

Các quy tắc cụ thể:

- Resolve mỗi attribute name thành `AttrId` trước vòng lặp node, edge,
  neighbor, path, source, candidate, batch hoặc sample.
- Callback, predicate và lambda chạy lặp lại phải capture ID đã resolve.
- Thuật toán gọi thuật toán khác phải truyền ID xuống internal helper; không
  resolve lại theo từng source, spur, candidate, retry hoặc path.
- Trong hot loop không gọi `attr_id`, `at("...")`, `find("...")`,
  `contains("...")`, YAML lookup hay lookup chuỗi/hash tương đương.
- Ưu tiên field, contiguous array/vector và indexed access. Raw neighbor path
  chỉ dùng sau khi node/index và lifetime đã được xác thực ở boundary.
- `AttrId` thuộc về `AttributeRegistry` của một graph. Hai graph không liên
  quan phải resolve riêng, kể cả khi giá trị số của ID tình cờ giống nhau.
- `Vertex` là dense index. Stable edge ID có thể có lỗ sau khi xóa edge; dùng
  `edge_id_capacity()` cho storage đánh chỉ số theo edge ID, không dùng số
  lượng live edges.

Một thay đổi trả đúng kết quả nhưng đưa string/hash lookup trở lại hot loop vẫn
là regression thiết kế và không được merge.

## 5. Những điều phải giữ nguyên trong Graph/DiGraph

- `Graph` và `DiGraph` là simple graph; không âm thầm thêm multigraph hoặc
  Python object label vào core.
- Mọi API nằm trong paired surface phải giữ overload cho cả `Graph` và
  `DiGraph`, cùng argument order/default/return shape tương ứng.
- Edge, neighbor, predecessor/successor và generator order là một phần của
  contract. Cùng seed phải tạo cùng graph và cùng thứ tự cạnh với oracle.
- Dense `Vertex` là lựa chọn kiến trúc. Input dùng sparse label phải relabel ở
  boundary thay vì biến core thành dictionary graph.
- Checked views dùng ở boundary; unchecked/raw/indexed path chỉ dùng trong
  loop đã xác thực. View/range không được sống lâu hơn graph và structural
  mutation làm iterator/range đang hoạt động mất hiệu lực.
- Graph copy/conversion phải bảo toàn registry mapping và ordering theo contract
  hiện tại.
- Public string overload được giữ để tương thích; implementation phải chuyển
  sang một internal ID-taking path trước compute.

Các bất đối xứng có chủ đích, không phải API còn thiếu:

- `WaxmanGenerator` chỉ hỗ trợ `Graph`.
- `GmlLoader::load` trả `Graph`; directed GML dùng API có tên riêng vì C++
  không overload theo return type.
- Simple-path facade được hỗ trợ là `nx::shortest_simple_paths`; không có một
  global facade thứ hai.
- Các placeholder rỗng được nêu trong `graph/API.md` không hứa hẹn thêm DFS
  hoặc loader surface.

Không mở rộng surface chỉ vì NetworkX có thêm một hàm. Trước tiên phải xác
nhận Virne thực sự cần nó, xác định kiểu C++ tĩnh phù hợp và bổ sung contract,
Graph/DiGraph pairing, parity test cùng benchmark tương ứng.

## 6. Những điều phải giữ nguyên trong Random

- `PyRandom` khớp subset đã tài liệu hóa của CPython 3.10
  `random.Random` với seed integer không âm trong miền cố định.
- `NumpyRandomState` khớp subset đã tài liệu hóa của NumPy 1.26.4 legacy
  `RandomState`; nó không phải `numpy.random.Generator`.
- Hai đối tượng có stream độc lập. Không thay implementation bằng distribution
  STL nếu việc đó làm đổi bit, ordering hoặc state continuation.
- Output không phải tiêu chí duy nhất. Validation order, error path, empty
  input, zero-sized shape và **next state** đều thuộc contract.
- Raw calls `genrand_uint32()` và `next_uint32()` cùng tiêu thụ stream với API
  cấp cao; thêm một raw draw sẽ thay toàn bộ kết quả phía sau.
- `NdArray<T>` là owning contiguous shape/storage value nhỏ, không phải một
  bản triển khai NumPy tổng quát. Không tự thêm stride, broadcasting, dtype
  động hoặc arbitrary-axis semantics vào class này.
- Bulk/vector path phải bảo toàn chính xác thứ tự draw của scalar oracle.

Nếu cần thêm distribution mới, hãy đọc implementation của đúng phiên bản
CPython/NumPy đã pin, chốt state-consumption oracle trước, rồi mới tối ưu. Một
test chỉ so histogram hoặc tolerance phân phối là không đủ.

## 7. Dependency và memory-layout hack

Production C++ dependency chỉ được đặt dưới `libs/` và link thủ công bằng
CMake. Không cài Boost, yaml-cpp, tabulate hoặc dependency production mới vào
OS, Conda hay global compiler prefix.

Baseline đang khóa:

| Thành phần | Phiên bản/layout |
|---|---|
| Boost | 1.85.0 |
| yaml-cpp | 0.8.0 |
| tabulate | 1.4.0 |
| Toolchain fast-path đã kiểm chứng | GCC 11.4.0 + libstdc++ 11 |
| Python oracle | CPython 3.10.12 |
| NetworkX | 3.4.2 |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |

`.venv` là ngoại lệ chỉ dành cho test/oracle; production C++ không được link
hoặc phụ thuộc runtime vào nó.

Graph cố ý đọc layout nội bộ của Boost 1.85.0. Random có fast path phụ thuộc
layout GCC/libstdc++ đã pin. Đây là rủi ro được chấp nhận có guard compile-time,
không phải quyền nâng phiên bản tùy ý. Mọi upgrade phải thực hiện lại layout
audit, cập nhật pin/guard/tài liệu và chạy đầy đủ clean build, oracle,
benchmark, ASan/UBSan trước khi chấp nhận.

## 8. Quy trình thêm hoặc thay đổi chức năng

### Trước khi viết code

- Kiểm tra `git status --short` và giữ nguyên thay đổi không thuộc phạm vi của
  mình; không mặc định dirty worktree là file có thể xóa.
- Xác định thay đổi thuộc supported surface hiện tại hay là một API mới.
- Đọc declaration, canonical document, test và benchmark liên quan.
- Nếu port hành vi Python, đối chiếu đúng phiên bản oracle đã pin và xác định
  cả ordering/error/state semantics.
- Thiết kế public string boundary và internal ID path trước khi triển khai hot
  loop.

### Trong khi triển khai

- Giữ public signature NetworkX-shaped đã chốt, trừ khi thay đổi contract được
  yêu cầu rõ ràng.
- Bổ sung Graph/DiGraph pair khi API thuộc paired surface.
- Không để lookup tên trong callback hoặc helper được gọi lặp lại.
- Kiểm tra self-loop, empty graph/input, removed edge-ID hole, invalid index,
  zero-sized shape, tie order và seeded continuation khi có liên quan.
- Không sửa test để hợp thức hóa output mới nếu chưa chứng minh output cũ sai
  so với canonical oracle.

### Trước khi bàn giao

- Đồng bộ public-header inventory và canonical document.
- Chạy clean Release build và toàn bộ CTest.
- Chạy đúng differential oracle của subsystem.
- Chạy benchmark gate, không chỉ microbenchmark có lợi cho thay đổi.
- Với layout/raw-memory thay đổi, chạy full ASan/UBSan và upgrade checklist.
- Ghi rõ API nào chỉ được kiểm tra parity, API nào thực sự có timing gate.

## 9. Lệnh kiểm chứng chuẩn

Từ thư mục `virne.cpp`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
cmake --build build --target benchmark_random \
  random_differential_harness -j2
ctest --test-dir build --output-on-failure

.venv/bin/python benchmarks/compare_nx.py
.venv/bin/python benchmarks/compare_graph_completion.py
.venv/bin/python benchmarks/compare_generators.py
.venv/bin/python benchmarks/compare_gml.py --binary build/gml_harness
.venv/bin/python random/differential_test.py \
  --cpp build/random/random_differential_harness
.venv/bin/python random/benchmark_compare.py \
  --cpp build/random/benchmark_random
.venv/bin/python benchmarks/check_view_overhead.py
```

Đây là minimum release gate cho Graph/Random. Build thành công một target hoặc
chạy một unit test riêng lẻ không đủ để kết luận hoàn tất.

## 10. Baseline cần bảo vệ

| Gate | Baseline tại mốc bàn giao |
|---|---:|
| Base Graph/DiGraph correctness | 91/91 |
| Base Graph/DiGraph timing | 32/32 nhanh hơn Python, 4.51x–116.26x |
| Completion/order correctness | 34/34 |
| Completion/order timing | 34/34 nhanh hơn Python, 3.60x–542.50x |
| Seeded generators | 91/91 exact, gồm edge order; parity-only |
| GML | exact; 2/2 timing rows nhanh hơn, 4.16x–14.63x |
| Random differential | 1,260 CPython + 2,368 NumPy + 262,145 large-randint values |
| Random timing | 7/7 nhanh hơn Python/NumPy, 1.20x–53.24x |
| Clean Release CTest | 11/11 PASS |
| Full ASan/UBSan CTest | 11/11 PASS |

Tổng cộng 75 timing rows đang so trực tiếp với Python/NetworkX/NumPy đều nhanh
hơn. Generator suite chỉ là correctness/order oracle nên không được mô tả là
nhanh hơn.

View/raw là gate khác: nó so public C++ view với raw C++ access. Median closure
run là node attributes `1.001x`, edge attributes `0.999x`, adjacency `1.000x`,
với gate `view/raw <= 1.05x`. Cả ba là zero-overhead parity về mặt thống kê;
đặc biệt không được tuyên bố adjacency nhanh hơn raw một cách ổn định.

Timing tuyệt đối thay đổi theo máy và tải hệ thống. Exact output/order/state,
strict pass/fail gate và xu hướng regression mới là dữ liệu cần bảo vệ.

## 11. Hướng mở rộng PyG/LibTorch sau này

Graph core hiện tại đã có nền tảng phù hợp: dense `Vertex`, stable edge ID,
pre-resolved `AttrId` và contiguous/indexed traversal. Adapter PyG/LibTorch
sau này nên là một lớp bên ngoài graph core:

- chuyển topology thành `edge_index`/tensor một lần tại boundary;
- snapshot attribute theo ID vào contiguous tensor/buffer;
- cache mapping cần tái sử dụng và chỉ rebuild khi topology/schema thay đổi;
- không đưa Torch type, allocator hoặc dependency vào public Graph/DiGraph;
- seed Python-compatible RNG, NumPy-compatible RNG và Torch RNG ở lớp điều
  phối, không trộn state của chúng.

Không cần can thiệp hoặc làm động lại graph core chỉ để tạo một PyG adapter.

## 12. Những thay đổi không được thực hiện âm thầm

- Nâng Boost, GCC/libstdc++, yaml-cpp, tabulate hoặc Python oracle version.
- Fallback sang dependency hệ thống khi `libs/` thiếu.
- Đưa string/hash/YAML lookup vào hot loop.
- Dùng `AttrId` của graph này trên một graph không liên quan.
- Đổi seeded graph edge order vì topology vẫn giống nhau.
- Thay Random algorithm bằng API có cùng phân phối nhưng khác bit/state.
- Thêm Python object labels, multigraph, ndarray tổng quát hoặc dynamic return
  variant vào core mà không có một quyết định mở rộng contract riêng.
- Gọi một API "nhanh hơn Python" khi nó chưa nằm trong timing gate.
- Chỉ cập nhật code mà bỏ qua public inventory, oracle và benchmark docs.

## 13. Definition of done cho người phát triển tiếp

Một thay đổi Graph/Random chỉ hoàn tất khi tất cả câu sau đều đúng:

- [ ] Public signature và canonical documentation thống nhất.
- [ ] Graph/DiGraph pairing hoặc intentional exception được ghi rõ.
- [ ] Không có string/hash lookup trong hot loop hoặc repeated callback.
- [ ] Exact value, order, error path và RNG state được test khi liên quan.
- [ ] Seeded generator giữ đúng edge order.
- [ ] Clean Release build và toàn bộ CTest pass.
- [ ] Differential oracle liên quan pass.
- [ ] Strict benchmark gate liên quan pass và không che giấu regression.
- [ ] Raw-memory/layout change đã qua ASan/UBSan và compatibility review.
- [ ] Dependency production vẫn chỉ nằm trong `libs/`.

## 14. Điểm bắt đầu của phiên phát triển kế tiếp

Tại mốc bàn giao không có hạng mục Graph/Random còn dang dở đã biết. Vì surface
hiện tại được xem là hoàn thiện, người tiếp theo nên bắt đầu bằng một yêu cầu
nghiệp vụ cụ thể, xác nhận nó có thực sự đòi hỏi thay đổi graph core hay không,
rồi mới mở rộng contract. Nếu mục tiêu là PyG/LibTorch, ưu tiên adapter ngoài
core theo mục 11.

Trước mọi thay đổi, hãy chạy baseline trên máy hiện tại và lưu kết quả. Sau đó
chỉ so sánh trên cùng build type, cùng binary, cùng oracle version và điều kiện
CPU tương đương. Đây là cách phân biệt regression thật với nhiễu benchmark.

## 15. Prompt khởi động cho phiên kế tiếp

Sao chép nguyên văn prompt dưới đây để bắt đầu một phiên đọc codebase mới. Nó
cố ý yêu cầu audit và báo cáo trước, chưa cho phép agent sửa code:

```text
Bạn đang tiếp quản repository VirneCpp tại thư mục `virne.cpp`.

Mục tiêu của lượt đầu tiên là đọc và hiểu codebase để chuẩn bị cho các bước
phát triển tiếp theo. Chưa sửa code, chưa format lại file, chưa cài dependency,
chưa thay phiên bản thư viện và chưa chạy hành động phá hủy dữ liệu. Nếu
worktree đang dirty, coi mọi thay đổi hiện có là của người dùng và không được
ghi đè hoặc hoàn tác.

Hãy thực hiện theo thứ tự:

1. Đọc đầy đủ các tài liệu sau:
   - `DEVELOPMENT_HANDOFF.md`
   - `README.md`
   - `DEPENDENCIES.md` và `DEPENDENCIES.sha256`
   - `graph/API.md`
   - `random/README.md`
   - `benchmarks/README.md` và `benchmarks/RESULTS.md`
   - `API_MUST_BUILD.md`
   - tài liệu canonical của subsystem liên quan nếu yêu cầu mới chỉ ra một
     subsystem cụ thể.

2. Kiểm tra read-only trạng thái repository và lập bản đồ codebase:
   - xem `git status --short`, nhưng không reset/checkout/xóa file;
   - dùng `rg --files` để lập danh sách module, public header, implementation,
     test, benchmark và CMake target;
   - đọc top-level `CMakeLists.txt` cùng CMake file của các module;
   - phân biệt production source, test/oracle, benchmark, tài liệu và
     workspace-local dependency dưới `libs/`.

3. Với Graph/DiGraph, lần theo code từ public declarations qua implementation,
   views, algorithms, `nx` facade, generator, GML, cache và tests. Xác nhận:
   - paired Graph/DiGraph surface và các ngoại lệ có chủ đích;
   - dense `Vertex`, stable/holey edge ID và graph-local `AttrId` semantics;
   - ordering/lifetime/invalidation contract;
   - nơi public string được resolve một lần sang ID;
   - mọi node/edge/neighbor/path/source/candidate hot loop chỉ dùng ID/index.

4. Với Random, lần theo `PyRandom`, `NumpyRandomState`, `RandomContext`,
   `NdArray`, implementation, differential oracle, unit tests và benchmark.
   Xác nhận output, validation order, empty/zero-shape behavior và continuation
   state theo đúng CPython 3.10/NumPy 1.26.4 subset đã tài liệu hóa.

5. Đọc các guard và direct-memory optimization liên quan Boost 1.85.0 và GCC
   11.4/libstdc++ 11. Không đề xuất nâng pin hoặc thay layout hack nếu chưa có
   explicit compatibility task và chưa lập đủ kế hoạch oracle/sanitizer/
   benchmark.

6. Nếu cần kiểm chứng trạng thái, chỉ chạy các build/test/benchmark đã được
   tài liệu hóa và không cài package. Production dependency phải lấy từ
   `libs/`; `.venv` chỉ dùng làm Python oracle. Trong lượt đọc đầu tiên, ưu
   tiên kiểm tra artifacts hiện có và nêu rõ lệnh nào đã hoặc chưa chạy.

Quy tắc thiết kế bắt buộc phải giữ trong mọi đề xuất sau này:

- Public API có thể nhận string để query, nhưng phải resolve đúng một lần ở
  boundary.
- Mọi hot loop, callback và nested helper phải dùng `Vertex`, `EdgeId`,
  `AttrId` hoặc contiguous index; cấm string/hash/YAML lookup lặp lại.
- Không dùng `AttrId` giữa hai registry không liên quan.
- Không thay đổi public signature, seeded output/edge order hoặc RNG stream
  semantics nếu chưa có yêu cầu thay đổi contract rõ ràng.
- Không thêm dependency production vào OS/Conda/global environment; chỉ dùng
  `libs/` và explicit CMake target.
- PyG/LibTorch sau này phải là adapter ngoài graph core, dùng indexed/tensor
  conversion tại boundary.

Kết thúc lượt đầu tiên bằng một báo cáo có dẫn chứng file và dòng, gồm:

1. sơ đồ module và luồng phụ thuộc chính;
2. public API surface liên quan đến yêu cầu mới;
3. các bất biến correctness, ordering, lifetime và RNG state phải bảo vệ;
4. các hot path, ID conversion boundary và memory-layout hack quan trọng;
5. test/oracle/benchmark target tương ứng;
6. khác biệt thực tế giữa code và tài liệu, nếu có;
7. rủi ro và một kế hoạch triển khai theo bước nhỏ cho yêu cầu tiếp theo.

Không sửa code trong lượt audit này. Không kết luận một API thiếu chỉ vì
NetworkX/Python có API đó; trước hết phải đối chiếu supported surface đã đóng
băng của VirneCpp. Không tuyên bố nhanh hơn nếu case đó không nằm trong timing
gate. Nếu không tìm thấy sai lệch, hãy nói rõ đã đối chiếu những file nào thay
vì tạo ra công việc giả.
```

Sau khi nhận báo cáo audit này, người dùng mới cung cấp hoặc xác nhận phạm vi
thay đổi cụ thể. Khi đó agent phải lập kế hoạch dựa trên evidence đã đọc, giữ
nguyên các contract trong tài liệu này và chạy đúng gate liên quan trước khi
bàn giao.
