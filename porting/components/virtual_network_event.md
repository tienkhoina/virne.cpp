# Component API: `network.VirtualNetworkEvent`

State: **COMPLETE / FROZEN** on 2026-07-29.

The event leaf is lines 18--52 of
`../virne/virne/network/virtual_network_request_simulator.py`, whose pinned
SHA-256 is
`970E63F9DAC59F60E2ED1786606DC87D3271AF062BA1D8D6C67AEF8D3C7478E1`.
The network-generation-chain contract was read first. Only this previously
unclear class body was opened to lock its validation and representation.

## Python behavior and typed boundary

The dataclass stores `id`, `type`, `v_net_id`, and `time` in constructor order.
`__post_init__` rejects a type not equal to `0` or `1`, a negative request ID,
then a negative time. It does not validate `id`. `repr` and `str` both produce
`VirtualNetworkEvent(v_net_id=..., time=..., type=..., id=...)`.
Dynamic item get/set delegates directly to `getattr`/`setattr`; mutation after
construction can therefore violate every invariant.

Python equality admits `False`/`True` and `0.0`/`1.0` as event types, arbitrary
integer-like IDs, and `NaN` time because `NaN < 0` is false. Those dynamic
lanes are boundaries. Native events use fixed integer/enum/double fields,
reject invalid enum values, negative request IDs, negative or NaN time, and
retain positive infinity. Native setters preserve these invariants.

## Stable C++ API

All names are in `virne::network`.

```cpp
using VirtualRequestId = std::int64_t;
using VirtualEventId = std::size_t;

enum class VirtualEventType : std::uint8_t { leave = 0, arrival = 1 };

struct VirtualNetworkEventInput {
    VirtualEventId id;
    VirtualEventType type;
    VirtualRequestId virtual_network_id;
    double time;
};

class VirtualNetworkEvent {
public:
    explicit VirtualNetworkEvent(VirtualNetworkEventInput);
    VirtualNetworkEvent(VirtualEventId, VirtualEventType,
                        VirtualRequestId, double);

    VirtualEventId id() const noexcept;
    VirtualEventType type() const noexcept;
    VirtualRequestId virtual_network_id() const noexcept;
    double time() const noexcept;
    void set_id(VirtualEventId) noexcept;
    void set_type(VirtualEventType);
    void set_virtual_network_id(VirtualRequestId);
    void set_time(double);
    std::string repr() const;
};

std::vector<VirtualNetworkEvent> make_virtual_network_events(
    const std::vector<VirtualNetworkEventInput>&, std::size_t workers = 1);
void stable_sort_virtual_network_events(std::vector<VirtualNetworkEvent>&);
```

Fixed fields are direct scalar members. Accessors are inline direct loads; no
string, map, reflection, or attribute lookup exists in event loops. Batch
construction writes pre-sized indexed slots. Worker zero/one is sequential;
wider caller-configured values use deterministic contiguous ranges, retain
input order, and rethrow the lowest input-index validation error. There is no
automatic worker policy. Stable time sort preserves input order for ties, so
an arrival block followed by a leave block retains all equal-time arrivals
before equal-time leaves.

## Accepted gate

The differential artifact passes 20 classified cases: 13 shared cases, one
native NaN-rejection extension, and six explicit Python-only dynamic
boundaries. Coverage includes ordinary/extreme fields, exact representative
repr, validation order, positive infinity, setters, batch workers `0/1/2/8`,
lowest-index error, stable equal-time order, and concurrent independent
batches. The oracle extracts only the pinned class AST, so no simulator,
Torch, solver, or system import is required.

The single accepted construct-and-sort benchmark uses 131,072 events, one
warm-up, and three repetitions. C++ is 47.206x, 53.776x, and 34.298x faster
than Python at caller-configured workers `1/2/8`, respectively, with identical
entry count, output bytes, and checksum. The accepted benchmark and its JSON
artifact are frozen and must not be rerun or updated. See
`../results/virtual_network_event_2026-07-29.md`.
