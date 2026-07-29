# Frozen foundation manifest

The following directories are read-only inputs to the Python-to-C++ port. Port
work may link and call them but must not change their behavior or source:

| Directory | Files | Aggregate SHA-256 | Git tree (when tracked) |
|---|---:|---|---|
| `graph/` | 69 | `32e9ab7d4a6e89c9443f433469c374c933bb3568a348c0a284448f7651248459` | `c0c50b7c3bb7ab500051f2a2206a2f7674728ecc` |
| `csv/` | 8 | `170a68f055aece99b45a41ee4039ca282ef22ddbf5376cabbc34ab754d67a530` | `b260ff400b8617705a590c1b1a4c8b9292f200a7` |
| `config/` | 12 | `d9881b1c59a9e9a32c3599e12d8630e373079d667552d93efe25f3fe0c7a72a3` | `ef54f9c3d1d11a2b2e4149f11592d8e3e22dfcf9` |
| `libs/yaml-cpp/` | 397 | `8ef1f48c64160474b818ac4644f19bc6b2b204a5f74c07c13389448d951bda8a` | vendored payload |

The aggregate digest algorithm is deterministic: use CMake `list(SORT)` on
relative POSIX-style paths, compute lowercase SHA-256 for each file, append records as
`path<TAB>sha256<LF>`, then SHA-256 the complete UTF-8 record stream. `.git` and
`__pycache__` directories are excluded.

`porting/verify_frozen.cmake` implements the check and is registered as CTest
`frozen_component_integrity`. A mismatch is a hard failure. Update these values
only as a separate, explicitly approved foundation change; never update them to
make an unrelated port pass.

Manifest captured from C++ baseline commit
`5e77c4447aa23f9698dd9b628b173168e3407909` on 2026-07-27.
