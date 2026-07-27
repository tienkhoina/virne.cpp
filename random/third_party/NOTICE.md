# NumPy legacy RandomState attribution

`NumpyRandomState` contains a C++ adaptation of algorithms from the NumPy
**1.26.4** legacy random implementation. It is pinned to that release because
`RandomState` promises stable seeded streams and Virne fixtures depend on the
exact order in which MT19937 words are consumed.

Reference sources from the NumPy `v1.26.4` tag:

- `numpy/random/src/mt19937/mt19937.c` and `mt19937.h`
- `numpy/random/src/legacy/legacy-distributions.c`
- `numpy/random/src/distributions/distributions.c`
- `numpy/random/mtrand.pyx`
- `numpy/random/_bounded_integers.pyx.in`

Copyright (c) 2005-2023, NumPy Developers. NumPy is distributed under the BSD
3-Clause license reproduced in `NUMPY_LICENSE.txt`.

This is source attribution, not a runtime or build dependency. Production C++
does not include NumPy headers, load NumPy binaries, import Python, or search an
environment for NumPy. The repository-local NumPy 1.26.4 installation is used
only by optional differential and benchmark scripts.
