# Registry benchmark scope correction

The numeric output gate in
`online_cluster_complete_2026-09-10_registry_benchmark.json` passed, but its timing
comparison is **invalid and not an accepted speed measurement**. Likewise,
`online_cluster_complete_2026-09-10.json` must not be used as the final cluster
acceptance record. Both artifacts are retained unchanged as diagnostic evidence.

The native `LearningSolver` constructs its model lazily in the first call to
`learn`, inside the measured epoch. The Python fixture eagerly built the policy
before starting its clock. Thus the advertised exclusion of model construction
was false on the native side. This is a failed benchmark-scope gate, not a
performance optimization of production code.

The corrected benchmark performs one greedy evaluation before timing on both
sides. This constructs the native model and warms the same policy path without
sampling or updating weights. Only this invalid registry timing row is reopened.
The environment timing row was already valid and is reused without rerunning.
All differential gates also remain closed.

The next diagnostic registry row under `online_cluster_accepted_2026-09-10*`
fixed lazy model initialization, but still launched a new native process for
each sample. Its three warmups therefore did not warm native autograd state,
whereas the Python runtime retained its process across samples. That row is
also not an accepted comparable timing; its byte equality remains valid.

The final harness now performs all three warmups and five samples in one
process per runtime, checks every repetition's outputs and retains the same
timed epoch. The environment benchmark is reused throughout.

The authoritative replacement is `online_cluster_final_2026-09-10.json`.
