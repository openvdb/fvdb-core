# Sparse Convolution Semantics Unification Plan

Status: semantic design and implementation sequence ratified; only the bounded
release-policy choices listed in Section 19 remain; no implementation has been
started

Primary motivation: [issue #668](https://github.com/openvdb/fvdb-core/issues/668)

Relevant discussion:

- [Mark Harris's diagnosis](https://github.com/openvdb/fvdb-core/issues/668#issuecomment-4987628144)
- [Jonathan Swartz's plan/topology observation](https://github.com/openvdb/fvdb-core/issues/668#issuecomment-4987861785)

## 1. Executive decision

fVDB sparse convolution should be defined as a restriction of a dense PyTorch
cross-correlation to integer lattice coordinates. Sparsity changes storage and
the set of coordinates at which results are materialized. It must not change
the kernel phase, the source/target connections, the accumulated values, or
the meaning of transposition.

The unifying object is therefore not a forward grid builder, a transpose grid
builder, or a backend-specific plan. It is one bipartite convolution relation:

```text
fine coordinate p  <---- kernel tap u ---->  coarse coordinate q
```

Forward convolution evaluates this relation from fine to coarse. Transposed
convolution evaluates the same relation from coarse to fine. A generated output
topology is the projection of the relation reached from an active input set. An
explicit output topology restricts that projection. The transpose of an
existing plan reverses its exact stored edges; it does not rediscover them.

The executable specification for the entire pipeline is:

1. Replace every structurally active input coordinate by the scalar value one.
2. Replace every kernel tap by the scalar value one and disable bias.
3. Densify the signal on a coordinate-aware canvas large enough that no output
   touched by the signal is cropped.
4. Run the corresponding PyTorch operation.
5. Require the sparse implementation to produce exactly the same integer count
   at every output coordinate.
6. Define the generated sparse topology as exactly the coordinates whose count
   is strictly positive.
7. With the topology fixed, repeat with the actual features and weights and
   require the values and gradients to match PyTorch.

Matching only the positive coordinate set is insufficient. The all-one output
is the degree of each output vertex in the convolution graph, so exact count
matching also detects duplicated edges, missing edges, and many tap-ordering
errors.

This proposal makes PyTorch the numerical and tap-phase authority. Minkowski
Engine and other sparse libraries are useful comparison points, but they do not
define fVDB semantics.

## 2. What this resolves

This design gives one answer to each currently conflated question:

| Question | Answer |
|---|---|
| Which kernel taps touch which coordinates? | The Torch-equivalent integer relation in Section 5. |
| What is an automatically generated forward topology? | The nonzero support of an all-one dense forward convolution with no border cropping. |
| What is an automatically generated transpose topology? | The nonzero support of an all-one dense transposed convolution with no border cropping. |
| What does an explicit target grid mean? | Evaluate the same relation only at those target coordinates. |
| What is the transpose of a particular plan? | The exact reversal of that plan's finite edge set. |
| Is transposed convolution an inverse? | No. It is the adjoint connectivity pattern; it need not recover its input. |
| Does stride change even-kernel centering? | No. Stride samples one fixed Torch kernel phase. |
| Does a numerically zero feature remove topology? | No. Active coordinates are structural and are treated as potentially nonzero. |
| Does bias grow topology? | No. Bias is applied after a finite output domain has been selected. |
| Is `coarsened_grid(S)` the same as `conv_grid(K, S)`? | No. One groups physical cells; the other projects a convolution relation. |
| What do world transforms do? | They register fine and coarse index lattices. They do not invent or alter kernel offsets. |

The same oracle applies at every pipeline boundary:

| Pipeline boundary | Required comparison |
|---|---|
| Kernel geometry | Every tap's sparse offset equals the offset observed from a single-tap dense Torch probe. |
| Generated grid | Its coordinates equal the positive support of the dense all-one result. |
| Topology construction resources | Forward staging is bounded by active inputs plus exact valid emissions, not every rejected input/tap candidate. |
| Rulebook | Its per-coordinate edge counts equal the dense all-one values, not merely their support. |
| Forward executor | Actual sparse values equal dense cross-correlation at the same global coordinates. |
| Generative transpose grid | Its coordinates and counts equal full, uncropped dense `conv_transpose3d`. |
| Exact plan transpose | Its edge set is the exact reversal of the forward plan and satisfies the dot-product identity. |
| Backward/autograd | Input and weight gradients equal dense Torch gradients. |
| Backend lowering | Every backend realizes the same rulebook. |
| Grid transform | Output coordinate `q` maps to the documented fine-lattice anchor in world space. |

This prevents a grid builder from "passing topology" while the evaluator uses a
different phase, which is the class of failure exposed by issue #668.

## 3. Scope and non-goals

### 3.1 In scope

- Forward and transposed sparse 3D convolution.
- Stride one and strided convolution.
- Odd, even, cubic, and anisotropic kernels.
- Automatically generated and explicitly supplied topologies.
- Exact plan transposition.
- Integer coordinates, including negative coordinates.
- Batched grids with per-grid voxel sizes and origins.
- All convolution backends and their autograd behavior.
- The relationship between index-space convolution and world-space transforms.
- A test oracle that is independent of production topology code.
- Peak-memory complexity of generated-topology construction.

The equations include dilation so that the geometry does not need to be
redesigned later. The first implementation may keep the public API at dilation
one, provided unsupported dilation fails explicitly.

### 3.2 Not in scope

- Treating transposed convolution as a numerical inverse.
- Dynamically pruning topology because actual feature values happen to be zero
  or cancel.
- Making pooling, cell aggregation, or `coarsened_grid` aliases of convolution.
- Adopting another sparse library's behavior when it conflicts with the stated
  Torch equivalence contract.
- Optimizing special cases before their equivalence to the canonical relation
  has been mechanically proved.
- Defining an infinite bias-supported topology. A nonzero bias on an infinite
  lattice would be nonzero everywhere and is not a useful sparse topology rule.

## 4. Vocabulary and domains

Use `fine` and `coarse` for the two lattice roles, independent of execution
direction:

- `F` is the finite set of active fine-grid coordinates `p`.
- `C` is the finite set of active coarse-grid coordinates `q`.
- `u` is a zero-based kernel tap coordinate.
- Forward execution maps features on `F` to features on `C`.
- Transposed execution maps features on `C` to features on `F`.

Calling both grids `source` and `target` inside low-level code is dangerous
because those names reverse with direction. The normalized internal topology
should always be expressed as a fine/coarse relation. Public plans may still
expose source and target according to execution direction.

An active coordinate is structural. Its runtime feature may be zero. The plan
cannot depend on runtime feature values because plans are reused for different
features and weights.

## 5. Normative index-space geometry

All equations below are componentwise in three dimensions. The one-dimensional
form is sufficient to understand each axis.

For each axis, define:

- `K > 0`: kernel size.
- `S > 0`: stride.
- `D > 0`: dilation; initially `D = 1` in the public fVDB API.
- `E = D * (K - 1) + 1`: effective kernel extent.
- `P_before = floor((E - 1) / 2)`.
- `P_after = E - 1 - P_before`.
- `u in {0, ..., K - 1}`: the kernel tap index.
- `r(u) = D * u - P_before`: the tap's fine-lattice offset.
- `a`: an integer registration offset between the fine and coarse lattices.

The default is `a = 0`. Section 9 defines how `a` relates to grid transforms.

The single authoritative connection law is:

```text
p = a + S * q + r(u)
```

or, in vector form,

```text
p = a + S .* q + D .* u - P_before.
```

This is PyTorch cross-correlation, not a spatially flipped mathematical
convolution. Stride changes which output anchors are sampled; it does not
change `r(u)`. Only `P_before` enters the infinite-lattice connection law.
`P_after` records the other half of Torch's phase split for documentation and
for any future explicitly bounded mode; it does not crop full sparse support.

### 5.1 Default even-kernel phase

The default phase extends PyTorch `padding="same"` tap placement from stride one
to all strides. PyTorch documents that string `same` padding preserves shape
and is only supported at stride one. The sparse full-support operation is a
deliberate infinite-lattice extension of its local tap phase, not a claim that
PyTorch accepts `padding="same"` with larger stride.

For dilation one:

| K | `P_before` | `P_after` | `r(u)` |
|---:|---:|---:|---|
| 1 | 0 | 0 | `{0}` |
| 2 | 0 | 1 | `{0, 1}` |
| 3 | 1 | 1 | `{-1, 0, 1}` |
| 4 | 1 | 2 | `{-1, 0, 1, 2}` |
| 5 | 2 | 2 | `{-2, -1, 0, 1, 2}` |
| 6 | 2 | 3 | `{-2, -1, 0, 1, 2, 3}` |

A local probe in the repository's `fvdb` environment, using PyTorch 2.10.0,
confirmed these placements tap by tap for kernel sizes one through six. This
probe must become a small pinned regression test rather than remain an
assumption.

The current gather/scatter evaluator's kernel start,
`floor(-K / 2 + 1)`, agrees with this table. Its complete probe logic already
implements the proposed relation for `a=0`: forward output `q` probes fine
coordinate `p=S*q+r(u)`, while transposed output `p` solves `p-r(u)=S*q` and
probes that coarse `q` only when divisible. This evaluator is the existing
canonical path around which the builders should converge.

The even-kernel topology paths encode several different effective footprints:

- the forward production builder enumerates `d in [0, K-1]` in
  `p+d=S*q`, so its effective evaluator offset is
  `r=-d in [-(K-1), 0]`;
- the transpose production builder directly emits
  `r=d in [0, K-1]`;
- the strided forward topology test helper treats the receptive interval as
  `[-K//2, K//2]`, which contains `K+1` taps when `K` is even;
- the transpose topology and dense strided helpers use
  `r=k-K//2`, or `[-K//2, K-1-K//2]`, which is `[-K/2, K/2-1]`
  for even `K`.

None equals the Torch-aligned `[-1, 0, 1, 2]` footprint for `K=4` in all
directions. The test repair must inventory these separately rather than
describing them as one shared `K//2` convention.

### 5.2 Forward values

For forward weights `W[o, i, u]`, input features `x[p, i]`, and no bias:

```text
y[q, o] = sum over i,u of
          W[o, i, u] * x[a + S*q + r(u), i].
```

Coordinates outside the active fine set have value zero.

### 5.3 Transposed values

The mathematical transpose of the finite forward operator accumulates along
the same triples in the opposite direction:

```text
x_bar[p, i] += W[o, i, u] * y_bar[q, o]

for every edge satisfying p = a + S*q + r(u).
```

There is no spatial kernel flip in the edge definition. Statements such as
"transpose is convolution with a flipped kernel" are derived implementation
identities under particular dense conventions; they are not a safe primary
specification for sparse topology.

PyTorch's `ConvTranspose3d` documentation describes the operator as the
gradient of `Conv3d` with respect to its input and explicitly notes that it is
not an actual inverse. The fVDB public transposed module stores weights in
fVDB's `[C_out, C_in, ...]` convention, whereas PyTorch's functional transpose
expects `[C_in, C_out, ...]`. The dense test adapter must transpose the two
channel axes exactly once. That layout adaptation does not alter spatial taps.

Two valid uses must be distinguished:

- An independently learned `SparseConvTranspose3d` has its own public weight
  `V[C_out_transpose, C_in_transpose, ...]`. It uses transposed spatial
  connectivity but is not claimed to be the adjoint of some other weighted
  layer.
- For the exact weighted adjoint of a forward layer with
  `W[C_coarse, C_fine, ...]`, the transposed plan has input channels
  `C_coarse` and output channels `C_fine`, and the caller must pass
  `W_adjoint = W.transpose(0, 1).contiguous()` in fVDB's public layout
  `[C_fine, C_coarse, ...]`. The current plan reverses `channel_pairs`, but it
  does not own or transpose a weight tensor.

The API documentation currently blurs these cases. Slice 0 must make the
distinction normative; it should not force independent decoder weights and an
exact tied-weight adjoint into one meaning.

## 6. Topology is dense structural support

### 6.1 Structural mask

For a fine active set `F`, define:

```text
m_F[p] = 1 if p is active, else 0.
```

This mask deliberately ignores actual feature values. A stored zero is still a
structurally active coordinate and must not change a reusable plan.

### 6.2 Automatically generated forward topology

Apply a scalar all-one kernel to `m_F`:

```text
degree_C[q] = sum over u of m_F[a + S*q + r(u)].
```

The full-support, non-bordered forward topology is:

```text
C_full = {q in Z^3 | degree_C[q] > 0}.
```

Required invariants:

- Every generated output coordinate has positive degree.
- The integer degree equals the dense PyTorch all-one result exactly.
- Duplicate `(p, q, u)` edges are forbidden.
- A numerically cancelling real convolution does not remove the coordinate.

`GridBatch.conv_grid` and a forward plan with `target_grid=None` should implement
this policy.

### 6.3 Automatically generated transposed topology

For a coarse active set `C`, every coarse coordinate spreads through every tap:

```text
F_full = {a + S*q + r(u) | q in C, u in kernel taps}.
```

Equivalently, `F_full` is the positive support of dense transposed convolution
of the coarse structural mask with an all-one kernel, with no output cropping.

`GridBatch.conv_transpose_grid` should implement this policy. Once "full
support" is named, this topology is not ambiguous.

### 6.4 Explicit target topology

An explicitly supplied target is a restriction, not a new convolution rule.

For forward execution with `C_user`, build only relation edges whose coarse
coordinate is in `C_user`. For transposed execution with `F_user`, build only
edges whose fine coordinate is in `F_user`.

An explicit target may contain degree-zero rows. Those rows produce zero before
bias and bias after the module adds bias. This is mathematically valid but can
be surprising. Plans should expose a coverage report containing at least:

- number and fraction of degree-zero output rows;
- minimum, maximum, and histogram of output degree;
- number and fraction of degree-zero input columns;
- optionally, sample coordinates for each failure class.

Generated full-support targets must treat a zero-degree output as an internal
error. Explicit targets should allow it, while offering a strict mode that
rejects it.

### 6.5 Submanifold convolution

Stride-one convolution with `target_grid=source_grid` is simply an explicit
restriction to the input topology. It is not a different tap relation. This is
how fVDB can match dense Torch values at the active input coordinates without
materializing the full grown support.

### 6.6 Bias

Bias must not participate in topology generation. On an infinite lattice, a
nonzero bias would make every coordinate nonzero. The order is:

1. Choose a finite output domain by full structural support or explicit target.
2. Evaluate the linear convolution on that domain.
3. Add bias to every materialized output row.

Dense value tests with bias compare only on that same selected finite domain.

### 6.7 Name the topology policy symmetrically

Plans in both directions should store one explicit topology policy:

- `full_support`: generate the target as the complete positive structural
  support of the selected direction;
- `restricted`: use the supplied target grid and intersect the canonical
  relation with it.

For backward compatibility, an omitted policy may initially be inferred from
whether `target_grid` is `None`, but the normalized plan must store the resolved
policy. Forward and transposed factories should use the same names and rules.
`from_plan_transposed` preserves the original finite domains and records
restricted topology plus exact-transpose provenance.

This decision does not define a future finite rectangular or "bounded" policy.
Such a policy needs its own coordinate and cropping specification before being
added; it must not be inferred from `P_after`.

## 7. Three transpose use cases that must remain distinct

### 7.1 Transpose of a particular forward plan

A forward plan defines a finite matrix, including its exact fine domain,
coarse domain, phase, and edge set. Its transpose is unique:

- swap the source and target feature domains;
- reverse every stored edge;
- retain its kernel tap identifier;
- reverse supported channel pairs;
- do not rebuild the relation from the grids.

`ConvolutionPlan.from_plan_transposed(plan)` should therefore share an immutable
rulebook or create a constant-time reverse view. The current gather/scatter
rebuild is correct for `a=0` because its two probe directions implement the
same relation on the swapped grids. Replacing it is therefore a simplification
and a defense against future skew, not a repair for the present issue.

For `GatherScatterDefaultTopology`, the reverse view is specifically a swap of
`gatherIndices` with `scatterIndices`, `featureTotalVoxels` with
`outputTotalVoxels`, and the direction flag. Tap-grouped offsets are unchanged,
and the execution kernel is already shared. The explicit-target transposed
builder remains necessary for plans that are not derived from a forward plan.

For this backend, `direction` is metadata and entry-point validation, not an
execution-time geometric operation. The public forward and transpose wrappers
check that the flag names the expected entry point, but both dispatch to the
same forward operation, and their backward operations perform the same gather,
matrix, and scatter sequence. Reversing the index arrays is the complete
mathematical reversal. An executor must not see the transposed flag and perform
a second swap. The two currently separate backward structs are duplicate
implementations and should collapse to one shared operation in Slice 5.

The defining test is:

```text
dot(A_W x, z) == dot(x, A_W^T z), with bias = 0.
```

Under the current public fVDB layout, this test must execute the transposed plan
with `W.transpose(0, 1).contiguous()`, as specified in Section 5.3. This
identity must hold for arbitrary sparse domains, including domains with
zero-degree rows or columns.

### 7.2 Generative transposed convolution

Given only a coarse active grid, a generative transpose uses `F_full` from
Section 6.3. It generally grows beyond any fine grid that may previously have
existed.

The API should permit this intentionally, for example by allowing a transposed
plan with `target_grid=None` and an explicit
`topology_policy="full_support"` policy.

### 7.3 Transpose onto a requested grid

Encoder-decoder networks often save a fine encoder grid and use it as the
decoder target. This is the transposed relation restricted to that saved grid.
It is not guaranteed that every saved fine coordinate is reachable. Coverage
is determined by the forward relation, not by the fact that the coordinate was
saved.

When the decoder is intended to be the exact transpose of its encoder plan,
`from_plan_transposed` is safer and more informative than reconstructing a new
transposed plan from two grids.

## 8. Coverage across kernel sizes and strides

The full-support topology guarantees that every generated output row has an
edge. It does not guarantee that every input fine coordinate participates in a
strided forward convolution.

For one axis, define the covered residue set:

```text
R = {(a + r(u)) mod S | u = 0, ..., K - 1}.
```

A fine coordinate `p` can participate in the infinite-lattice forward operator
only if `p mod S` is in `R`. Three-dimensional coverage requires this condition
on every axis.

Consequences:

- With dilation one and `K >= S`, all residues are covered.
- With dilation one and `K = S`, every residue is represented exactly once per
  axis. Every fine coordinate has exactly one spatial forward connection.
- With `K < S`, some fine coordinates are intentionally skipped, exactly as in
  dense strided convolution. A transpose cannot populate those forward-null
  columns.
- With `K > S`, some residues occur at multiple taps, so an input may contribute
  to multiple coarse coordinates.
- With dilation, coverage depends on the residues generated by `D`; `K >= S`
  alone is no longer sufficient.

### 8.1 Round-trip reachability law

For every geometry and fine active set `F`, let `C_full` be its generated
forward topology and let `F_round` be the generated full-support transpose of
`C_full`. Then:

```text
{p in F | there exist q,u with p = a + S*q + r(u)} is a subset of F_round.
```

Every coordinate in that left-hand set also has degree at least one in the
exact transpose of the forward plan onto `F`. The proof is constructive: if
`(p,q,u)` generated `q` in the forward projection, the identical triple emits
`p` in the transpose projection and becomes a reversed plan edge.

This is the general invariant violated by issue #668. It must be checked across
the complete geometry matrix, not only for `K=S=4`. The residue condition keeps
the statement correct for `K<S` and dilated footprints.

### 8.2 Proactive coverage diagnostics

Plan construction should compute `R` at constant geometry cost. If any axis
does not cover every stride residue, emit a once-per-geometry warning by
default stating that the convolution can ignore fine coordinates, that this
matches dense Torch sampling, and that `coarsened_grid` or pooling should be
used when block aggregation is intended. Provide an explicit way to acknowledge
or suppress the warning for intentional sampling.

The static warning describes possible holes. After the rulebook is built, the
coverage report should separately give the actual zero-degree source count for
the supplied grid. This distinction avoids claiming data loss when all active
coordinates happen to lie on covered residues while still making `K=1, S>1`
and similar migrations self-explaining.

## 9. World-space transforms and lattice registration

### 9.1 Keep index geometry and world registration separate

For a fine grid with voxel size `h_f` and origin `O_f`, fVDB defines the world
position of integer coordinate `p` as:

```text
T_f(p) = O_f + h_f .* p.
```

The convolution output is a sampled feature lattice, not a partition of the
fine voxels into larger physical cells. The coarse output coordinate `q` is
registered to the fine anchor `a + S*q`. Therefore:

```text
h_c = S .* h_f
O_c = O_f + h_f .* a.
```

The canonical generated convolution grid uses `a = 0`:

```text
h_c = S .* h_f
O_c = O_f.
```

For a generated transposed grid, the inverse registration is:

```text
h_f = h_c / S
O_f = O_c                 # canonical a = 0
```

This policy intentionally leaves even-kernel asymmetry in `r(u)`. Shifting the
world origin by half a voxel to label the receptive-field centroid would make a
stride-one even-kernel output cease to be registered to the same Torch index
lattice as its input. Kernel phase belongs to operator geometry, not hidden
transform metadata.

### 9.2 Validation for explicit grids

For each batch element, a fine/coarse grid pair is convolution-compatible only
if:

```text
h_c / h_f == S
a = (O_c - O_f) / h_f is integer-valued.
```

Comparisons need a documented floating tolerance, but the resulting `a` must be
rounded once and stored as an integer in the plan. The connection law must then
use that stored `a`. A transform must never be accepted and subsequently
ignored.

The first correctness release will support only uniformly `a = 0` and will
reject nonzero or fractional registrations. `ConvolutionGeometry` should retain
the conceptual field, but the evaluator should not pay for a device-side
per-batch phase lookup before nonzero registration has a concrete use case and
performance review. A later extension may support integer `a` per batch
element without changing the equations in this document.

Examples in one dimension:

| Fine `(h, O)` | Coarse `(h, O)` at `S=2` | Result |
|---|---|---|
| `(1, 0)` | `(2, 0)` | Valid, canonical `a=0`. |
| `(1, 0)` | `(2, 1)` | Mathematically integer `a=1`; rejected in the first release and reserved for a future extension. |
| `(1, 0)` | `(2, 0.5)` | Invalid for an integer-sampled Torch convolution lattice. |
| `(1, 0)` | `(1, 0)` | Invalid scale for stride two. |

### 9.3 Why `coarsened_grid` is different

`coarsened_grid(S)` represents grouped physical cells. Its current origin shift,
`0.5 * (S - 1) * h_f`, places a coarse cell at the centroid of an `S`-wide fine
block. That is appropriate for pooling and geometric aggregation.

A convolution lattice uses Torch output anchors and its kernel footprint may
overlap, leave gaps, or extend beyond such blocks. The two operations may
occasionally produce compatible coordinates, but they must not share a shortcut
or silently substitute for one another.

In particular, an even coarsening factor produces a half-integer block-center
shift and is not an integer-registered convolution lattice under this proposal.
This is not an error in coarsening; it is evidence that the grids represent
different concepts.

## 10. Independent dense PyTorch oracle

The oracle must call PyTorch without importing or reusing production topology
builders. It should run per batch element because sparse batches may have
different coordinate bounds and transforms.

### 10.1 Forward, no-border canvas

Let `p_min` and `p_max` be the componentwise bounds of the active fine set. Let
`r_min = -P_before` and `r_max = D*(K-1)-P_before`.

Compute a deliberately conservative output-coordinate canvas:

```text
q_canvas_min = floor_div(p_min - a - r_max, S)
q_canvas_max = ceil_div(p_max - a - r_min, S).
```

These bounds include every output that can overlap the input and make the dense
input canvas include the complete active-coordinate bounding box. They may add
one zero output at either end, which is useful: a stride residue that dense
convolution does not sample remains visibly zero instead of being omitted by
the canvas construction. For a nonempty input:

1. Create a zero dense input canvas whose global fine-coordinate bounds are
   `a + S*q_canvas_min + r_min` through
   `a + S*q_canvas_max + r_max`.
2. Inject the sparse structural mask or actual features at their global fine
   coordinates.
3. Call `torch.nn.functional.conv3d` with `padding=0`, the requested stride, and
   dilation.
4. Label dense output index `j` with global coarse coordinate
   `q_canvas_min + j`.

The bounded conservative canvas exposes every output receptive field whose
bounding box can overlap the input bounds, plus at most the intentional zero
margin described above. PyTorch returns zero at holes, and the positive
all-one outputs select the exact sparse support.

This avoids relying on PyTorch's finite-tensor output-size convention as a
topology policy. It also avoids `padding="same"`, which PyTorch does not support
for stride greater than one. The local kernel phase remains the same.

### 10.2 Transpose, no-border canvas

Let `q_min` be the minimum active coarse coordinate. Densify the coarse input
over its bounding box and call:

```python notest
torch.nn.functional.conv_transpose3d(
    dense_input,
    fvdb_weights.transpose(0, 1).contiguous(),
    stride=stride,
    padding=0,
    dilation=dilation,
    output_padding=0,
)
```

Label dense output index `j` with global fine coordinate:

```text
p = a + S*q_min + r_min + j.
```

Using `padding=0` is intentional: it retains the full spread of every input
instead of cropping the output at a finite tensor border. `output_padding` is
not a connectivity mechanism. PyTorch documents that it only resolves output
shape ambiguity and does not add values or zero padding to the output.

### 10.3 Topology and value passes

Run two distinct oracle passes:

- Structural pass: one scalar channel, value one at every active coordinate,
  all-one kernel, no bias. Compare integer counts exactly and take `> 0` for
  topology.
- Numerical pass: actual features and weights on the same coordinate-aware
  canvas. Compare values only after topology has independently passed.

The structural pass must use positive quantities so that cancellation cannot
hide support. Small CPU cases should use float64 and counts small enough to be
exactly representable.

### 10.4 Bound oracle allocation

The dense oracle is for small, adversarial correctness cases, not realistic
sparse-scene bounding boxes. Before allocating, the helper must compute both
spatial site counts and the byte size of every explicit input, output, weight,
and gradient tensor.

Use these default hard limits in the test helper:

```text
MAX_DENSE_ORACLE_SPATIAL_SITES = 2**20       # for either input or output canvas
MAX_DENSE_ORACLE_TENSOR_BYTES = 64 * 2**20  # combined explicit tensor payload
```

The helper must raise a dedicated error with the requested shape and estimated
bytes before allocation if either limit is exceeded. Required CI cases must be
designed below the limits rather than silently skipped. Large-bounding-box
coverage belongs in the sparse scalar relation oracle, tiled impulse cases, or
property tests that do not allocate the bounding volume. The limits may be
lowered by constrained test environments, but raising them should be an
explicit local choice.

## 11. Worked reproduction of issue #668

The three-dimensional failure is the Cartesian product of a one-dimensional
endpoint error.

Take:

```text
fine active coordinates p = 0, ..., 15
K = 4
S = 4
D = 1
a = 0
r(u) = -1, 0, 1, 2
```

The complete forward output and its all-one counts are:

| q | Fine receptive coordinates | Count on `p=0..15` |
|---:|---|---:|
| 0 | `-1, 0, 1, 2` | 3 |
| 1 | `3, 4, 5, 6` | 4 |
| 2 | `7, 8, 9, 10` | 4 |
| 3 | `11, 12, 13, 14` | 4 |
| 4 | `15, 16, 17, 18` | 1 |

Thus full-support `conv_grid(4, 4)` must contain `q=0..4`, not `q=0..3`.

The current `K == S` fast path uses floor coarsening and produces only `0..3`.
The current evaluator uses `r=-1..2`, so transposing from that incomplete coarse
grid reaches only fine coordinates `-1..14`. Fine coordinate 15 is a genuine
zero column of that incorrectly assembled finite plan.

For a `16^3` fine cube, the number of original coordinates lying on at least one
far face is:

```text
16^3 - 15^3 = 721.
```

That is the observed zero count. It is not an accumulation bug in the current
gather/scatter evaluator. It is a mismatch between the topology produced for
the forward lattice and the phase evaluated by the plan.

With the correct `5^3` coarse topology, the transpose restricted to the original
`16^3` fine grid gives every fine row one spatial edge because `K=S=4` covers
every stride residue exactly once.

The current unshifted `K == S` shortcut is therefore invalid unless each
admitted case is derived from the canonical phase-aware relation. In fact, the
`K=S=2` proof generalizes. At dilation one, on each axis where `K=S`:

```text
p - a + P_before = K*q + u, with 0 <= u < K
q = floor((p - a + P_before) / K)
u = (p - a + P_before) mod K
```

Euclidean quotient and remainder give exactly one `(q,u)` for every integer
`p`, including negative coordinates. Thus when `K=S` componentwise, forward
full support has a relation-proved `O(N)` direct projection for every kernel
size, not merely size two. The current ordinary floor formula is the special
case `P_before=0`, which covers sizes one and two. For uniform `K=S=2`, it gives
`p=-1 -> q=-1` and `p=-2 -> q=-1`, so the existing low-memory coordinate path
should remain unchanged behind a proof test. For `K=S=3` or `4`, the optimized
path must include the `P_before` shift rather than falling back to unshifted
coarsening.

The exact `K=S` transpose specialization emits the phase-aware block
`p=a+K*q-P_before+u`. The existing unshifted pure-subdivision path remains
exact for sizes one and two; larger kernels keep the same `K_volume` output
cardinality but require the shifted base.

The broad `K == 1` shortcut is wrong when `S>1`: a dense one-tap strided
convolution samples only one stride residue, whereas floor coarsening assigns
every fine coordinate to a coarse block. Slice 3a should replace the two broad
predicates with these relation-derived `K=S` projections and add the stronger
`K=1,S=1` public identity return described in Sections 12.5 and 17.1. No
optimized form may survive without a proof test against the relation for both
negative and positive coordinates.

## 12. Proposed internal architecture

### 12.1 `ConvolutionGeometry`

Create one immutable internal geometry value shared by Python-visible plans and
C++/CUDA builders. It should contain:

- kernel size;
- stride;
- dilation;
- `P_before` and `P_after` or a named phase policy;
- registration offset `a`, stored and asserted as uniformly zero in the first
  release, with per-grid nonzero support deferred;
- a convolution-semantics version.

It should provide host/device-safe helpers for:

- mapping a linear tap index to `u`;
- computing `r(u)`;
- computing fine `p` from coarse `q` and tap `u`;
- testing whether a fine `p` maps to a coarse `q` for a tap;
- correct divisibility and floor/ceiling behavior for negative coordinates;
- transform compatibility.

No topology builder or evaluator should contain a second hand-written kernel
start formula. This includes accelerated paths whose present odd-kernel
restrictions happen to make a local `-K/2` formula correct.

### 12.2 Normalized fine/coarse rulebook

Represent the convolution graph once, independent of execution direction. A
rulebook edge contains or implies:

```text
(batch, fine_index, coarse_index, tap_index).
```

The compact representation may remain grouped by tap for efficient GEMMs. Its
normative invariants are:

- each stored edge satisfies the geometry equation;
- each valid fine/coarse/tap triple appears exactly once;
- indices are in bounds for their named domains;
- offsets are monotone and terminate at the exact pair count;
- topology construction is deterministic as a set even if storage ordering is
  backend-specific.

Forward execution gathers fine and scatters coarse. Transposed execution
gathers coarse and scatters fine. Backward operations follow from those same
edges.

### 12.3 Plans

A plan should own:

- normalized fine and coarse grids;
- `ConvolutionGeometry`;
- one normalized rulebook;
- execution direction;
- resolved `full_support` or `restricted` topology policy;
- channel-pair constraints;
- coverage diagnostics;
- backend realization of the same rulebook.

`from_plan_transposed` flips direction and swaps public source/target views. It
must not call the topology builder again.

### 12.4 Topology builders

Both generated-grid builders should call the same geometry helpers used by the
rulebook:

- forward builder: enumerate `(p, u)`, retain divisible solutions, and emit
  `q = (p - a - r(u)) / S`;
- transpose builder: enumerate `(q, u)` and emit `p = a + S*q + r(u)`;
- deduplicate emitted coordinates per batch;
- attach the transform derived in Section 9.

Topology correctness does not authorize an unbounded candidate buffer. Let `N`
be the number of active forward-input voxels and let:

```text
M = number of valid (p,u) emissions before output-coordinate deduplication.
```

The forward CUDA builder must use `O(N+M)` auxiliary storage, not
`O(N*K_volume)`, when most tap candidates fail divisibility. Use the evaluator's
known two-sweep shape:

1. Count valid tap emissions for each input voxel using the shared relation.
2. Prefix-sum those counts with overflow-safe offsets and preflight the exact
   allocation size.
3. Allocate coordinate and batch-index arrays for exactly `M` rows.
4. Fill those arrays in a second sweep, then deduplicate them into the output
   grid.

This removes the full candidate mask and makes peak staging memory proportional
to realizable emissions. For dilation one, `a=0`, and `K=S` on every axis,
exactly one tap tuple matches each `p`, so `M=N` even for `K=S=3` or `4`.
The general two-sweep algorithm may still perform `O(N*K_volume)` divisibility
work. The phase-aware quotient formula in Section 11 must handle componentwise
`K=S` directly, however, making issue #668's forward projection `O(N)` in both
work and staging rather than paying 64 tap checks per input.

Generative transpose has `M=N*K_volume` because every tap is a real emission;
its current subdivision/general paths already stage that order of data, so
correcting the shortcut's phase does not create the same asymptotic regression.
It still requires an allocation preflight, and a chunked fill is an allowed
fallback for genuinely large `M`.

The relation-derived componentwise `K=S` specializations from Section 11 bypass
general enumeration. Forward computes the phase-aware quotient directly;
transpose emits the phase-aware block directly. For sizes one and two these may
reuse the existing floor-coarsening and pure-subdivision coordinate logic. For
larger sizes they must include `P_before`; concretely, the transpose subdivision
base changes from `S*q` to `a+S*q-P_before` without changing its allocation
size. All variants attach convolution transforms, not the block-centroid
transforms of coarsening and refinement helpers. The all-one and scalar-relation
tests remain mandatory.

Beyond these proved cases, the builders may be optimized independently only
after their output is checked against the scalar reference and all-one Torch
oracle and their peak-memory complexity is checked against `O(N+M)`.

### 12.5 Backends

Backend selection may change storage and execution strategy only. It may not
select a different geometry. Every backend should be testable against the same
frozen rulebook or at least against an exact exported edge set.

The dense backend should be treated as an implementation backend, not as the
test oracle. The independent oracle must remain simple test code so that a bug
in production densification does not certify itself.

Three existing backend boundaries require explicit treatment:

- `_MatmulBackend` is valid for `K=1, S=1` only when source and target have the
  identical topology and ordering. In the first release, the sufficient guard
  should be shared `GridBatchData` identity: `is_same` is pointer identity, not
  a content-and-ordering comparison. The canonical `conv_grid(K=1,S=1)` is the
  identity relation and both `Grid.conv_grid` and `GridBatch.conv_grid` should
  return the source object itself instead of rebuilding an equal grid. This
  preserves the automatic fast path and makes the public identity behavior
  explicit. A distinct explicit target, even if content-equal, should
  conservatively use the normal rulebook unless a real ordered-topology
  equality predicate is added later. Returning `source.jagged_like` while
  reporting a different target is invalid.
- The current dense forward backend is semantically valid only for its
  coincidental admitted geometry: `K=3`, `S=1`, and the same source/target
  grid. It hardcodes `padding=1` without enforcing `K=3`.
- The current dense transposed path additionally passes fVDB
  `[C_out, C_in, ...]` weights directly to PyTorch's
  `[C_in, C_out, ...]` interface. Unequal channel counts normally fail. Equal
  counts merely make the call shape-compatible: it still computes the
  transposed channel map and is correct only for a single channel or a weight
  matrix symmetric in the two channel axes at every spatial tap. It must be
  adapted exactly once and restricted to supported geometry, or the dense
  backend must be disabled in the first release.
- `PredGatherIGemm` currently uses `-K/2`, but its enforced odd uniform kernels
  `{3,5,7}` and strides `{1,2}` make that equivalent to `-P_before` and avoid
  all current shortcuts. It may remain enabled under those restrictions. The
  restrictions must not be relaxed until the backend consumes shared geometry
  or has an equivalent pinned phase assertion.

The default gather/scatter realization is already direction-agnostic after the
rulebook's arrays are oriented for execution. Its duplicated forward and
transposed backward structs should be replaced by one implementation; retaining
two byte-equivalent paths creates exactly the future-skew surface this design
is intended to remove.

The equal-channel dense-transpose distinction has a two-channel witness. For a
single active center tap, let the public fVDB channel matrix and input be:

```text
V = [[1, 2], [3, 4]]     x = [5, 7]
```

The fVDB contract gives `V*x=[19,43]`. Passing `V` directly to PyTorch's
transpose interface interprets its channel axes oppositely and gives
`V^T*x=[26,38]`. Equal dimensions avoid a shape exception but do not adapt the
layout. The equality holds only when `V=V^T` for every tap (including the
one-channel case).

## 13. Test strategy

### 13.1 Test layers

#### Layer A: pure integer geometry

Implement a small, obviously scalar Python reference that enumerates coordinates
and taps directly from the equation. It must have no dependency on fVDB topology
operations.

Check exact coordinate sets, exact edge sets, tap IDs, degree vectors, negative
division, and residue coverage.

#### Layer B: independent dense Torch oracle

Use the no-border canvases in Section 10. Check:

- all-one counts exactly;
- positive-support topology exactly;
- actual forward and transpose values;
- input, weight, and output gradients;
- fVDB/Torch channel and spatial-axis adapters.

All cases must pass the allocation preflight in Section 10.4.

#### Layer C: impulse basis tests

Impulse tests are the primary localization tool:

- one active input and one active kernel tap identifies one expected edge;
- sweep every tap, including even kernels;
- use unique asymmetric tap values to expose spatial flips and permutations;
- use multiple separated impulses to isolate copies of the stencil;
- use colliding impulses to verify additive accumulation;
- place impulses at every stride residue;
- repeat near zero and at negative coordinates.

For a linear operator on a fixed topology, the impulse basis is a complete
behavioral characterization. Random inputs are useful stress tests but are not
the semantic authority.

#### Layer D: algebraic transpose tests

For random finite sparse domains and float64 CPU data:

- export or reconstruct the forward edge matrix;
- require `from_plan_transposed` to contain its exact reversed edges;
- check the zero-bias dot-product identity using
  `W_adjoint=W.transpose(0, 1).contiguous()` at the fVDB transposed-plan API;
- compare the transposed plan with PyTorch autograd's input gradient;
- check that applying a transpose is not described or tested as inversion.

#### Layer E: transform tests

Check:

- `voxel_size_out == stride * voxel_size_in` for generated forward grids;
- canonical origin preservation;
- generated transpose is the inverse lattice registration;
- nonzero integer registration phase is either honored or rejected explicitly
  during the staged rollout;
- fractional phase and scale mismatch fail before topology construction;
- anisotropic sizes, origins, strides, and per-batch metadata;
- `voxel_to_world(q)` agrees with the fine anchor `a + S*q`;
- convolution grids and coarsened grids retain distinct transform contracts.

#### Layer F: backend parity

Run the same cases through every supported backend. Compare topology or exported
edges exactly, then values and gradients within narrow dtype-specific
tolerances. A backend-specific limitation must fail at plan construction; it
must not silently change semantics.

Exercise the constant-time reversed gather/scatter view through both public
directions and both backward paths. These tests must prove that swapping the
index arrays is sufficient and that the metadata direction flag never causes a
second reversal.

### 13.2 Required geometry matrix

The inexpensive CPU reference should cover the Cartesian product broadly. CUDA
tests can use a pairwise subset.

At minimum include:

- kernel sizes `1, 2, 3, 4, 5, 6` on each axis;
- mixed kernels such as `(2, 3, 4)` and `(5, 2, 3)`;
- strides `1, 2, 3, 4, 5`;
- mixed stride `(1, 2, 3)`;
- all three relationships `K < S`, `K = S`, and `K > S`;
- canonical phase; exercise positive/negative integer registration in the pure
  oracle while requiring the first production release to reject nonzero `a`;
- singleton, dense block, hollow block, disconnected clusters, and irregular
  sparse active sets;
- coordinates around `-S`, `-1`, `0`, `S-1`, and `S` to expose signed division;
- forward full support, transpose full support, explicit restriction,
  submanifold restriction, and exact plan transpose;
- the round-trip reachability law in Section 8.1 for every geometry and active
  set;
- one and multiple channels;
- bias off for topology, then bias on for selected-domain value checks;
- batch size one, multiple nonempty grids, and empty grids.

Backend boundary tests must also include `K=1, S=1` with different explicit
source/target topologies, dense kernels other than three, and transposed dense
execution with both unequal channel counts and equal channel counts using an
asymmetric channel matrix. These cases expose silent mislabeling, wrong
padding, an outright shape error, and the subtler runnable-but-transposed
channel map respectively. Add a `K=1,S=1,target=None` test that requires the
generated target to share source grid data and retain the matmul fast path, plus
a distinct-but-content-equal explicit target that must fall back safely.

Migration-specific geometry tests should prove the two shortcut
characterizations in Section 17.1: `K=1,S>1` canonical support is contained in
the old shortcut support and every removed row has zero pre-bias degree, while
`F={3},K=S=4` produces the disjoint one-dimensional outputs `{0}` and `{1}`
under the old and canonical rules.

Specialization tests must prove the phase-aware quotient and block formulas for
componentwise `K=S` at sizes one through six, including mixed axes and negative
coordinates. They must pin that uniform `K=S=2` reduces to the existing
unshifted floor-coarsening/subdivision result, that `K=3` and `4` apply the
nonzero `P_before` shift, and that `K=1,S=1` returns the original public grid
object while `K=1,S>1` never enters that identity path.

When dilation becomes public, add coprime and non-coprime `(D, S)` cases before
enabling it.

### 13.3 Exactness and tolerances

- Coordinate sets, tap identities, edge sets, and degree counts are exact.
- Scalar integer-valued impulse cases should be exact in float32 while their
  accumulation remains exactly representable.
- CPU numerical oracle tests should prefer float64.
- GPU tests must disable TF32 in correctness lanes.
- Floating tolerances should be derived from dtype and accumulation depth, not
  a blanket large tolerance.
- A topology mismatch must never be converted into a numerical tolerance.

### 13.4 Repair the current test infrastructure

The existing impulse-first direction is correct, but several helpers encode
odd-kernel assumptions:

- replace raw `K // 2` formulas with `ConvolutionGeometry` in test adapters;
- stop constructing an even-kernel range with `K + 1` positions;
- replace the assertion that forward and transpose generated topologies are
  identical at stride one with the actual relation. They coincide for symmetric
  odd footprints, but are reflected for the chosen even phase;
- make the dense canvas carry an explicit global coordinate origin;
- add even and mixed-axis cases to every core topology/value suite;
- retain random tests only as supplemental stress coverage.

The test oracle should remain separate from production geometry code even after
the production code is unified.

### 13.5 Landing expected failures without red CI

Slice 1 should land the independent scalar and dense-oracle tests as ordinary
passing tests. A production-facing Python assertion that demonstrates behavior
scheduled for Slice 3a, 3b, or 5 should carry both:

```python notest
@pytest.mark.conv_semantics_pending(slice="3a", issue=668)
@pytest.mark.xfail(strict=True, reason="issue #668; remove in Slice 3a")
```

Register `conv_semantics_pending` in `pyproject.toml`, specifically by adding a
`markers` entry under the existing `[tool.pytest.ini_options]` table, so it is
searchable and does not produce unknown-marker warnings. `strict=True` makes an
early fix produce an `XPASS` failure, forcing the implementing slice to remove
the marker rather than silently accumulating stale expected failures. Use
`slice="3b"` for transform-registration failures and `slice="5"` for
backend-boundary failures.

```toml
markers = [
  "conv_semantics_pending: temporary strict-xfail tracking for issue #668",
]
```

Do not land known-failing C++ tests under a permanent `DISABLED_` name. Pure
reference tests can land in Slice 1; production C++ assertions that cannot use
strict xfail should land atomically with the corresponding Slice 3a, 3b, or 5
implementation. Before release, no `conv_semantics_pending` marker, related
strict xfail, or campaign-specific disabled test may remain.

### 13.6 Topology-construction resource tests

Peak memory is a release property for topology builders, not merely a benchmark
annotation. Add test-visible accounting for `N`, `K_volume`, exact emission
count `M`, and requested staging bytes, then measure the complete builder above
an idle baseline.

The benchmark matrix must include forward `K=S` at sizes two, three, and four;
the issue #668 geometry; representative `K<S` and `K>S` cases; and generative
transpose. Sweep `N` geometrically and compare multiple kernel volumes so that
both input scaling and candidate-volume scaling are visible. Report:

- exact candidate and output counts;
- peak live CUDA bytes above baseline, with peak reserved bytes as a secondary
  allocator diagnostic;
- peak CPU resident bytes for the CPU path;
- topology-construction wall time;
- final grid size separately from transient staging.

For the current masked forward path, the `K=S` diagnostic is:

| Uniform `K=S` | Candidate rows per input | Valid emissions per input | Candidate staging bytes per input |
|---:|---:|---:|---:|
| 2 | 8 | 1 | 136 |
| 3 | 27 | 1 | 459 |
| 4 | 64 | 1 | 1,088 |

The last column is the 17-byte coordinate, batch-index, and mask lower bound;
it excludes compacted output and grid construction. The replacement should
stage one 16-byte coordinate/batch row per valid emission, plus count/offset
bookkeeping, rather than paying these rejected-candidate multipliers.

For forward dilation-one `K=S`, assert `M=N` and require staging to remain
`O(N)`. Uniform `K=S=2` must exercise the proved floor-coarsening specialization
and show no topology-memory regression. Uniform `K=S=3` and `4` must exercise
the phase-aware direct projection without allocating a full `N*K_volume`
coordinate or mask tensor. At least one non-specialized sparse-emission case,
such as `K=3,S=4`, must exercise count-then-fill and demonstrate that allocation
tracks exact `M` rather than the rejected candidate volume.

CI need not allocate a ten-million-voxel grid, but the report must include an
analytical checked-arithmetic row for it. At `N=10,000,000` and `K=4`, the
current CUDA candidate arrays alone require:

```text
N * 4^3 * (12 coordinate bytes + 4 batch-index bytes + 1 mask byte)
= 10,880,000,000 bytes
= approximately 10.13 GiB
```

That is a lower bound before masked-compaction output and grid construction,
versus 160,000,000 bytes (approximately 153 MiB) for one coordinate and batch
row per input. A replacement that merely moves this allocation or relies on a
larger GPU fails the resource test.

The measured release-gate results are recorded in
[Convolution Semantics Resource Report](convolution_semantics_resource_report.md).

## 14. Current-code audit against the proposal

The present implementation contains multiple locally reasonable conventions
that agree for the mostly odd-kernel tests and diverge elsewhere:

1. `GatherScatterDefault.cu` is already the canonical `a=0` relation in both
   directions, not merely the correct kernel start. Forward probes
   `p=S*q+r(u)`; transpose tests divisibility of `p-r(u)` and probes
   `q=(p-r(u))/S`. Its `floor(-K/2+1)` start matches the proposed Torch phase.
   Forward and transpose dispatch the same execution operation; their backward
   structs are byte-equivalent in substance. `topo.direction` is checked by
   the public wrappers but is not read by the execution math.
2. `BuildGridForConv.cu` uses a floor-coarsening shortcut when `K == S` or
   `K == 1`. It does not implement the evaluator relation for `K=S>=3` or for
   `K=1, S>1`; its componentwise dilation-one `{1,2}` subfamily is exact.
3. The general even-kernel forward grid path enumerates `p+d=S*q` with
   `d in [0,K-1]`, giving the effective relation offset
   `r=-d in [-(K-1),0]` rather than the evaluator footprint.
4. The general `BuildGridForConvTranspose.cu` path directly emits
   `r in [0,K-1]` for even kernels. That matches the evaluator for `K=2` but
   not for even `K>=4`. Its pure-subdivision shortcut for `K == S` or `K == 1`
   is exact only in the proved componentwise dilation-one `{1,2}` subfamily;
   it is wrong for `K=S>=3` and `K=1,S>1`.
5. Both generated convolution grid paths copy the source voxel size and origin
   unchanged, even when stride changes lattice spacing.
6. `from_plan_transposed` rebuilds a gather/scatter topology. That rebuild is an
   exact edge reversal today because the evaluator implements the same relation
   in both directions; it is redundant work and a future-skew hazard, not a
   current source of issue #668.
7. Test helpers encode multiple even-kernel ideas: the strided forward topology
   helper uses a `K+1` symmetric interval, while transpose and dense strided
   helpers use the `K`-tap `k-K//2` phase. Neither consistently matches the
   evaluator for even kernels.
8. `_MatmulBackend` is selected for every `K=1,S=1` plan before target-grid
   compatibility is checked. Execution returns data shaped and labeled like the
   source, so a different explicit target is silently reported but not honored.
   The available `is_same` predicate is only `GridBatchData` pointer identity,
   and today's automatic `conv_grid(1,1)` rebuilds a new data object instead of
   preserving the identity relation.
9. The dense backend hardcodes `padding=1` for every kernel size. Its transposed
   path also passes public fVDB weights directly to PyTorch without swapping
   channel axes, causing an error for unequal channel counts. Equal counts only
   make the call runnable: asymmetric channel weights still produce the wrong
   map. The direct path is accidentally correct only for a single channel or
   per-tap channel-symmetric weights.
10. `PredGatherIGemm.cu` contains another start formula, `-K/2`. It is correct
    under the backend's current odd-kernel restriction `{3,5,7}`, but would
    silently choose the rejected even phase if that restriction were relaxed.
11. Core semantic suites concentrate on odd kernels, allowing the phase split
    and backend boundaries to remain hidden.
12. The two default gather/scatter backward structs duplicate the same
    direction-independent computation. This is not a current numerical bug,
    but it is dead copy-paste geometry surface that can drift later.
13. The general CUDA forward-grid path allocates coordinates, batch indices,
    and a validity mask for all `N*K_volume` candidates before compaction. The
    broad `K==S` shortcut currently hides that cost. Removing it naively would
    make issue #668's `K=S=4` builder stage at least 17 bytes times 64 candidates
    per input voxel even though exactly one tap tuple is divisible. The CPU path
    has a corresponding `O(N*K_volume)` work increase but not this same masked
    candidate allocation. Generative transpose already emits every one of its
    `N*K_volume` candidates, so its asymptotic staging is unchanged.

The immediate issue is therefore not evidence that the impulse-first approach
failed. The impulses exposed that grid builders, shortcuts, test oracles, and
specialized backends had never been forced to agree with the evaluator that
already implements the intended relation.

## 15. Alternatives considered and rejected

| Alternative | Why it is rejected |
|---|---|
| Treat Minkowski Engine as the authority. | The project goal is dense Torch equivalence. Another sparse library may make a different but internally valid topology choice. |
| Use `-K//2` for every even kernel. | It gives `[-2, -1, 0, 1]` for `K=4`, while Torch `same` and the current evaluator use `[-1, 0, 1, 2]`. |
| Change kernel anchoring as a function of stride, for example from `K-S`. | The same dense kernel sampled at a different stride does not change tap phase. This would make weights mean different spatial offsets when stride changes. |
| Define every `K == S` case as ordinary block coarsening/subdivision. | It conflicts with the Torch phase beyond the proved componentwise dilation-one `{1,2}` family and caused issue #668 directly. |
| Delete the broad forward shortcuts and fall through to the current masked candidate builder. | It restores semantics but stages `N*K_volume` candidates. At `K=S=4`, it spends 64 candidate slots to retain exactly one emission per input and creates a multi-gigabyte regression. |
| Remove the proved `K=S=2` specialization for uniformity. | It changes no semantics, discards a simple signed-coordinate proof, and needlessly replaces an `O(N)` low-constant path with tap enumeration. |
| Generate transpose topology with one convention and evaluate it with another. | A topology with no corresponding evaluator edges creates silent zero rows. Both must project the same relation. |
| Rebuild a plan when transposing it. | The finite transpose is already known exactly by reversing edges. Today's gather/scatter rebuild preserves it, but adds work and an avoidable future-skew surface. |
| Spatially flip the kernel as the primary transpose definition. | It obscures tap identity and fails easily for stride and asymmetric even footprints. Edge reversal is direct and testable. |
| Use PyTorch `output_padding` to recover missing sparse coordinates. | PyTorch specifies `output_padding` as output-shape disambiguation, not added connectivity or values. It cannot repair a missing forward column. |
| Shift generated world origins to the receptive-field centroid. | For even kernels it would unregister stride-one output index `q` from Torch input index `q`. The asymmetry is already explicit in `r(u)`. |
| Preserve source voxel size for strided output. | It labels a stride-`S` sampled lattice as though adjacent output indices were one fine voxel apart, breaking composition and world queries. |
| Infer topology from actual nonzero feature values. | Plans are structural and reusable. Cancellation and stored zeros would make topology data-dependent and unstable. |
| Let bias define generated support. | Nonzero bias has infinite support before a finite output domain is chosen. |
| Accept arbitrary source/target transforms but ignore them. | Values may look plausible in index space while representing the wrong world-space alignment. Invalid registration must fail closed. |

## 16. Implementation and review sequence

Each slice should be reviewable and should leave one fewer independent source
of geometry. Do not begin with a local patch to the issue reproduction.

### Slice 0: ratify the semantic contract

Deliverables:

- freeze the ratified contract in Sections 5, 6, 7, and 9;
- name the default phase `torch_same_phase` or an equally explicit term;
- decide the compatibility window for legacy behavior;
- record the PyTorch version probe for even kernels;
- agree that full support means no finite border cropping;
- ratify symmetric `full_support` and `restricted` topology policy names;
- distinguish independent transposed weights from the exact-adjoint weight
  adapter in Section 5.3;
- fix the first release at fail-closed, uniform `a=0`;
- record that the dense backend must be fixed or disabled, while
  `PredGatherIGemm` may remain under its current restrictions.

Exit criterion: reviewers can compute every edge for a one-dimensional example
without consulting implementation code.

### Slice 1: build independent red tests

Deliverables:

- scalar integer relation oracle;
- coordinate-aware dense Torch oracle;
- issue #668 as a minimal one-dimensional test and original three-dimensional
  regression;
- even-kernel tap sweeps;
- negative-coordinate and residue tests;
- transform expectation tests;
- exact all-one degree checks;
- the universal round-trip reachability law;
- asymmetric-channel adjoint tests using the explicit weight transpose;
- red tests for matmul target mismatch and dense-backend padding/channel layout;
- allocation preflight tests for the bounded dense oracle;
- checked topology-allocation accounting and the Section 13.6 benchmark
  harness, with production resource failures assigned to Slice 3a;
- strict expected-failure mechanics and the tracking marker from Section 13.5
  for production behavior scheduled in Slices 3a, 3b, and 5.

Exit criterion: tests distinguish current behavior from the approved behavior
for the expected cases and pass against the standalone references.

### Slice 2: introduce shared geometry

Deliverables:

- immutable `ConvolutionGeometry`;
- one host/device tap-offset implementation;
- correct signed divisibility helpers;
- stored uniform-zero registration with nonzero `a` rejected;
- transform compatibility calculation in report-only mode;
- removal or explicit odd-kernel guarding of duplicated kernel-start formulas.

Exit criterion: plans store the canonical geometry, evaluator paths consume it
or are mechanically guarded as equivalent, and the report-only transform check
does not reject grids still emitted by the legacy builders.

### Slice 3a: unify index-space topology

Deliverables:

- forward topology projection from the canonical relation;
- transpose topology projection from the canonical relation;
- replacement of the broad `K == S` and `K == 1` predicates by proved
  specializations only;
- a componentwise dilation-one `K=S` direct projection using the phase-aware
  quotient/block formulas, retaining the existing unshifted implementation
  when every axis has size one or two, especially uniform `K=S=2`, with
  signed-coordinate proof tests;
- an identity-preserving `conv_grid(K=1,S=1)` short-circuit that returns the
  public source object rather than rebuilding it;
- a count-then-fill CUDA forward path for non-specialized geometries, with
  `O(N+M)` staging, overflow-safe allocation preflight, and the Section 13.6
  resource tests;
- symmetric `full_support` and `restricted` factory behavior;
- proactive residue warnings and actual coverage reporting;
- internal positive-degree assertions;
- removal of all Slice 3a expected-failure markers.

Exit criterion: all generated topologies and exact all-one counts match both
independent references for the full CPU matrix and selected CUDA matrix, issue
#668 is closed in index space, forward staging obeys `O(N+M)`, and transform
compatibility remains report-only.

### Slice 3b: correct world registration and enforce it

Deliverables:

- corrected generated forward voxel sizes `h_c=S*h_f` and canonical origins;
- inverse generated-transpose registration;
- explicit-grid transform diagnostics made hard failures in the same change as
  the generated-transform correction;
- anisotropic and per-batch transform tests;
- migration diagnostics for callers that supplied formerly ignored transforms;
- removal of all Slice 3b expected-failure markers.

Exit criterion: every generated grid satisfies Section 9, incompatible
explicit grids fail before topology construction, and all transform tests pass.

Splitting 3a from 3b is intentional. The index-space repair that closes issue
#668 does not depend on changing world metadata, while the registration repair
changes explicit-target compatibility. Each change remains independently
reviewable, revertible, and bisectable; the report-only check from Slice 2 is
the temporary boundary between them. The split is not permission to ship the
intermediate state: Slices 3a and 3b must be included in the same release, with
no release tag or published artifact that contains 3a without 3b. If the main
branch feeds nightly artifacts, use a stacked integration branch or otherwise
withhold those artifacts until both slices land.

### Slice 4: normalize the rulebook and plan transpose

Deliverables:

- direction-independent fine/coarse rulebook;
- edge validation in debug/test builds;
- the small constant-time gather/scatter transpose view described in Section
  7.1, while retaining the explicit-target transpose builder;
- coverage report and strict output-coverage option;
- a generative full-support transposed-plan policy.

Exit criterion: reversed edge equality and dot-product adjoint tests pass for
all topology policies.

### Slice 5: backend convergence

Deliverables:

- gather/scatter forward, transpose, and backward on the normalized rulebook;
- collapse the duplicate default gather/scatter backward structs and document
  that direction is validated by wrappers but not reapplied by the executor;
- accelerated-backend parity;
- `_MatmulBackend` guarded by shared grid-data identity, preserving the
  generated `K=1,S=1` fast path from Slice 3a and sending distinct explicit
  targets through a rulebook;
- dense-backend behavior expressed with the same geometry and corrected weight
  adapter, or the backend disabled until that is true;
- `PredGatherIGemm` kept enabled under its current odd-kernel/stride
  restrictions and prevented from expanding without the shared phase;
- explicit failure for unsupported combinations;
- removal of all Slice 5 expected-failure markers.

Exit criterion: no enabled backend implements a different topology or tap
phase; any temporarily local formula is mechanically constrained and pinned as
equivalent to shared geometry.

### Slice 6: migration, documentation, and release

Deliverables:

- update the basic convolution tutorial with the graph model and issue example;
- document full-support versus explicit/restricted topology;
- correct claims that transpose inverts convolution or always restores values;
- document convolution-lattice versus coarsened-cell transforms;
- release note listing affected configurations;
- checkpoint/model migration note;
- time and peak-memory measurements for topology construction, plus execution
  performance measurements;
- removal schedule for any legacy compatibility option.

Exit criterion: a user can predict topology, values, and transforms from public
documentation, and all release gates in Section 18 pass.

## 17. Compatibility and migration

### 17.1 Expected behavior changes

The numerically stable region is broader than stride one. For all-odd kernels
that do not enter the `K==S` or all-one shortcuts, both current generated
topology sets are already correct at every stride. The forward builder's
effective `r=-d` and the transpose builder's `r=+d` produce the same set because
the odd footprint is symmetric; the evaluator then supplies the correct tap
identities and values. This includes the dominant `K=3,S=1` and `K=3,S=2`
configurations.

The default `SimpleUNet` convolution path is numerically inside that stable
region: it uses kernel three and stride-one convolution, with resolution
changes performed by max-pooling/coarsening and refinement. Its default
convolution topology and values should not change.

Expected changes are:

- even-kernel generated topology paths whose current footprints disagree with
  the evaluator, excluding explicitly proved coincidences such as the retained
  uniform `K=S=2` shortcuts;
- shortcut configurations where `stride == kernel_size` but floor
  coarsening/subdivision is not the canonical relation, including uniform
  `K=S>=3`; uniform `K=S=2` is a proved exact case and keeps its shortcut;
- kernel size `K=1` with `S>1`;
- public object identity for `Grid.conv_grid(1,1)` and
  `GridBatch.conv_grid(1,1)`, which now return their source object;
- world-space metadata for every generated strided convolution grid, whose
  voxel size changes from `h` to `S*h` even when its integer topology was
  already correct;
- explicit grid pairs with incompatible transforms, which begin failing rather
  than being silently interpreted in index space;
- callers relying on the currently invalid matmul-with-different-target or
  general dense-backend behavior.

The `K=1,S=1` identity return changes no coordinates or convolution values, but
it is observable public API behavior. Python object identity and `is_same` now
hold, automatically constructed plans report `has_fixed_topology=True`, and no
new grid allocation occurs. This is the semantic identity law made concrete,
but it must be documented so callers do not discover the aliasing change
accidentally.

The affected regions of the two broad shortcut predicates have different
migration character and must not be described as one class.

For `K=1,S>1`, the new forward topology is always a subset of the old
floor-coarsened topology. If `p=S*q` is active, then `floor(p/S)=q`, so every
new coordinate was already present. Conversely, every removed old coordinate
has zero degree under the evaluator: its only possible probe is `p=S*q`, and
that fine coordinate is inactive. The transpose shortcut has the same property:
canonical output `{S*q}` is a subset of the old subdivided block, and every
removed residue row has zero linear degree. Thus this migration prunes
structurally zero rows and does not change the convolution contribution on any
retained row. It can still remove bias-bearing output rows because bias is
applied after topology selection, so topology and bias-visible behavior must be
called out explicitly.

The `K=1,S>1` population change is data-dependent, not universally a `1/S^3`
ratio: it is unchanged for a fully dense aligned block, can fall to zero for a
grid entirely on uncovered residues, and approaches a `1/S^3` ratio relative
to the old low-density result under a uniform random-residue model. The warning
should describe residue sampling and zero-row pruning, not catastrophic loss of
previously nonzero convolution results.

For `K=S>=3`, neither old nor new forward topology contains the other in
general, so this is a genuine remapping rather than zero-row pruning. In one
dimension, `F={3}` with `K=S=4` gives old floor-coarsened `C_old={0}` but the
canonical phase gives `C_new={1}`; the sets are disjoint. Generated transpose
support likewise changes from the old block phase to the canonical tap phase.
This configuration, together with other even-phase changes, is where a bounded
legacy geometry mode has real migration value.

Exact gather/scatter plan transposition should not change values today; the
reverse view removes redundant rebuilding and makes the existing equivalence
structural.

Weight tensor spatial ordering should not change if the current evaluator's tap
ordering is retained. That minimizes checkpoint migration: affected models
change because their topology and registration were wrong, not because weights
are arbitrarily permuted.

### 17.2 Legacy handling

Correct behavior should become the default. If deployed checkpoints require a
transition path, provide an explicit, versioned legacy geometry mode for a
bounded release window. It must:

- be opt-in;
- be serialized or otherwise visible in model configuration;
- emit a targeted warning for affected kernel/stride combinations;
- never be selected heuristically from grid shapes;
- have a documented removal release.

Topology or plan caches must include the geometry-semantics version so old and
new rulebooks cannot collide.

### 17.3 Documentation language

Use these terms consistently:

- "transposed convolution" or "adjoint connectivity," not "inverse
  convolution";
- "full-support topology" for generated non-bordered output;
- "explicit/restricted topology" for a user target;
- "fine/coarse convolution lattice" for plan geometry;
- "coarsened cell grid" for pooling/coarsening geometry.

## 18. Release gates and falsifiers

The proposal is falsified, or the implementation is not ready, if any of the
following is true:

1. A production edge does not satisfy `p = a + S*q + r(u)`.
2. A valid triple satisfying the equation and both explicit domains is absent
   from the rulebook.
3. A triple appears more than once.
4. A generated output coordinate has degree zero.
5. An all-one degree differs from the independent dense Torch result.
6. Actual sparse values differ from the coordinate-aware dense result outside
   narrow numerical tolerances.
7. A plan transpose is not the exact reverse edge set of its source plan.
8. The dot-product adjoint identity fails.
9. Topology depends on runtime feature values.
10. Stride changes the kernel's tap phase.
11. Negative coordinates behave differently from their positive translated
    equivalents.
12. A backend produces a different relation.
13. A generated grid's transform violates its lattice registration.
14. An incompatible explicit transform is accepted and ignored.
15. Any participating fine coordinate violates the round-trip reachability law
    in Section 8.1 or has zero degree in the exact transpose of its forward
    plan.
16. The concrete `K=4, S=4` issue regression leaves any original `16^3` fine
    coordinate disconnected after exact plan transposition.
17. `_MatmulBackend` is selected without shared grid-data identity, or an
    automatically generated `K=1,S=1` target fails to preserve that identity.
18. The dense backend remains enabled for a geometry or weight layout it does
    not implement exactly.
19. A dense-oracle helper can allocate beyond the Section 10.4 limits without
    failing preflight.
20. The first release accepts nonzero `a` without implementing it in every
    topology and evaluator path.
21. Public documentation still implies that transposed convolution is a value
    inverse or that a saved target is necessarily fully reachable.
22. A gather/scatter executor applies direction as a second geometric swap
    after it has already received execution-oriented index arrays.
23. The forward topology builder allocates `O(N*K_volume)` candidate storage
    when the exact valid-emission count is asymptotically smaller, or any
    dilation-one `K=S` resource test fails to demonstrate `M=N` and `O(N)`
    staging.
24. A released or published artifact contains Slice 3a's index-space semantics
    without Slice 3b's matching world-space registration.

Release readiness additionally requires:

- all relevant Python tests run from `tests/` against the installed package;
- the C++ test suite passes through `./build.sh ctest`;
- correctness tests cover the required geometry matrix;
- no `conv_semantics_pending` marker, related strict xfail, or
  campaign-specific disabled C++ test remains;
- the Section 13.6 topology report includes construction time and peak memory,
  including `K=S` sizes two through four and the checked ten-million-voxel
  estimate; uniform `K=S=2` retains its proved low-memory path;
- execution benchmarks show any regression explicitly, with correctness taking
  precedence over reintroducing an unproved shortcut;
- release notes identify the semantic and transform changes.

## 19. Ratified decisions and remaining review questions

The three external design reviews and supervising re-audits resolve the original
questions as follows:

1. Dense PyTorch cross-correlation, extended to an infinite lattice without
   border cropping, is the numerical authority.
2. The even-kernel phase in Section 5 is ratified and independently verified
   against PyTorch 2.10 for `K=1..6`.
3. The first release is fail-closed at uniform `a=0`. General integer phase
   remains in the equations but is not an implementation requirement yet.
4. Generated transforms preserve the Torch output anchor rather than shifting
   to an even-kernel receptive-field centroid.
5. Both directions name `full_support` and `restricted` symmetrically, and a
   generative full-support transposed plan is part of the target API.
6. `PredGatherIGemm` remains enabled under its current odd-kernel and stride
   restrictions. `_MatmulBackend` uses shared grid-data identity as its
   first-release guard, while public `conv_grid(1,1)` returns its source object
   and preserves that identity. The dense backend is corrected or disabled
   before release; equal transpose channel counts alone are not evidence of
   correctness.
7. Exact plan transpose and an independently weighted transposed convolution
   are both supported meanings, with the weight-layout distinction in Section
   5.3 made explicit.
8. Default gather/scatter direction is expressed by oriented index arrays. The
   direction flag validates the public entry point but must not trigger another
   executor swap, and duplicate backward implementations are removed.
9. Index-space topology unification and world-space registration correction
   land as independently reviewable Slices 3a and 3b. Transform validation
   remains report-only between them and becomes hard only with the generated
   transform fix. They are review slices, not separate release units, and must
   ship together.
10. Production failures carried across slices use a registered tracking marker
    plus strict xfail, and every such marker is removed before release.
11. `K=1,S>1` is documented as zero-row pruning before bias, whereas
    `K=S>=3` can genuinely remap topology and is the stronger case for a legacy
    mode.
12. The general forward CUDA builder uses count-then-fill staging bounded by
    `O(N+M)`, while componentwise `K=S` uses its direct `O(N)` quotient path.
    Peak memory is a release gate; correcting `K=S=4` may not introduce an
    `N*K_volume` candidate allocation.
13. The broad shortcut predicates are removed, but the proved componentwise
    dilation-one `K=S` quotient/block specialization replaces them. The
    existing unshifted path is retained where `P_before=0`, including uniform
    `K=S=2`; larger kernels use the phase-aware shift.

The remaining policy questions are deliberately narrower:

- Is a legacy geometry mode needed at all, and if so for exactly how many
  releases?
- Should the public topology-policy enum land in the first implementation or be
  stored internally while the existing `target_grid is None` spelling remains
  the compatibility surface?
- What performance-regression budget is acceptable for topology generation
  before a relation-proved optimization is required?

## 20. The idea in one paragraph

A sparse convolution is a dense Torch stencil restricted to a finite set of
stored coordinates. The stencil equation creates a bipartite graph between a
fine lattice and a coarse lattice. Propagating an all-one structural mask and
an all-one kernel through that graph gives exact edge counts; its positive
support is the generated topology. Real features and weights are then evaluated
on the same edges. Transposed convolution reverses those edges, while a
generative transpose projects them onto a full fine support and an explicit
target restricts them. Stride changes lattice sampling, not kernel phase, and
world transforms only register the two lattices. Once that graph is the single
source of truth, forward topology, transpose topology, planning, execution,
autograd, backends, and tests can no longer drift into different ideas of the
operation.

## References

- [fVDB issue #668](https://github.com/openvdb/fvdb-core/issues/668)
- [PyTorch `conv3d` documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv3d.html)
- [PyTorch `ConvTranspose3d` documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html)
- [`fvdb/convolution_plan.py`](../../fvdb/convolution_plan.py)
- [`GatherScatterDefault.cu`](../../src/fvdb/detail/ops/convolution/GatherScatterDefault.cu)
- [`BuildGridForConv.cu`](../../src/fvdb/detail/ops/BuildGridForConv.cu)
- [`BuildGridForConvTranspose.cu`](../../src/fvdb/detail/ops/BuildGridForConvTranspose.cu)
- [`PredGatherIGemm.cu`](../../src/fvdb/detail/ops/convolution/PredGatherIGemm.cu)
- [`VoxelSizeUtils.h`](../../src/fvdb/detail/utils/VoxelSizeUtils.h)
- [`convolution_utils.py`](../../fvdb/utils/tests/convolution_utils.py)
- [`simple_unet.py`](../../fvdb/nn/simple_unet.py)
- [`test_conv_ground_truth.py`](../../tests/unit/test_conv_ground_truth.py)
- [`test_conv_transpose_ground_truth.py`](../../tests/unit/test_conv_transpose_ground_truth.py)
