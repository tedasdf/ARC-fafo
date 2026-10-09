


Every layer should have the below 5 cases:

| Pillar / Check | Objective | What It Validates | Example Test / Method |
| :--- | :--- | :--- | :--- |
| **1. Special-Case Correctness** | Anchor advanced layers to basic primitives | Guarantees the generalized layer reduces to standard operations (e.g., `torch.cummax` or identity) at structural limits. | Zeroing out the kernel or setting $\tau \to 0$ to match native cumulative max. |
| **2. Equivariance** | Ensure spatial or group symmetry | Verifies predictions transform correctly under symmetry group transformations (rotations/reflections). | Testing outputs against permuted inputs across $D_4$ group actions (`product(range(4), [False, True])`). |
| **3. Group Weight Correctness** | Map parameters to proper geometric orbits | Confirms parameters route correctly to specific directions based on distance and symmetry constraints. | Asserting expected weight values across directional orbits given distinct parameters. |
| **4. Outputs & Gradients** | Ensure numeric validity and backpropagation | Checks that forward passes yield finite values and backward passes produce non-zero, valid gradients. | Asserting `torch.isfinite` and checking gradient shapes for model parameters and inputs. |
| **5. Adapter Logic & Integration** | Handle dimensions, residuals, and pass-throughs | Validates multi-tensor wrappers, slice operations, and clean handling of dimensions with/without target axes. | Passing synthetic multi-tensors and checking if non-target axes bypass or correctly apply residual additions. |
| **6. Extensions & Diagnostics** | Monitor metadata, timing, and debugging flags | Ensures optional profiling/timing flags record accurately without altering core functional outputs. | Monkeypatching timers or checking that timing dictionaries populate correctly when requested. |


| Testing Pillar / Check | Morphological / LSE (`test_lse.py`) | D4 Direction Share (`test_d4.py`) | Triton GPU (`test_triton.py`) | Shift Primitives (`test_shift_primitives.py`) | Shift Integration / Glue (`test_shift_glue.py`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Special-Case Correctness** | ✅ **Present** (Check with `reference_direction`) | ❌ **Not Applicable** | ✅ **Present** (Tested via reference parity) | ✅ **Present** (Tested via delta pulse shifts) | ❌ **Not Applicable** (Glue logic) |
| **2. Equivariance** | ✅ **Present** | ✅ **Present** (`test_d4_direction_share_equivariance`) | ❌ **Not Applicable** | ❌ **Not Applicable** | ❌ **Not Applicable** |
| **3. Group Weight Correctness** | ❌ **Not Applicable** | ✅ **Present** (`test_d4_direction_share_orbit_weights`) | ❌ **Not Applicable** | ✅ **Present** (Tied conv orbit alignment) | ❌ **Not Applicable** |
| **4. Outputs & Gradients** | ✅ **Present** (Regression output/grad check) | ✅ **Present** (`test_d4_direction_share_parameter_gradients`) | ✅ **Present** (Comprehensive input/kernel grad & layout checks) | ❌ **Not Present** (Pure forward pulse check) | ❌ **Not Present** (Missing backward loss and input grad verification) |
| **5. Adapter Logic & Integration** | ✅ **Present** | ✅ **Present** (`test_d4_adapter_axes_residual_and_passthrough`) | ❌ **Not Applicable** | ❌ **Not Applicable** | ✅ **Present** (Multi-tensor active shapes, masks, and residuals) |
| **6. Extensions & Diagnostics** | ✅ **Present** (Timing flag tests) | ❌ **Not Present** | ❌ **Not Present** | ❌ **Not Present** | ❌ **Not Present** |