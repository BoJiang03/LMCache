# SPDX-License-Identifier: Apache-2.0
"""Which isolated scenarios (T0.9-T0.11, T3) apply to a model, and why.

This lives in its own module because two callers must agree on the answer
and there is no way to notice if they stop agreeing:

* ``test_isolated_paths`` parametrizes the scenarios it RUNS from here.
* ``certify`` describes the scheduling regimes it CLAIMS, and lists the
  exclusions, from here.

Those are the same statement seen from two sides -- a certificate is only
true if its claim matches what ran -- and they drifted apart once already:
``capacity_eviction`` and ``preemption`` were opened to sliding-window
hybrids without the certificate's ``scheduling`` list learning about it, so
Gemma 4's certificate under-claimed two scenarios it had passed while still
listing a chunked-prefill regime it never exercised. Deriving both from one
predicate is the fix.

Every exclusion below is a property of the model, never a list of the
models that happened to be tried: a newly registered model has to land on
the correct side by default.
"""

# First Party (test-local)
from specs import HybridFamily, ModelSpec

CHUNKED_PREFILL = "chunked_prefill"
CAPACITY_EVICTION = "capacity_eviction"
PREEMPTION = "preemption"
MP_CONNECTOR = "mp_connector"

# Every scenario name this module can emit. ``isolated_cases`` checks its own
# registry against this, so a rename cannot silently produce a model with
# fewer scenarios than the certificate claims.
ALL_SCENARIOS = (CHUNKED_PREFILL, CAPACITY_EVICTION, PREEMPTION, MP_CONNECTOR)


def isolated_scenarios(spec: ModelSpec) -> tuple[str, ...]:
    """The isolated scenarios that apply to ``spec``.

    ``mp_connector`` and ``capacity_eviction`` apply to every model: every
    scenario builds its harness through
    ``isolated_cases._scenario_harness``, which brings up a real MP cache
    server. Eviction additionally needs a cap that can hold one whole cache
    object, which the deepest hybrids supply via
    ``ModelSpec.eviction_capacity_gb``; a model that needs one and has not
    measured it fails the scenario loudly instead of being skipped, which is
    the safe direction for a certificate.

    ``chunked_prefill`` is excluded for reasons that are not plumbing: it
    pins the batched-token budget far below one prompt so that scheduler
    steps land inside an image span. A ``RECURRENT_STATE`` hybrid
    needs the opposite -- a step wide enough for one whole 544-784 token
    block, so its state snapshot lands on a block boundary -- which is
    contradictory by construction. A ``SLIDING_WINDOW`` hybrid's smaller
    blocks would in principle allow it, but that is untested, and the
    certificate says which of the two reasons applies.

    A model with ``mm_bidirectional_attention`` is excluded for a THIRD,
    independent reason, and it is why this exclusion cannot be keyed on the
    hybrid family: vLLM forces ``disable_chunked_mm_input`` for a
    mm-prefix-LM (``platforms/cuda.py``) and then refuses to start when the
    batched-token budget is below the model's worst-case mm item. So a
    budget small enough to split an image span aborts engine init, and one
    large enough to start cannot split the span -- contradictory the same
    way align mode is. Molmo 2 is the first NON-hybrid model in this class;
    before it, every such model happened to be a hybrid and the family gate
    hid the second reason (Gemma 3 and Gemma 4 are both mm-prefix-LMs too).

    ``preemption`` needs a MEASURED GPU block pool on a hybrid. The pool
    has to sit above what one max-length request costs (vLLM refuses to
    start below that) and below what the running batch costs, and that
    window follows from the model's KV bytes per token, so it cannot be
    derived from the spec -- an unmeasured hybrid would fail engine init
    rather than test anything. A ``RECURRENT_STATE`` hybrid is excluded
    whatever it declares, because for that family the window is empty
    rather than unknown: align mode's one-block step budget keeps two
    requests from ever running at once, so a pool big enough to survive has
    nothing to preempt, and every pool small enough to create pressure
    aborts the engine in vLLM's block-pool bookkeeping once the connector
    is attached. ``certify`` carries the measurements; the gate is on the
    family because align mode is the cause.

    Args:
        spec: The model under certification.

    Returns:
        Scenario names, in the order the parametrization should run them.
    """
    scenarios: list[str] = []
    if spec.hybrid_family is HybridFamily.NONE and not spec.mm_bidirectional_attention:
        scenarios.append(CHUNKED_PREFILL)
    scenarios.append(CAPACITY_EVICTION)
    pool_sizable = spec.hybrid_family is not HybridFamily.RECURRENT_STATE
    if pool_sizable and (not spec.hybrid_block_tokens or spec.preemption_gpu_blocks):
        scenarios.append(PREEMPTION)
    scenarios.append(MP_CONNECTOR)
    return tuple(scenarios)
