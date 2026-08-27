# SPDX-License-Identifier: Apache-2.0
"""Model registry for the multimodal acceptance suite.

Adding support certification for a new model means adding one ``ModelSpec``
entry here. The test logic is model-agnostic; only this declarative spec (and
optional ``extra_suites`` flags for special architectures) is per-model.
"""

# Standard
from dataclasses import dataclass, field
import enum
import os


class HybridFamily(enum.Enum):
    """Why vLLM splits a model's KV cache into more than one group.

    Both families need the MP deployment path, because the in-process
    connector does not advertise support for vLLM's hybrid KV cache
    manager. They need DIFFERENT engine settings, though, so the suite has
    to tell them apart rather than treating "hybrid" as one thing.

    Attributes:
        NONE: One KV cache group; the in-process path works.
        RECURRENT_STATE: Mamba/Gated-DeltaNet linear-attention layers whose
            state is a per-sequence page rather than per-token KV. Requires
            ``mamba_cache_mode="align"``, which in turn requires vLLM
            prefix caching, and a scheduler step wide enough for one whole
            block so the state snapshot lands on a block boundary.
        SLIDING_WINDOW: Sliding-window layers mixed with full-attention
            layers, all of them ordinary paged KV. Needs none of the align
            settings -- the groups differ in window and block size, not in
            kind -- so the engine keeps the suite's default scheduling and
            prefix caching stays off.
    """

    NONE = "none"
    RECURRENT_STATE = "recurrent_state"
    SLIDING_WINDOW = "sliding_window"


@dataclass(frozen=True)
class ModelSpec:
    """Declarative description of one model under certification.

    Attributes:
        key: Short identifier used in ``LMCACHE_MM_E2E_MODELS``.
        hf_id: HuggingFace model id (smallest usable variant, to bound cost).
        modalities: Modalities to certify ("image", "video", "audio").
        max_model_len: Engine context length for the test run.
        gpu_memory_utilization: Fraction of GPU memory for the engine.
        extra_suites: Special-architecture add-on suites this model needs
            (e.g. "deepstack", "bidirectional_image_attn", "modality_lora").
            A declared suite the repo cannot currently run is NOT dropped
            from the spec: it is a property of the model, and ``certify``
            turns it into an exclusion so the gap is stated rather than
            forgotten (see ``certify.DEEPSTACK_NOT_COVERED``).
        chat_template_kwargs: Extra kwargs passed to every ``llm.chat`` call
            (test engine, baseline runner, and MME parity runs alike), e.g.
            ``{"enable_thinking": False}`` for hybrid-thinking models whose
            template supports disabling the reasoning preamble. Must be
            JSON-serializable. The suite's oracles (baseline exact match,
            semantic probes, MME yes/no parsing) assume the answer lands
            within each request's small ``max_tokens`` budget, so thinking
            MUST be disabled here for models that would otherwise emit a
            reasoning preamble.
        mme_mm_processor_kwargs: ``mm_processor_kwargs`` for the MME parity
            engines only (real photos of arbitrary size, unlike the suite's
            small synthetic images). Must cap the per-image token count so a
            question fits the 8192-token parity context; the kwarg names are
            model-processor-specific (Qwen: ``max_pixels``; GLM:
            ``size.longest_edge`` in total pixels). Must be
            JSON-serializable.
        min_decode_tokens: Floor applied to every request's ``max_tokens``
            (suite, baselines, and MME parity alike). Models that lead with a
            short preamble before the answer even with thinking disabled
            (e.g. GLM's ``<|begin_of_box|>``-boxed answers) need enough
            budget for the answer to land inside the generated text, or the
            semantic probes and MME yes/no parsing read only preamble. 0
            keeps each request's own budget.
        mme_max_tokens: Decode budget for the MME parity runs specifically.
            Real MME photos draw a much longer preamble than the suite's
            synthetic images (reasoning over OCR'd code, artwork, posters),
            so a budget that suffices for the suite can still truncate MME
            answers and fail the parse-ratio gate. 0 falls back to
            ``min_decode_tokens`` (and the parity runner's own default when
            that is also 0).
        mme_max_flip_fraction: Per-item answer-flip budget for the MME
            parity gate, as a fraction of the question count. 0 keeps the
            gate's default (0.5%, calibrated on short-answer models). Long
            chain-of-thought models need a higher floor: with vs. without
            the LMCache connector are two different (each fully
            deterministic) numeric regimes, and a 200+-token reasoning
            chain amplifies the regime difference into ~1% of borderline
            answers landing on the other side or a repetition loop
            (parse ''), with no corruption signature. Widening this is safe
            because it loosens only the COUNT: the gate separately requires
            the flips to be two-sided, and every source of regime noise
            flips answers both ways while KV corruption only degrades. Real
            corruption is still caught by that direction test, by the
            byte-identical replay oracles, and by the hit-ratio gate.
        mme_min_parse_ratio: Floor on the fraction of BASELINE answers that
            must parse to yes/no. 0 keeps the gate's 0.9 default. The gate
            exists to catch a model whose verdict never lands inside the
            decode budget -- then all three passes parse to '' and the
            flip/score comparisons pass while measuring nothing -- so it
            may only be lowered for a model that ABSTAINS rather than
            truncates, and only after measuring that a larger budget does
            not help. Gemma 4-E4B declines 239 of 2374 questions
            (artwork 117, celebrity 82, landmark 14) and resolved 0 of them
            at 8, 64 or 256 tokens, so its ceiling is 0.8993 whatever the
            budget. Lowering the floor stays safe because a refusal is
            stable text: if the hit path corrupted one of those answers,
            the parsed verdict would change from '' and the flip counter
            would see it.
        mme_max_local_cpu_gb: LMCache local-CPU capacity (GB) for the MME
            parity run. 0 keeps the runner's 40 GB default, which holds the
            full benchmark's KV for the certified GQA-2 models (28-36
            KB/token). A wider-KV model overflows it -- the pass-2 replay
            revisits requests in store order, the LRU scan evicts every
            entry before its revisit, and the hit-ratio gate fails at ~0
            with zero flips (pure recompute, not corruption). Size it to
            hold the whole benchmark: questions x prompt tokens x KV bytes
            per token.
        hybrid_block_tokens: LMCache chunk size ``N`` in tokens for a model
            whose KV cache vLLM splits into more than one group (0 = single
            group). Such models run the suite on the MP deployment path —
            the in-process connector does not support vLLM's hybrid KV
            cache manager — with an MP cache server at ``chunk_size = N``
            and ``--separate-object-groups``. Which further engine settings
            are mandatory depends on ``hybrid_family``. Hit granularity
            becomes ``N``, so the conftest pads every request's prompt to
            span multiple blocks and the tests read their chunk tolerance
            from ``harness.chunk``.

            ``N`` must be a common multiple of every PAGED group's block
            size, which is what LMCache itself requires (it rejects
            registration otherwise: "chunk size must be a multiple of
            engine group tokens_per_block"); recurrent-state groups hold
            one page per sequence and impose no such constraint. The
            harness validates this against the live engine. For the
            Mamba/GDN hybrids the only paged group is full attention, so
            ``N`` equals the block size vLLM prints at startup ("Setting
            attention block size to N tokens..."). A sliding-window hybrid
            can have SEVERAL paged groups at different block sizes --
            Gemma 4 pairs 512-wide full-attention layers (block 16) with
            256-wide sliding layers (block 32), because vLLM equalizes
            page size by varying the block size -- and then ``N`` is their
            common multiple, NOT the ``cache_config.block_size`` vLLM
            reports.
        hf_overrides: ``hf_overrides`` for every engine the suite starts
            for this model (test engine, baseline runner, MME parity), to
            repair a model config vLLM cannot read as shipped. Must be
            JSON-serializable, and must be identical across those engines:
            it changes the model's geometry, so a baseline built without it
            would not be comparable. Gemma 4 needs it because transformers
            5.15 folds the per-layer attention dims into
            ``per_layer_config`` and stops exposing the flat
            ``global_head_dim`` / ``num_global_key_value_heads`` names that
            vLLM still reads with ``getattr(config, name, <sliding
            value>)`` -- so without this the full-attention layers are
            built at the sliding-window geometry and weight loading dies
            on a shape mismatch.
        parity_benchmark: Which benchmark the parity check (T0.6) scores
            for this model -- a ``benchmark_parity.BENCHMARKS`` key; empty
            means "mme". Set it to "mmau" for a model certified on audio,
            whose quality cannot be measured by an image benchmark. Note
            that the ``mme_*`` knobs below apply to whichever benchmark is
            selected here despite their names; renaming them is a separate
            mechanical change.
        mm_encoder_attn_backend: vLLM multimodal-encoder attention backend
            for every engine the suite starts for this model; empty leaves
            vLLM's own choice. Applies to the same set of engines as
            ``hf_overrides`` and for the same reason -- it changes how the
            encoder computes, so a baseline without it is not comparable.
            Qwen3-Omni needs "TORCH_SDPA" on vLLM 0.23.0, whose vision
            tower hands its ``cu_seqlens`` to the attention kernel without
            moving it to the device; profiling then aborts with
            ``cu_seqlens_q must be on CUDA`` even for an audio-only run,
            and modality limits cannot route around it (setting a modality
            to 0 skips loading its weights and fails later on a meta
            tensor). Fixed upstream in 0.27.1 at
            ``qwen3_omni_moe_thinker.py:982``.
        trust_remote_code: Pass vLLM's ``trust_remote_code`` to every engine
            the suite starts for this model. Only for a repo whose CONFIG
            cannot be read without it -- Molmo 2 ships ``auto_map`` and
            transformers 5.15 refuses the config outright ("contains custom
            code which must be executed"), even though vLLM implements the
            model natively (``molmo2.py``) and never runs the repo's own
            modeling code. Set it on the smallest set of models that need
            it, and never to work around a processor bug: it executes code
            from the model repo in every suite subprocess. Applies to the
            same set of engines as ``hf_overrides`` -- if the engine under
            test can read the config and the baseline cannot, there is no
            baseline to compare against.
        mm_bidirectional_attention: Whether the model attends BIDIRECTIONALLY
            over its multimodal placeholder span -- vLLM's
            ``ModelConfig.is_mm_prefix_lm``. It is a routing fact, not a
            performance note: ``platforms/cuda.py`` forces
            ``disable_chunked_mm_input`` for such a model, and vLLM then
            refuses to start at all when ``max_num_batched_tokens`` is
            below the model's worst-case mm item. That makes the
            chunked-prefill scenario contradictory for this class -- a
            budget small enough to split an image span aborts engine init,
            and one large enough to start cannot split it -- so
            ``isolated_routing`` excludes the scenario and the certificate
            says so. Measured 2026-08-22 via ``create_model_config()`` for
            all 12 registered models: True for Molmo 2-4B and Gemma 3-4B
            only. Gemma 4-E4B measures False, which is worth stating rather
            than silently encoding -- it was registered partly to exercise
            bidirectional image attention, and on vLLM 0.23.0 it does not
            carry the flag its predecessor does.
        media_first_template: Whether this model's chat template renders
            media items BEFORE the conversation text. The suite isolates its
            cases with a per-case salt at the head of the prompt, which only
            works if text comes first; Molmo 2 emits ``<|image|>`` ahead of
            ``<|im_start|>``, so the image span becomes the prompt's first
            ~750 tokens and is byte-identical across cases. Measured
            2026-08-22: two T0.4 phases using the same image share a
            762-token prefix, and the later phase reused the earlier one's
            entry -- correct caching, but it made the cross-image assertion
            compare against the wrong entry. Setting this True moves the
            case identity into the synthetic media itself
            (``catalog.case_media_bits``). The harness re-derives this from
            the live tokenizer and raises if the spec disagrees, so a new
            model cannot land on the wrong side by silence.
        media_prefix_stable: Whether adding a media item to a prompt leaves
            the tokens of the items already there unchanged -- i.e. whether
            the one-image prompt is a token PREFIX of the same-image-plus-one
            prompt. Every model registered before Molmo 2 is; Molmo 2 is not,
            measured 2026-08-22 on the live engine: ``t22-A`` (787 tokens)
            and ``t22-AC`` (1553) share exactly ONE token, because its
            processor emits a layout derived from the whole image SET before
            any single image's tiles. T2.2 (partial sharing) is deselected
            when this is False -- its premise is the prefix, so running it
            would report a cache failure for a prefix that does not exist.
            NOT auto-validated, unlike ``media_first_template``: the check
            needs EXPANDED prompts, which means putting requests through the
            engine, which would seed the cache the tests then measure.
        supports_system_role: Whether this model's chat template accepts a
            ``system`` message. Every suite request carries one -- it holds
            the per-case salt, which is what keeps otherwise-identical
            requests from sharing a cache prefix -- so a template that
            rejects it fails every request rather than degrading. Molmo 2's
            template raises ``jinja2 TemplateError: Conversation roles must
            alternate user/assistant/...`` on a system message; setting this
            False folds the same text, salt included, into the front of the
            first user message, which keeps the salt ahead of the media
            items and so keeps case isolation intact.
        hybrid_family: Which kind of multi-group KV cache this model has
            (see ``HybridFamily``); it selects the mandatory engine
            settings. Must be set exactly when ``hybrid_block_tokens`` is.
        hybrid_object_groups: Number of LMCache cache objects stored per
            token block under ``--separate-object-groups`` (full-attention
            KV + recurrent-state groups; 2 for the Qwen3.5 family). Used
            by the storage-conservation bounds. 0 for non-hybrids.
        isolated_gpu_utilization: GPU fraction for the engines
            ``isolated_cases.py`` starts, overriding its 0.35 default
            (0 = use the default). That default assumes an isolated engine
            may share the GPU with the acceptance session's engine, which
            leaves too little for a model whose weights alone exceed it --
            27B in bf16 needs 0.37 before a single KV block. Raising it is
            safe because the scenarios each run in their own subprocess and
            the isolated module is collected before the acceptance module,
            so no session engine is alive yet.
        mp_server_l1_gb: L1 capacity (GB) for the MP cache servers the
            suite starts, overriding their hybrid defaults (0 = use them).
            A model's per-block cost is ``blocks x layers x page_size``,
            which varies by two orders of magnitude across the registered
            hybrids; a capacity that cannot hold one test's working set
            makes the cache evict mid-test and the store-conservation
            audits fail for a reason that has nothing to do with LMCache.
        preemption_gpu_blocks: vLLM GPU block pool for the preemption
            scenario, overriding its 128 default (0 = use it). The pool has
            to straddle a window: big enough to hold one max-length request
            (vLLM refuses outright otherwise) and too small to hold the
            whole batch (or nothing is preempted and the scenario reports
            itself vacuous). 128 blocks is that window for a uniform
            16-token-block model, but a hybrid pays per group -- Gemma 4-E4B
            needs 0.11 GiB for one request while 128 of its blocks give
            0.03 GiB -- so the deeper hybrids have to raise it.
        eviction_capacity_gb: Cache capacity (GB) for the capacity-eviction
            scenario, overriding the default for this model's deployment
            path (0 = use it). The scenario needs a cap small enough that
            its traffic overflows it several times over and large enough to
            hold at least one whole cache object -- a cap below one object
            cannot store anything, and the run then fails for a reason that
            has nothing to do with eviction. A block's objects cost
            ``layers x page_size`` and that spans two orders of magnitude
            across the registered hybrids (12 MB on Qwen3.5-2B against
            ~205 MB on the 27Bs), so the deepest models have to raise it.
            Must be a whole multiple of the MP host allocator's 64 MB
            expansion unit, which silently rounds a sub-unit request up.
        answer_extract_pattern: Regex whose LAST match's group(1) is the
            model's final answer inside a generated text ('' = the whole
            text is the answer). For models that phrase a preamble before a
            marked answer (GLM: ``<|begin_of_box|>...<|end_of_box|>``), the
            hit-path replay oracle compares extracted answers when the full
            texts differ: miss pass (KV computed) and hit pass (KV loaded)
            are different numeric regimes, and a verbose preamble gives the
            regime noise many tokens to flip, while a real KV corruption
            flips the marked answer itself.
    """

    key: str
    hf_id: str
    modalities: frozenset = frozenset({"image"})
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.6
    extra_suites: frozenset = field(default_factory=frozenset)
    chat_template_kwargs: dict[str, object] = field(default_factory=dict)
    mme_mm_processor_kwargs: dict[str, object] = field(default_factory=dict)
    min_decode_tokens: int = 0
    mme_max_tokens: int = 0
    mme_max_flip_fraction: float = 0.0
    mme_min_parse_ratio: float = 0.0
    mme_max_local_cpu_gb: float = 0.0
    hybrid_block_tokens: int = 0
    hybrid_family: HybridFamily = HybridFamily.NONE
    hybrid_object_groups: int = 0
    hf_overrides: dict[str, object] = field(default_factory=dict)
    parity_benchmark: str = ""
    mm_encoder_attn_backend: str = ""
    trust_remote_code: bool = False
    mm_bidirectional_attention: bool = False
    media_first_template: bool = False
    media_prefix_stable: bool = True
    supports_system_role: bool = True
    isolated_gpu_utilization: float = 0.0
    mp_server_l1_gb: float = 0.0
    preemption_gpu_blocks: int = 0
    eviction_capacity_gb: float = 0.0
    answer_extract_pattern: str = ""

    def __post_init__(self) -> None:
        """Reject a spec whose hybrid fields disagree.

        The chunk size and the family are two halves of one statement --
        "this model runs on the MP path, with these mandatory engine
        settings". Half a statement would either put a model on the MP
        path with the wrong settings or leave a multi-group model on the
        in-process path, where engine init fails with a message about
        converting KV cache specs that names no model.

        Raises:
            ValueError: If exactly one of ``hybrid_block_tokens`` and
                ``hybrid_family`` is set.
        """
        is_hybrid = self.hybrid_family is not HybridFamily.NONE
        if bool(self.hybrid_block_tokens) != is_hybrid:
            raise ValueError(
                f"{self.key}: hybrid_block_tokens="
                f"{self.hybrid_block_tokens} and hybrid_family="
                f"{self.hybrid_family.value} must be set together "
                f"(a chunk size without a family, or a family without a "
                f"chunk size, is half a spec)"
            )


# MME photos are arbitrarily large; cap them at ~768 image tokens per photo
# (Qwen smart-resize: tokens <= max_pixels / 28^2; GLM: 4 pixels per 14x14
# patch merge, same 602112-pixel budget) so a question fits the parity
# engine's 8192-token context.
_MME_PIXEL_BUDGET = 768 * 28 * 28

MODEL_SPECS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in [
        ModelSpec(
            key="qwen2.5-vl-3b",
            hf_id="Qwen/Qwen2.5-VL-3B-Instruct",
            modalities=frozenset({"image", "video"}),
            mme_mm_processor_kwargs={"max_pixels": _MME_PIXEL_BUDGET},
        ),
        ModelSpec(
            key="qwen2-vl-2b",
            hf_id="Qwen/Qwen2-VL-2B-Instruct",
            modalities=frozenset({"image", "video"}),
            mme_mm_processor_kwargs={"max_pixels": _MME_PIXEL_BUDGET},
            # Measured on vLLM 0.27.1 (2026-08-26): three full MME parity
            # runs flip 19/19/18 answers pass2-vs-pass1 (0.80%), a
            # deterministic 18-question core (pairwise jaccard >= 0.90),
            # with pass1 byte-identical to the no-LMCache baseline and the
            # retrieved KV bit-identical to the computed KV. The flips are
            # NOT a cache defect: a plain-vLLM control (prefix caching on,
            # no LMCache anywhere) reproduces the same 18 flips, same
            # directions, when pass 2 is submitted one request at a time --
            # first-token logits shift by one bf16 quantum (+-0.125)
            # between batched and small-step execution, and ~1% of MME
            # questions sit within one quantum of the yes/no boundary. The
            # hit path enters the small-step regime through retrieve-
            # completion staggering admission. Same phenomenon class as
            # glm-4.6v-flash's documented numeric-regime divergence; see
            # records/2026/08/26/10_.
            mme_max_flip_fraction=0.01,
        ),
        ModelSpec(
            key="internvl3.5-2b",
            # The transformers-native export (InternVLForConditionalGeneration).
            # The OpenGVLab-format repo (InternVLChatModel) ships custom config
            # code and would need trust_remote_code, which the suite's engines
            # do not pass.
            hf_id="OpenGVLab/InternVL3_5-2B-HF",
            modalities=frozenset({"image", "video"}),
            # InternVL tiles photos into 448x448 crops worth 256 tokens each,
            # plus a 256-token global thumbnail whenever a photo is tiled
            # (HF processor: crop_to_patches); max_patches=2 caps a photo at
            # 768 image tokens, the same budget the Qwen/GLM pixel caps encode.
            mme_mm_processor_kwargs={"max_patches": 2},
            # The Qwen3-1.7B backbone is GQA-8: 28 layers x 8 KV heads x
            # 128 dims = 112 KB/token, 3-4x the certified GQA-2 models.
            # Full MME (~2374 x <=1000-token prompts) is ~250 GB of KV;
            # at the 40 GB default the pass-2 LRU scan hit ~0 (measured
            # 2026-08-20: hit_ratio 0.013 with 0 flips and 0.00 score
            # delta -- pure recompute). 280 GB holds the whole run.
            mme_max_local_cpu_gb=280.0,
        ),
        ModelSpec(
            key="qwen3-vl-2b",
            hf_id="Qwen/Qwen3-VL-2B-Instruct",
            modalities=frozenset({"image", "video"}),
            # DeepStack: the vision tower emits multiscale features from ViT
            # layers 5/11/17 that vLLM injects into LLM layers 0-2 via a
            # per-step side buffer OUTSIDE the paged KV. The add-on suite
            # that verified the LMCache resume path against KV recomputed
            # with that injection needed to read stored KV back, which only
            # the removed in-process path could do; the declaration stays
            # so every certificate carries the gap as an exclusion.
            extra_suites=frozenset({"deepstack"}),
            # New-style processor size cap (Qwen2VLImageProcessorFast):
            # 16x16 patches with 2x2 merge = 1024 pixels per token, so
            # 786432 total pixels cap a photo at the same ~768-token budget
            # the other specs encode.
            mme_mm_processor_kwargs={
                "size": {"shortest_edge": 65536, "longest_edge": 786432}
            },
            # Qwen3-1.7B-class backbone, GQA-8: 28 layers x 8 KV heads x
            # 128 dims = 112 KB/token, same as InternVL3.5-2B; the full MME
            # run needs the same 280 GB local-CPU capacity (see that spec).
            mme_max_local_cpu_gb=280.0,
        ),
        ModelSpec(
            key="qwen3.5-2b",
            hf_id="Qwen/Qwen3.5-2B",
            modalities=frozenset({"image", "video"}),
            # GDN hybrid: 24 layers = 6 full attention + 18 Gated-DeltaNet
            # linear attention. The linear layers' recurrent state is cached
            # as opaque pages on the MP path (align mode); upstream validates
            # only TEXT KV caching for these models (mp/hybrid_models.rst),
            # and this suite's run is the image/video validation.
            hybrid_block_tokens=544,
            hybrid_family=HybridFamily.RECURRENT_STATE,
            hybrid_object_groups=2,
            # Same new-style pixel cap as Qwen3-VL (1024 px/token).
            mme_mm_processor_kwargs={
                "size": {"shortest_edge": 65536, "longest_edge": 786432}
            },
            # Only 6 full-attention layers (12 KB/token), but every 544-token
            # block also stores a fat GDN state page (~13 MB/object measured);
            # ~26 MB per MME question needs well over the 40 GB default.
            mme_max_local_cpu_gb=280.0,
        ),
        ModelSpec(
            key="qwen3.6-27b",
            hf_id="Qwen/Qwen3.6-27B",
            modalities=frozenset({"image", "video"}),
            # 27B needs ~52 GB of weights before any KV pool.
            gpu_memory_utilization=0.8,
            # Same GDN hybrid architecture class as Qwen3.5-2B
            # (Qwen3_5ForConditionalGeneration), four times as deep: 64
            # layers = 16 full attention (4 KV heads x 256) + 48
            # Gated-DeltaNet. No DeepStack (deepstack_visual_indexes is
            # empty), so no add-on suite.
            #
            # Measured at engine init: unified block 784 tokens, four KV
            # cache groups (3x MambaSpec of 16 layers + 1x
            # FullAttentionSpec of 16) at an identical 3.2 MB page size,
            # which bucket by sliding-window size into 2 object groups.
            # That page size is PER LAYER (3211264 / 784 = 4096 B/token =
            # K+V for 4 heads x 256 dims), so a block costs all 64 layers:
            # ~205 MB, i.e. 262 KB/token -- 5x Qwen3.5-2B, not a third of
            # it. Every capacity number below follows from that.
            hybrid_block_tokens=784,
            hybrid_family=HybridFamily.RECURRENT_STATE,
            hybrid_object_groups=2,
            # Thinks out loud by default -- an 8-token budget returns
            # "The user wants to identify the dominant color" instead of a
            # color. Its chat template takes enable_thinking, and with it
            # off the synthetic probes answer "Red." / "Blue" directly.
            chat_template_kwargs={"enable_thinking": False},
            # Same new-style pixel cap as the rest of the Qwen3-VL family
            # (16x16 patches, 2x2 merge = 1024 px/token). The model's own
            # preprocessor default is 16.7M pixels, which would blow the
            # parity engine's context on a single photo.
            mme_mm_processor_kwargs={
                "size": {"shortest_edge": 65536, "longest_edge": 786432}
            },
            # Measured (2026-08-21), all on the gate's PARSED-answer metric
            # (raw text also churns case/whitespace: 85 raw differences for
            # 12 parsed ones, which the gate rightly ignores):
            #   baseline rerun, identical config ... 0 flips (byte-identical)
            #   no LMCache, max_num_seqs changed .... 2 (0.084%)
            #   LMCache miss pass vs plain vLLM ..... 1
            #   LMCache hit pass vs miss pass ...... 12 (0.505%)
            # The hit path diverges 6x more than a batch-shape change, and
            # that is inherent to hybrids rather than a defect: a hit
            # RESTORES a Gated-DeltaNet state page instead of recomputing
            # it, so unlike full attention (where loaded KV is bit-identical
            # to computed KV) there is no identical arithmetic to hope for.
            # It scales with linear depth -- 18 GDN layers on Qwen3.5-2B
            # flip 0.21%, 48 here flip 0.505%. Per-question inspection
            # agrees: the 12 land on borderline recognition items (6 of 12
            # in `landmark`), the hit pass is right 5 times against the miss
            # pass's 7 (a corrupting cache would be one-sided), and the
            # score moves +1.0 of 2179. 1% covers it; the 10-point score
            # gate still catches a real regression.
            mme_max_flip_fraction=0.01,
            # The full MME run stored 196784 tokens = 251 blocks = 51 GB,
            # measured, with no eviction against 120 GB (the controller
            # evicts at 80% = 96 GB).
            mme_max_local_cpu_gb=120.0,
            # The 64-image pressure case alone stores 64 x 4 blocks = 52 GB
            # and the rest of the session adds to it, so the suite's 60 GB
            # hybrid default evicts mid-test: measured 92 resident keys
            # vanishing at image 62 of 64, which is exactly what T0.7's
            # resident-key audit is there to catch.
            mp_server_l1_gb=200.0,
            # 52 GB of weights is 0.37 of an H200 before any KV block, so
            # the isolated modules' 0.35 default cannot even load the model
            # ("No available memory for the cache blocks").
            isolated_gpu_utilization=0.75,
            # The eviction scenario's one-unit (64 MB) MP default cannot hold
            # a single object here: a 784-token block costs ~205 MB across
            # all 64 layers, arriving as two objects of which the
            # recurrent-state one is ~154 MB. 8 units (512 MB) is the first
            # size that holds a whole block with room for the 0.80 eviction
            # watermark to act on, and the scenario stays far from vacuous
            # because the traffic is enormous by comparison. Measured
            # 2026-08-22: 6 resident objects, 513802240 bytes = 0.957 of the
            # cap, against 11.65 GB of intended traffic -- 21.7x overflow.
            eviction_capacity_gb=0.5,
        ),
        ModelSpec(
            key="qwen3.8-27b",
            hf_id="Qwen/Qwen3.8-27B",
            modalities=frozenset({"image", "video"}),
            # Every field below is copied from qwen3.6-27b deliberately: the
            # two config.json files are IDENTICAL apart from
            # ``transformers_version`` (4.57.1 vs 5.8.0.dev0) -- same
            # ``Qwen3_5ForConditionalGeneration``, same 64 layers with the
            # same ``layer_types`` (16 full attention + 48 Gated-DeltaNet),
            # same 5120 hidden / 256 head_dim / 4 KV heads, same empty
            # ``deepstack_visual_indexes``, same 27-layer vision tower. So
            # 3.8-27B is a retrained 3.6-27B, not a new architecture, and
            # vLLM 0.23.0 runs it on the code path this suite already
            # certified -- the "needs a newer vLLM" note in
            # records/2026/08/21/2_ was research, not measurement, and is
            # wrong for this model.
            #
            # Consequences of that identity, each inherited rather than
            # re-derived: block geometry (784 tokens, 4 KV cache groups ->
            # 2 object groups), per-block cost (~205 MB = 262 KB/token
            # across 64 layers), and therefore every capacity number.
            gpu_memory_utilization=0.8,
            hybrid_block_tokens=784,
            hybrid_family=HybridFamily.RECURRENT_STATE,
            hybrid_object_groups=2,
            chat_template_kwargs={"enable_thinking": False},
            mme_mm_processor_kwargs={
                "size": {"shortest_edge": 65536, "longest_edge": 786432}
            },
            # A PREDICTION, not yet a measurement. records/2026/08/21/10_
            # argues hybrid flip rates track linear-attention depth (18 GDN
            # layers -> 0.21%, 48 -> 0.505%) because a hit restores a lossy
            # recurrent-state page instead of reproducing KV bit-for-bit.
            # This model has the same 48 GDN layers, so the same ~0.5% and
            # the same 1% budget should hold; a rate far off that number
            # falsifies the depth argument and belongs in a record rather
            # than in a widened budget here.
            mme_max_flip_fraction=0.01,
            mme_max_local_cpu_gb=120.0,
            mp_server_l1_gb=200.0,
            isolated_gpu_utilization=0.75,
            eviction_capacity_gb=0.5,
        ),
        ModelSpec(
            key="gemma-4-e4b",
            hf_id="google/gemma-4-E4B-it",
            modalities=frozenset({"image", "video"}),
            # NOT marked mm_bidirectional_attention, and that is a
            # measurement rather than an omission: `create_model_config()`
            # reports is_mm_prefix_lm=False for this model on vLLM 0.23.0
            # while reporting True for Gemma 3-4B. This model was picked
            # partly to exercise bidirectional image attention (vLLM
            # #40106), so the flag being absent here is itself the finding:
            # either the checkpoint does not use it, or vLLM is running the
            # image span causally for it. Not yet resolved.
            # A SLIDING-WINDOW hybrid, not a Mamba/GDN one: 42 layers of
            # 5 sliding (window 512, head_dim 256) to 1 full attention
            # (head_dim 512), which vLLM splits into 6 KV cache groups --
            # so it needs the hybrid cache manager and therefore the MP
            # deployment path, same as the GDN models but for a different
            # reason.
            #
            # Measured at engine init: 5 SlidingWindowSpec groups of 4
            # layers at block 32 plus 1 FullAttentionSpec group of 4 at
            # block 16, every group at page size 65536 (vLLM equalizes page
            # size by halving the block size for the twice-as-wide full
            # layers). Only 24 of the 42 layers appear, because
            # num_kv_shared_layers=18 makes the rest reuse another layer's
            # KV -- so the cost is 56 KB/token, not what layers x heads x
            # dims would suggest.
            #
            # The chunk is therefore 32, the common multiple of the two
            # paged block sizes, NOT the 16 that vLLM reports as
            # cache_config.block_size: LMCache rejects a chunk that is not
            # a multiple of every paged group ("chunk size 16 must be a
            # multiple of engine group 0 tokens_per_block 32"). Verified at
            # 32 on the MP path: pass 1 stored, pass 2 loaded 2304 tokens
            # back with identical text and local_cached 0.
            hybrid_block_tokens=32,
            hybrid_family=HybridFamily.SLIDING_WINDOW,
            # Two object-group buckets: the sliding groups (window 512) and
            # the full-attention groups (no window).
            hybrid_object_groups=2,
            # transformers 5.15 moved Gemma 4's per-layer attention dims
            # into per_layer_config and no longer exposes the flat
            # `global_head_dim` name that vLLM reads with
            # `getattr(config, "global_head_dim", config.head_dim)`, so
            # without this the 7 full-attention layers are built at the
            # sliding geometry (256) and their 512-wide weights fail to
            # load. The value is per_layer_config[<first full layer>]
            # .head_dim; 12B additionally needs
            # num_global_key_value_heads (it is attention_k_eq_v, with no
            # v_proj at all on full layers).
            hf_overrides={
                "allow_global_per_layer_attribute_access": True,
                "text_config": {
                    "allow_global_per_layer_attribute_access": True,
                    "global_head_dim": 512,
                },
            },
            # Images are a fixed 280 soft tokens
            # (vision_soft_tokens_per_image), so no pixel budget is needed
            # for the MME photos -- unlike every Qwen/GLM spec above.
            #
            # Declines to answer 239 of 2374 MME questions -- "I cannot
            # determine the name of the person" on celebrity/artwork/
            # landmark items -- and measured resolves 0 of them at 8, 64 or
            # 256 decode tokens, so 0.8993 is its ceiling and no budget
            # buys the gate's 0.9 default. See mme_min_parse_ratio.
            mme_min_parse_ratio=0.85,
            # 56 KB/token over ~2374 questions of <=1000 prompt tokens is
            # ~130 GB, well past the runner's 40 GB default.
            mme_max_local_cpu_gb=280.0,
            # NO preemption_gpu_blocks, and this is a retraction rather than
            # an omission. b1836ce1 set 512 here and recorded it as verified;
            # that verification ran the scenario WITHOUT the conftest's
            # hybrid prompt padding, which the suite always applies, so the
            # number describes an engine the suite never builds. Under the
            # padded prompts it is vacuous, and so is every other pool tried
            # (measured 2026-08-22, each the whole scenario):
            #
            #     blocks | pool tokens | preemptions
            #        512 |       2,314 | 0   <- the shipped value
            #        768 |       3,472 | 0
            #        992 |       4,484 | 0
            #       1024 |       4,629 | 0
            #       1152 |       5,208 | 0
            #   512, unpadded prompts  | 1   <- what b1836ce1 measured
            #
            # A 2.25x sweep with nothing in it, so this is not a number
            # waiting to be found. The mechanism: the scenario needs a pool
            # that admits all six prompts and still cannot hold their decode
            # growth, and this model's per-request footprint SATURATES --
            # its sliding window is 512 tokens while the padded prompt is
            # ~504 lookup tokens over a much longer span, so the sliding
            # groups free blocks behind the window and 112 more decode
            # tokens cost nothing there. Unpadded, prompts were 299 tokens,
            # inside the window, nothing was freed, and the batch could
            # outgrow the pool -- which is exactly why the old measurement
            # looked fine. Gemma 3-4B (window 1024, wider than its prompt)
            # does not saturate and still preempts at 1024 blocks.
            #
            # Widening the scenario's decode budget would restore the
            # pressure, but PREEMPTION_MAX_TOKENS is shared with every
            # certified model, so that is a separate change with its own
            # re-verification; recorded rather than done here.
        ),
        ModelSpec(
            key="gemma-3-4b",
            hf_id="google/gemma-3-4b-it",
            # Image only, and not a scoping choice: vLLM's Gemma 3
            # processor declares get_supported_mm_limits() == {"image":
            # None}, with no video entry, so the suite's video probes
            # deselect because the model cannot take video at all.
            modalities=frozenset({"image"}),
            # Measured 2026-08-22: is_mm_prefix_lm=True, i.e. vLLM does
            # attend bidirectionally over this model's image span. It
            # changes nothing about which scenarios run -- being a hybrid
            # already excludes chunked prefill -- but the certificate now
            # names both reasons instead of only the hybrid one. Gemma 4
            # measures False on the same vLLM; see that spec.
            mm_bidirectional_attention=True,
            # The second SLIDING_WINDOW hybrid, and the controlled
            # comparison Gemma 4 could not provide. Measured at engine init
            # (2026-08-22): 34 text layers as 29 sliding (window 1024) plus
            # 5 full attention, which vLLM splits into 7 KV cache groups --
            # 6 SlidingWindowSpec plus 1 FullAttentionSpec -- so it needs
            # the hybrid cache manager and therefore the MP deployment path.
            #
            # Unlike Gemma 4, every group is at block_size 16 and page size
            # 65536, so the chunk is 16 -- the SAME chunk the five
            # full-attention SUPPORTED certificates were taken at. Gemma 4
            # confounded chunk size (32) with multi-group sliding windows;
            # this model separates them, which is the point of certifying
            # it: chunk 16 with one group is green five times over, so a
            # failure here is attributable to the grouping alone.
            hybrid_block_tokens=16,
            hybrid_family=HybridFamily.SLIDING_WINDOW,
            # Two object-group buckets: the sliding groups (window 1024)
            # and the full-attention group (no window).
            hybrid_object_groups=2,
            # No hf_overrides: Gemma 3 keeps its attention dims as flat
            # config attributes, so vLLM reads them without the
            # per_layer_config indirection that Gemma 4 needs.
            #
            # 136 KB/token, measured (65536 bytes x 34 layers / block 16) --
            # 2.4x Gemma 4, which shares KV across 18 of its 42 layers
            # while Gemma 3 shares none.
            mme_max_local_cpu_gb=280.0,
            # No threshold overrides: the defaults pass with room to spare.
            # Measured 2026-08-22 on the full 2374-question MME parity
            # (deployment_path mp, granularity 16): pass1 scores 1715.68
            # against a baseline of 1715.68 -- 0 flips, byte-identical --
            # and pass2_hit 1714.93 for 1 flip against a budget of 11.87,
            # hit ratio 0.965, coverage 1.0056, parse ratio 0.992. The hits
            # really crossed the connector: pass2_local_cached_tokens is 0,
            # so none of the 667520 skipped tokens came from vLLM's own
            # prefix cache.
            #
            # The preemption pool. Derived first from vLLM's own refusal --
            # it reports 0.04 GiB for the default 128 blocks and names 272
            # as the length that buys (~328 KB/block), so one max-length
            # (2048) request needs ~964 blocks, the floor it will not start
            # below -- and then confirmed under the padded prompts the suite
            # actually uses: 1024 blocks buys 2,325 tokens and yields 1
            # preemption (2026-08-22).
            #
            # That re-measurement matters, because the same number for
            # Gemma 4-E4B did NOT survive it (see that spec). This model
            # keeps working for a reason: its sliding window is 1024 tokens,
            # wider than the ~700-token padded prompt, so its per-request
            # footprint does not saturate and the batch can still outgrow
            # the pool. Gemma 4's 512-token window is narrower than its
            # prompt, and that is the difference.
            preemption_gpu_blocks=1024,
        ),
        ModelSpec(
            key="glm-4.6v-flash",
            hf_id="zai-org/GLM-4.6V-Flash",
            modalities=frozenset({"image", "video"}),
            chat_template_kwargs={"enable_thinking": False},
            mme_mm_processor_kwargs={
                "size": {"shortest_edge": 12544, "longest_edge": _MME_PIXEL_BUDGET}
            },
            min_decode_tokens=64,
            # Real MME photos draw far longer reasoning than the suite's
            # synthetic swatches: at 64 tokens, 42% of baseline answers
            # truncated before the boxed verdict (parse-ratio gate FAIL).
            mme_max_tokens=256,
            # Measured (2026-08-20): flips are exactly reproducible across
            # runs (27 p1-vs-base, 17 p2-vs-p1 of 2374; baseline reruns are
            # byte-identical, self-flips 0) and per-question inspection
            # showed only benign reasoning drift / repetition-loop
            # truncation on borderline items -- deterministic numeric-regime
            # divergence, not corruption. Control: with NO LMCache at all,
            # changing only max_num_seqs (default -> 64) flips 5.10%
            # (121/2374, score -0.67pt) -- vLLM kernels are not
            # batch-invariant, and the connector's 1.14% sits well below
            # the engine's own batch-sensitivity floor. 1.5% covers it.
            mme_max_flip_fraction=0.015,
            # Tempered: the preamble may OPEN a spurious unclosed box marker,
            # so the answer group must not span another begin marker.
            answer_extract_pattern=(
                r"<\|begin_of_box\|>((?:(?!<\|begin_of_box\|>).)*?)<\|end_of_box\|>"
            ),
        ),
        ModelSpec(
            key="qwen3-omni-30b",
            hf_id="Qwen/Qwen3-Omni-30B-A3B-Instruct",
            # The first AUDIO model in the suite. Audio reaches vLLM through
            # its own processor, resampler and encoder, none of which the
            # image path touches, and its contribution to the LMCache cache
            # key had never been exercised.
            modalities=frozenset({"image", "audio"}),
            # NOT a hybrid, measured two ways: the config's thinker text
            # tower is 48 uniform layers with no ``layer_types`` and
            # ``sliding_window=None`` (the MoE is FFN-only and does not
            # affect KV geometry), and the engine reports a single KV cache
            # group. So it runs the in-process path, unlike the Qwen3.5/3.6/
            # 3.8 hybrids. GQA-4 over 48 layers at head_dim 128 is
            # 96 KB/token, confirmed against the engine's own
            # 56.02 GiB / 611,888 tokens.
            #
            # 59.4 GiB of weights needs most of the card.
            gpu_memory_utilization=0.85,
            # Required on vLLM 0.23.0; see mm_encoder_attn_backend. Without
            # it the engine cannot start AT ALL for this model, audio-only
            # runs included.
            mm_encoder_attn_backend="TORCH_SDPA",
            # Measured, not guessed: at the 0.35 default the three scenarios
            # that let vLLM profile its own KV memory (chunked_prefill,
            # capacity_eviction, mp_connector) all died with "No available
            # memory for the cache blocks" -- 0.35 of a 140 GiB H200 is
            # 49 GiB against 59.4 GiB of weights. The preemption scenario
            # survived only because num_gpu_blocks_override skips that
            # check, which is exactly how a too-small fraction hides.
            isolated_gpu_utilization=0.75,
            parity_benchmark="mmau",
            # MMAU rather than MME: the audio benchmark. Full 1000-question
            # parity measured 0 flips on both comparisons, byte-identical
            # scores across baseline/pass1/pass2 (66.90; music 70.06 /
            # sound 71.47 / speech 59.16) and a 1.000 lookup hit ratio, so
            # the default flip budget is untouched. Prompts average ~234
            # tokens, moving ~28 GB through the cache, well inside the
            # runner's 40 GB default -- no capacity override needed.
        ),
        ModelSpec(
            key="molmo2-4b",
            hf_id="allenai/Molmo2-4B",
            # The only member of the four-model "hitchhiker" batch that vLLM
            # 0.23.0 can serve at all; the other three fail before LMCache
            # is involved (see records/2026/08/22/10_).
            #
            # Image only, deliberately: the config carries frame tokens and
            # vLLM registers a video processor, but neither was exercised
            # here, so the certificate must not claim video.
            modalities=frozenset({"image"}),
            # transformers 5.15 refuses this repo's config without it
            # ("contains custom code which must be executed"), even though
            # vLLM implements Molmo 2 natively in `molmo2.py`. Measured: a
            # probe with the flag loads and answers a synthetic image
            # correctly; the same probe without it dies in ModelConfig.
            trust_remote_code=True,
            # Measured, not assumed: `create_model_config()` reports
            # is_mm_prefix_lm=True, so vLLM forces disable_chunked_mm_input
            # and refuses any batched-token budget below this model's
            # worst-case mm item (8134 tokens). The first NON-HYBRID model
            # in the suite with this property, which is why the
            # chunked-prefill exclusion had to stop being keyed on
            # hybrid family.
            mm_bidirectional_attention=True,
            # Its chat template raises `Conversation roles must alternate
            # user/assistant/...` on the system message every suite request
            # carries; the salt moves to the front of the user message
            # instead.
            supports_system_role=False,
            # ...which is still not enough, because the template then hoists
            # `<|image|>` above `<|im_start|>` -- the content order is not
            # the caller's to choose. So the case identity goes into the
            # media bytes instead. Measured: `<|image|>` precedes the
            # conversation in every rendered prompt.
            media_first_template=True,
            # And its prompt is not append-only in media either: adding a
            # second image changes the tokens from index 1 (measured: the
            # one-image and two-image prompts share exactly ONE token), so
            # T2.2's shared-prefix premise does not hold.
            media_prefix_stable=False,
            # Measured (2026-08-22) on the live engine: ONE KV cache group
            # (FullAttentionSpec, block 16, 36 layers, 64 KiB per page), so
            # 144 KB/token -- GQA-8 at head_dim 128, the widest KV of any
            # in-process model in the suite. Not a hybrid, so no
            # hybrid_block_tokens.
            #
            # Full MME is 2374 questions at ~770 prompt tokens (measured:
            # a 1540x1540 photo lands at 770 including the template), so
            # ~263 GB of KV. The 40 GB default would evict every entry
            # before its pass-2 revisit and fail the hit gate at ~0; 340 GB
            # holds the run with room for longer questions.
            mme_max_local_cpu_gb=340.0,
            # The 128-block default makes the preemption scenario VACUOUS
            # here (measured: 0 preemptions), and the reason is prompt
            # length in tokens, not KV width -- the pool is counted in
            # blocks, so every model gets the same 2048 tokens. One Molmo 2
            # image request is 787 tokens = 50 blocks against roughly a
            # tenth of that on the Qwen models, so at 128 blocks only two
            # of the six requests ever run and their decode growth
            # (7 blocks each) never overflows. The window is
            # [6 x 50 = 300, 6 x 57 = 342): admit all six prompts, refuse
            # to hold their decode growth. 320 sits inside it.
            preemption_gpu_blocks=320,
            # Eight units of the eviction path's default, because one is
            # not enough to hold a single request: at 64 MB this model
            # stored zero bytes (0 active allocations, 32 requests missing
            # on pass 1). At 512 MB the scenario is healthy -- 181 resident
            # keys, 427032576 bytes, 0.795 of the cap, against 3.70 GB of
            # intended traffic (6.9x overflow). See the reasoning at
            # isolated_cases.EVICTION_CAPACITY_GB.
            eviction_capacity_gb=0.5,
            # No mme_mm_processor_kwargs: the other specs cap photos at
            # ~768 image tokens, and Molmo 2's own processor already lands
            # there (770 total for a 1540x1540 input) without a cap.
        ),
    ]
}


def selected_model_keys() -> list[str]:
    """Return the model keys selected via ``LMCACHE_MM_E2E_MODELS``.

    Returns:
        The comma-separated keys from the environment, defaulting to
        ``["qwen2.5-vl-3b"]``.

    Raises:
        KeyError: If a selected key is not registered in ``MODEL_SPECS``.
    """
    raw = os.environ.get("LMCACHE_MM_E2E_MODELS", "qwen2.5-vl-3b")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    for k in keys:
        if k not in MODEL_SPECS:
            raise KeyError(
                f"Unknown model key {k!r}; registered keys: {sorted(MODEL_SPECS)}"
            )
    return keys
