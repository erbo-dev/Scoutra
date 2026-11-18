# G. Video-VLM Research
**Decision:** evaluate a Video-VLM alongside specialized models. The inspected evidence does not support replacing continuous detection, metric tracking, and persistent player identity with one large VLM.

## 1. What “video understanding” can mean
A model may answer a question about a known short clip, classify a window, propose an event timestamp, explain a foul, or maintain dense trajectories over a match. These are different tasks. Success on clip QA or a broad video benchmark does not establish the last capability.

The inspected acato implementation uses sampled frames and text requests to a vLLM server. Its historical dual-pass path turns 8B observations into text for 32B classification; its direct single-pass path sends visual windows to 32B. It is not one continuous 90-minute multimodal state. [C-dual] [C-triage]

## 2. General Qwen models requested by the brief

| Family | Verified public evidence | Relevant mechanism | Scoutra interpretation |
|---|---|---|---|
| Qwen3-VL | Official repository and technical report; 8B/32B model metadata and weight-file listings | Native image/video processing, timestamp-aware visual prompting, long context | Established baseline for visual event QA; long context is not guaranteed player memory [Q-readme] [Q-paper] |
| Qwen3.5 | Official 9B/27B model cards and Transformers documentation; real public checkpoints | Native multimodal architecture; hybrid attention/linear-state backbone in documented family | Candidate comparison, not merely a hypothetical future model; no Scoutra football validation [Q35-card] [Q35-doc] |
| Qwen3.6 | Official `Qwen/Qwen3.6-35B-A3B` card and shards listed | 35B total/3B active MoE; same Transformers family/model type as Qwen3.5; documented video input | Applicable as of this snapshot; all experts still consume storage; no verified football adaptation found in this audit [Q36-card] [Q35-doc] |

The old `QwenLM/Qwen3.5` repository URL redirects; the current repository is titled Qwen3.8 and documents the broader series. Model-specific cards are therefore better evidence for exact 3.5/3.6 checkpoint identity than assuming the old repository name is stable. This audit does not expand model recommendations to every later release. [Q-new]

Do not treat a long advertised context length as proof that jersey #8 remains the same physical player over 90 minutes. Input frame selection, spatial resolution, visual-token limits, attention/state behavior, and ambiguous evidence determine what information actually enters and survives the request.

## 3. Football-specific public models and implementations

| Model / source | Training source and artifact evidence | What it actually targets | Actor and temporal limitations | Reproducibility status |
|---|---|---|---|---|
| acatorcini Qwen3-VL-32B soccer v11 FP8 | A: custom amateur/youth clips; V: 8 safetensors shards listed | Goals, shots-on-target, free-kick motions, keeper saves/catches, corners, throws, goal kicks plus auxiliary states | Window event timestamps; no validated persistent actor association | Published merged artifact; narrow vLLM/quantization recipe; private training corpus/base ancestry [C-weights] [C-adapter] |
| acatorcini v11 LoRA | V: adapter/config/training metadata listed; predecessor v8 BF16 merge named | Same event specialization | Applying to original public base is not exact production model | Partial reproducibility: predecessor unavailable in audited public assets [C-adapter] [C-train] |
| SoccerChat Qwen2-VL-7B | SoccerNet-derived multimodal instructions, event/color/commentary/ASR context; adapter listed | Clip QA, event discussion, commentary and referee reasoning | Textual descriptions/names do not create a frame-to-player roster mapping; temporal spotting not established by QA alone | Public adapter; documented 4-bit T4 example uses 24 frames/limited pixels; benchmark used unquantized model [CHAT-paper] [CHAT-card] |
| OpenSportsLab Qwen3-VL-8B XFoul LoRA | OSL-XFoul; adapter and config metadata listed | Foul/referee-style VQA | Clip-level answers, not match-wide event localization or tracking | Native QwenVL source, PEFT path, quantization config; AGPL model/code terms [OSL-model] [OSL-card] |
| OpenSportsLab Qwen2.5-VL-7B XFoul LoRA | Model zoo identifies dataset and checkpoint | Same specialist foul VQA comparison | Same actor/spotting gap | Publicly documented checkpoint; not independently executed [OSL-zoo] |
| kpyu/soccernet-vlm-tsi | Card identifies curated SoccerNet/TGLG data and VLM-TSI adapter | Domain-adapted video-language understanding | No whole-match identity evidence established | Small adapter card/provenance only inspected; inference reproduction and base/license chain require further audit [TSI-card] |
| sportsvision/soccer-qa-4b | Card describes soccer QA; complete safetensors/inference files listed | 16-frame 256×256 QA using DWT-VJEPA2 + Llama 3.2-3B | Not Qwen; natural-language answer, no actor/time schema verified | CC-BY-NC-4.0; dataset detail/evaluation insufficient for recommendation [QA-card] |

**A: specialist QA comparison.** The OSL Qwen3-VL XFoul card reports 69.96% accuracy and 48.33% balanced accuracy on its stated benchmark. The library lists 65.77% / 36.59% for its Qwen2.5-VL variant and 72.24% / 50.00% for X-VARS. Those are foul-QA results, not actor-linked event spotting, and their gap between accuracy and balanced accuracy matters under class imbalance. No Scoutra transfer is established. [OSL-card] [OSL-zoo]

This is a focused discovery list, not an assertion that no other football weights exist. Hugging Face file listings establish artifacts, not numeric correctness or completeness of their training provenance.

### Relevant non-Qwen controls
InternVL3.5-8B has an official video-capable checkpoint and loading examples. InternVideo2.5 provides a long/rich-context video MLLM implementation. Either can be a matched-budget control, provided exact base/weight licenses and quantization kernels are checked. Neither was found to have football-specific evidence sufficient to replace a specialist pipeline in this audit. Their general video scores should not be ranked against acato's event-coverage numbers. [IV-card] [IV-video]

## 4. acato's actual event pipeline
The reusable pattern is **sample → propose → verify/ground → deduplicate → assemble evidence clips**. The source supports configurable windows, model swapping, VLM response parsing, field crops, ball-context/trajectory options, and post-filters. Optional flags should be audited for the chosen run; a config field alone does not prove an active production feature. [C-dual] [C-config]

Its current model card advertises v11 using 4–8 sampled frames for a short window, while config/source expose several other window/cadence modes. Preserve the exact mode in evaluation. The nine core event types named in the current architecture are not the same thing as every enum value in `models.py`; the enum also includes types not established as learned classes in that checkpoint. [C-arch] [C-models] [C-weights]

The schema includes optional actor/box/field fields. Inspection did not establish a current detector path assigning every event to a verified physical player. Goalkeeper reel assignment or kit color is much weaker than attributing every pass/tackle to the correct participant. [C-models] [C-dual]

**Reported evidence:** four tuned games, ±60s coverage; goal union precision 0.17 despite recall 1.00. Saves and shot proposals also involve proxy tiers. This is a recall-oriented retrieval strategy. Exact event totals, tight temporal localization and actor accuracy remain unproven. [C-status]

**Known errors reported by authors:** goalmouth outside frame, tiny-ball failure, silent parries, unreliable save subtypes, free-kick labels triggered by defensive clearances, and venue/framing shift. A kickoff after a goal is useful indirect evidence but does not show who scored or the exact scoring moment. [C-status]

## 5. Source and artifact contradictions to resolve
1. The v11 adapter was continued from an internal merged v8 model. Attaching it to the public original base cannot be assumed numerically equivalent. [C-adapter]
2. The v11 card declares a Tongyi license and commercial MAU language, while the official upstream Qwen3-VL-32B metadata declares Apache-2.0. The custom artifact's intended grant is unresolved; this report does not silently override either. [C-weights] [Q32-card]
3. The hardware card's minimum mentions NVLink, but the repository's operational guide reports two RTX 3090s without NVLink. Treat minimum-VRAM and topology statements as author recipes requiring a controlled load test, not authoritative guarantees. [C-weights] [C-gpu]
4. Older orchestration documents reference private scripts and files absent from the audited source tree. They are not verified available deployment code. [C-gpu]
5. A stated vLLM 0.19.1 requirement and “newer releases break output” are author observations; pinning makes a reproduction proposal precise, but this audit did not verify the version bug. [C-weights]

## 6. VRAM and quantized multi-GPU feasibility
The following are **parameter-storage lower-bound calculations**, not measured peak VRAM. Approximation: parameters × bits ÷ 8. Real models add vision components, scales, nonquantized layers, KV/state caches, activations, workspace, padding and serving buffers.

| Nominal parameter count | FP16/BF16 weight bytes, decimal GB | 8-bit | 4-bit ideal | Practical implication |
|---:|---:|---:|---:|---|
| 4B | 8 | 4 | 2 | Small-clip candidate; runtime still must fit |
| 8B | 16 | 8 | 4 | FP16 leaves little/no room on a 16-GB T4; quantized path is more plausible |
| 9B | 18 | 9 | 4.5 | A 4-bit 3.5 candidate needs model/kernel validation |
| 27B | 54 | 27 | 13.5 | 4-bit weight fit alone does not ensure 16-GB video inference |
| 32B | 64 | 32 | 16 | A 4-bit split over two T4s is a testable separate artifact/engine configuration |
| 35B MoE | 70 | 35 | 17.5 | Storage uses total, not active parameters |

The acato merged FP8 card reports ~34 GB. That artifact cannot be made into a two-T4 fit by setting tensor parallel size to 2: total nominal capacity is only 32 GB and runtime adds overhead. On H100, FP8 and larger windows are more plausible, but actual memory depends on the H100 variant and request limits. [C-weights] [V-quant]

**Technical support versus deployment proof:**
- vLLM supports tensor parallelism; acato publishes a TP=2 serving recipe. On 3090-class GPUs, the relevant implementation may use weight-only/fallback kernels rather than native FP8 W8A8. Do not equate the word “FP8” with hardware-native arithmetic. [C-weights] [V-quant]
- Turing/T4 has documented AWQ/GPTQ/bitsandbytes options, but not native FP8 W8A8 in the inspected vLLM matrix. Quantized multimodal model support must be tested at the chosen engine version. [V-quant]
- OpenSportsLib's native backend builds bitsandbytes config. Its quantized load path pins to the current CUDA device, while the other path can use a requested `device_map`. Thus “the library supports quantization” does not prove automatic quantized two-GPU sharding in every configuration. [OSL-model]
- Transformers `device_map="auto"` partitions layers; it is not identical to high-throughput tensor parallel inference. Both may communicate over PCIe and need separate correctness/throughput measurements. [Q36-card] [V-recipe]

## 7. Could a VLM replace significant parts of Scoutra?
**Potentially replaceable, subject to tests:** some hand-written scene/event proposal rules, selected clip classification, foul/event explanation, and report wording. A properly adapted model could reduce separate event classifiers if it achieves comparable actor+time precision/recall and compute cost.

**Keep specialized measurement initially:** player/ball detection at adequate spatial/temporal resolution; local tracking and persistent identity; camera calibration; metric trajectories and kinematics; ball-control evidence; exact counting. None of the inspected football VLM implementations establishes reliable replacement of these on full amateur matches. [C-dual] [OSL-model] [CHAT-card]

A grounded VLM request should receive the relevant clip, timestamps, candidate player IDs/boxes and evidence-quality flags. It should be allowed to abstain. Store its answer as an inference with provenance, not overwrite detector observations. Evaluate whether extra context causes model agreement without actual visual correctness.

Long-video conversation memory should be a structured event/identity store with retrieved evidence. An ever-growing chat transcript is not a reliable trajectory database. This is an architectural recommendation, not an empirically proven optimal design.

## 8. Questions experiments must settle
Does football adaptation beat a newer general Qwen model on unseen venues? Does crop/context selection help tiny players? What accuracy is lost by 4-bit quantization and sparse temporal sampling? Can a model return actor+time correctly without being supplied the answer in its prompt? Does two-stage filtering miss rare events? Does the model remain calibrated when the goal or ball is invisible? The experiment plan separates these questions and does not execute any of them.

## Direct sources

[C-dual]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/src/detection/dual_pass_detector.py "acato/soccer-video-pipeline — src/detection/dual_pass_detector.py"
[C-triage]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/src/detection/triage_scanner.py "acato/soccer-video-pipeline — src/detection/triage_scanner.py"
[Q-readme]: https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/README.md "QwenLM/Qwen3-VL — README.md"
[Q-paper]: https://arxiv.org/abs/2511.21631 "Qwen3-VL technical report"
[Q35-card]: https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/README.md "Official Qwen3.5-9B model card"
[Q35-doc]: https://huggingface.co/docs/transformers/model_doc/qwen3_5 "Transformers Qwen3.5 documentation"
[Q36-card]: https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/345d467b117d1fd5eda9b879a5988de9ae5f47a7/README.md "Official Qwen3.6-35B-A3B model card"
[Q-new]: https://github.com/QwenLM/Qwen3.8/blob/2ea10dc725823bf7c3e21ce8557cbe15245132ae/README.md "QwenLM/Qwen3.8 — README.md"
[C-weights]: https://huggingface.co/acatorcini/qwen3-vl-32b-soccer-v11-fp8/blob/ab56ac6743fc4f5ec299ed951f989d37f5874c56/README.md "acato v11 FP8 checkpoint card"
[C-adapter]: https://huggingface.co/acatorcini/qwen3-vl-32b-soccer-v11-lora/blob/5aca84c3c96f42d582346710fd22322ae0ed6d6b/README.md "acato v11 LoRA card"
[C-train]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/scripts/train_lora_v11.sh "acato/soccer-video-pipeline — scripts/train_lora_v11.sh"
[CHAT-paper]: https://arxiv.org/abs/2505.16630 "SoccerChat paper"
[CHAT-card]: https://huggingface.co/SimulaMet/SoccerChat-qwen2-vl-7b/blob/29871536004c0ac788af09cbb87969ab9f6e1410/README.md "SoccerChat checkpoint card"
[OSL-model]: https://github.com/OpenSportsLab/opensportslib/blob/b2fe8f16a35336d93b1b0eabe1741bbbbc36cd68/opensportslib/models/base/qwen_vl_native.py "OpenSportsLab/opensportslib — opensportslib/models/base/qwen_vl_native.py"
[OSL-card]: https://huggingface.co/OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora/blob/8c7721ff5fd74a1ad27f64203aa3218c5f9f5db1/README.md "OSL Qwen3-VL XFoul checkpoint card"
[OSL-zoo]: https://github.com/OpenSportsLab/opensportslib/blob/b2fe8f16a35336d93b1b0eabe1741bbbbc36cd68/docs/model-zoo.md "OpenSportsLab/opensportslib — docs/model-zoo.md"
[TSI-card]: https://huggingface.co/kpyu/soccernet-vlm-tsi/blob/7dfea238e0fdc65b0562821257e66a6170d43051/README.md "SoccerNet VLM-TSI checkpoint card"
[QA-card]: https://huggingface.co/sportsvision/soccer-qa-4b/blob/e0ef6b8b795fb91c0998c089ab2dc5c291b1ce05/README.md "Soccer QA 4B checkpoint card"
[IV-card]: https://huggingface.co/OpenGVLab/InternVL3_5-8B "InternVL3.5-8B official checkpoint"
[IV-video]: https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2.5 "InternVideo2.5 official implementation"
[C-config]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/src/config.py "acato/soccer-video-pipeline — src/config.py"
[C-arch]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/docs/architecture.md "acato/soccer-video-pipeline — docs/architecture.md"
[C-models]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/src/detection/models.py "acato/soccer-video-pipeline — src/detection/models.py"
[C-status]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/docs/pipeline_status.md "acato/soccer-video-pipeline — docs/pipeline_status.md"
[Q32-card]: https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct/blob/0cfaf48183f594c314753d30a4c4974bc75f3ccb/README.md "Official Qwen3-VL-32B model card"
[C-gpu]: https://github.com/acato/soccer-video-pipeline/blob/0f7e01e46e4023ed1bc456a8cdaf7ecb62f764b1/docs/vllm-gpu-orchestration.md "acato/soccer-video-pipeline — docs/vllm-gpu-orchestration.md"
[V-quant]: https://docs.vllm.ai/en/v0.19.1/features/quantization/#supported-hardware "vLLM 0.19.1 quantization hardware matrix"
[V-recipe]: https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3.5.md "vLLM Qwen3.5/3.6 serving recipe"
