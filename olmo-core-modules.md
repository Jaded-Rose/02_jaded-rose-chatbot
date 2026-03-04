# OLMo-core Module Reference

Repository: https://github.com/allenai/OLMo-core

OLMo-core (`src/olmo_core/`) is the core Python library from Allen AI for building, training, and evaluating OLMo language models (OLMo-2, OLMo-3).

## Module Overview

### 1. `nn` — Neural Network Components
All `torch.nn.Module` implementations for building transformer language models.

- **`nn.transformer`** — Full transformer model implementations: `Transformer`, `NormalizedTransformer`, `MoETransformer`. Many block variants (`ReorderedNormTransformerBlock`, `PeriNormTransformerBlock`, `LayerNormScaledTransformerBlock`, MoE hybrid blocks). Configured via `TransformerConfig` and `TransformerBlockConfig`.
- **`nn.attention`** — Attention mechanisms with multiple backends: FlashAttention 2/3/4, TransformerEngine, Torch native, plus Ring/Ulysses context-parallel attention. Includes `GatedDeltaNet` (recurrent sequence mixer) and KV cache management.
- **`nn.moe`** — Mixture-of-Experts: `DroplessMoE`, MoE routers (`MoELinearRouter`), MoE MLPs, load-balancing loss.
- **`nn.functional`** — `cross_entropy_loss`, `fused_linear_cross_entropy_loss`, `l2_normalize`.
- **`nn.hf`** — Hugging Face integration: convert state dicts between OLMo-core and HF formats, save/load HF models.
- **`nn.conversion`** — Generic state conversion: `StateConverter`, `StateMapping`, `StateMappingTemplate`.
- Other: `feed_forward.py`, `layer_norm.py`, `lm_head.py`, `rope.py` (RoPE), `residual_stream.py`, `buffer_cache.py`, `convolution.py`.

### 2. `data` — Datasets and Data Loading
Dataset, data loader, and config builders for the Trainer.

- **`data.composable`** — Composable data loading API: `TokenSource` → `InstanceSource` → `ComposableDataLoader`. Supports mixing, sampling, slicing, packing, curriculum learning, multi-epoch training. Key classes: `NumpyDocumentSource`, `ConcatAndChunkInstanceSource`, `PackingInstanceSource`, `MixingInstanceSource`, `ComposableDataLoader`.
- **`data.numpy_dataset`** — Numpy-backed datasets for fixed-sequence-length (FSL) and variable-sequence-length (VSL) training with curriculum support.
- **`data.data_loader`** — `NumpyDataLoaderConfig`, `NumpyFSLDataLoader`, `NumpyVSLDataLoader`.
- **`data.tokenizer`** — `TokenizerConfig`, `TokenizerName`.
- **`data.collator`** — `DataCollator` with `PaddingDirection`.
- **`data.mixes`** — Pre-defined data mix configurations (`DataMix`, `DataMixBase`).

### 3. `train` — Training Loop
Highly efficient, flexible language model trainer.

- **`train.Trainer` / `train.TrainerConfig`** — Main trainer: async checkpointing, any parallel strategy, async metric logging, callback system.
- **`train.train_module`** — `TrainModule`, `BasicTrainModule`, `TransformerTrainModule`, `TransformerPipelineTrainModule`. Configs for activation checkpointing and all parallelism types.
- **`train.callbacks`** — ~20 built-in callbacks: `CheckpointerCallback`, `EvaluatorCallback`, `SpeedMonitorCallback`, `WandBCallback`, `CometCallback`, `ProfilerCallback`, `GarbageCollectorCallback`, `GPUMemoryMonitorCallback`, `SlackNotifierCallback`, `BeakerCallback`, `BatchSizeSchedulerCallback`, `StabilityMonitorCallback`, `ModelMergeCallback`, etc.
- Helpers: `prepare_training_environment()`, `teardown_training_environment()`.

### 4. `distributed` — Distributed Training
APIs for distributed communication, bookkeeping, and checkpointing.

- **`distributed.checkpoint`** — High-level distributed checkpointing: `save_model_and_optim_state()`, `load_model_and_optim_state()`, `unshard_checkpoint()`, async save, remote storage (S3/GCS).
- **`distributed.parallel`** — Parallelism strategies with `DeviceMesh`: `DataParallelConfig` (FSDP/HSDP/DDP), `TensorParallelConfig`, `ContextParallelConfig`, `ExpertParallelConfig`, `PipelineParallelConfig`. Functions: `build_world_mesh()`, `get_dp_mesh()`, `get_tp_mesh()`, `get_cp_mesh()`, `get_pp_mesh()`, `get_ep_mesh()`.

### 5. `optim` — Optimizers and Schedulers
- **Optimizers**: `AdamWConfig`, `AdamConfig`, `LionConfig`, `MuonConfig`, `NorMuonConfig`, `DionConfig`, skip-step variants (`SkipStepAdamW`, `SkipStepLion`).
- **Schedulers**: `CosWithWarmup`, `LinearWithWarmup`, `WSD`, `WSDS`, `InvSqrtWithWarmup`, `ConstantWithWarmup`, `SequentialScheduler`, `HalfCosWithWarmup`, `ExponentialScheduler`.

### 6. `eval` — Evaluation
- `Evaluator`, `LMEvaluator` — Evaluators for running eval during training.
- `Metric`, `MeanMetric` — Metric abstractions.

### 7. `generate` — Text Generation
- `GenerationModule`, `GenerationConfig` — Base generation abstractions.
- `TransformerGenerationModule` — Autoregressive generation with a built-in chat CLI.

### 8. `float8` — Low-Precision Training
FP8 training via torchao:
- `Float8Config`, `AOFloat8LinearConfig`, `AOFloat8LinearRecipe`, `AOMXLinearConfig` (MXFP8/MXFP4).

### 9. `launch` — Experiment Launching
Launching experiments on Beaker, GCP. Rank reordering and host selection.

### 10. `model_ladder` — Model Scaling
- `ModelLadder`, `ModelConfigurator`, `RunConfigurator` — Scaling experiment framework.
- `TransformerModelConfigurator`, `Olmo3ModelConfigurator`, `TransformerSize`.
- `WSDSChinchillaRunConfigurator` — Chinchilla-optimal run configuration.

### 11. `ops` — Custom Operations
Autograd functions: `attach_auxiliary_loss()` for MoE auxiliary losses.

### 12. `kernels` — Custom CUDA Kernels
QuACK-based kernel implementations.

### 13. `testing` — Test Utilities
Pytest markers, device detection, `run_distributed_test()` helper for multi-GPU tests.

### 14. Utility Modules
- **`config.py`** — Base `Config` dataclass, `DType`, `StrEnum`, YAML/JSON serialization.
- **`io.py`** — File I/O for local, S3, GCS, HTTP via `cached_path`.
- **`utils.py`** — Seeding, device detection, logging.
- **`aliases.py`** — Type aliases.
- **`exceptions.py`** — Custom exceptions.
- **`fs_cache.py`** — Filesystem caching.
