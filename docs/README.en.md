# NALA-ModelForge-RS (`v1.5.0`) — English 🇬🇧🇺🇸

**100% Native Safe Rust + Native Apple Metal GPU (0.0% Python · 0 External Crates · `#![forbid(unsafe_code)]`)**  
The Tri-OS Precision Studio for **macOS (Apple Silicon & Intel)**, **Ubuntu/Linux (x64 & ARM64)**, and **Windows 11 (x64 & ARM64)** to surgically quantize, prune, abliterate, merge LoRAs, and plan multi-agent KV budgets for frontier LLMs up to 1.5 TB.

## Core Capabilities
- **Zero-Copy Drag & Drop Inspector:** Drop `.gguf`, `.safetensors`, or HuggingFace folders directly onto the native macOS window (`< 25 ms` header parse with zero RAM copy even on 1.5 TB models).
- **Domain-Selective Mixed-Precision & MoE Expert Pruning:** Keep Coding, Logic, Attention, Router Gates (`FP16`), and Embeddings high-precision (`BF16`/`Q8_0`/`Q6_K`) while compressing secondary weights with **Randomized Walsh-Hadamard Rotation (`H·D·x`)** to **Bonsai 1-Bit (`1.125 bpw`)** or **BitNet 1.58-Bit**, or structurally pruning MoE experts (`-46%` VRAM like Gyro-Coder).
- **Asymmetric Multi-User & Multi-Agent KV-Cache Planner:** Plan heterogeneous slots (e.g., `1× Human 1M` + `1× Orchestrator 1M` + `14× Agents 256K`) with `Q8_0`/`Q4_0` KV, Radix Prefix Sharing, YaRN RoPE scaling, and **NALA-Strata-RS NVMe Disk-Parking**.
- **True Uncensoring (`RSI-MF-03` & `RSI-MF-06`):** Norm-Preserving Biprojected Refusal Abliteration directly on streaming GGUF tensors (`F32`/`F16`/`Q8_0`/`Q4_0`, `< 16 MB` RAM, atomic source protection).
- **Multi-LoRA Stacker & Static Merger (`RSI-MF-07`):** Attach 1..N LoRA adapters (`.gguf`/`.safetensors`), guard spectral drift, and fuse statically into GGUF with **0 ms inference overhead**.
- **Direct Apple Silicon Metal GPU (`RSI-MF-05`):** **`28.7×` speedup / `6.66 GWeights/s`** (16.7M weights in `2.52 ms`, `0` bit mismatches on M4 Max).
- **NALA-RAG (`:18700`) & NALA-MCP-cORe (28 Tools):** Live Evidence-Lock RAG connector (`RSI-MF-08` 5-Zone Corpus-to-Imatrix/LoRA generator with Zero-PII redaction) + 28-tool JSON-RPC 2.0 MCP server (`--mcp-stdio`).

## 1-2-3 Quick Start
1. **macOS:** Open `.dmg` → drag `NALA-ModelForge.app` to `Applications` → launch!
2. **Ubuntu/Debian:** `sudo dpkg -i nala-modelforge-rs_1.5.0_amd64.deb && nala-modelforge-rs --open`
3. **Windows 11:** Extract `.zip` → double-click `NALA-ModelForge.exe`!
4. **Wire MCP-cORe & IDEs:** `nala-modelforge-rs --register-mcp`
