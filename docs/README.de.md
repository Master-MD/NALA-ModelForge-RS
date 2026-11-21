# NALA-ModelForge-RS (`v1.5.0`) — Deutsch 🇩🇪🇨🇭

**100 % Native Safe Rust + Native Apple Metal GPU (0,0 % Python · 0 externe Crates · `#![forbid(unsafe_code)]`)**  
Das Tri-OS Precision-Studio für **macOS (Apple Silicon & Intel)**, **Ubuntu/Linux (x64 & ARM64)** und **Windows 11 (x64 & ARM64)**, um riesige LLMs (bis 1,5 TB MoE) per Drag & Drop exakt auf deine eigene Hardware zuzuschneiden.

## Kernfunktionen
- **Zero-Copy Drag & Drop Inspektor:** Ziehe `.gguf`, `.safetensors` oder HuggingFace-Ordner direkt ins macOS-Fenster (`< 25 ms` Header-Scan ohne RAM-Kopie, selbst bei 1,5 TB Modellen!).
- **Domain-Selective Mixed-Precision & MoE Expert Pruning:** Coding, Logik, Attention (`attn_q/k/v/o`), Router (`FP16`) & Embeddings bleiben **FETT** (`BF16`/`Q8_0`/`Q6_K`), während sekundäre Gewichte mit **Randomized Walsh-Hadamard-Rotation (`H·D·x`)** auf **Bonsai 1-Bit (`1.125 bpw`)** oder **BitNet 1.58-Bit** komprimiert oder wie bei **Gyro-Coder** strukturell geprunt werden (`-46 %` VRAM).
- **Asymmetrischer Multi-User & Agenten KV-Planer:** z. B. `1× Hauptuser (1M Tokens)` + `1× Orchestrator (1M)` + `14× Spezialisten-Agenten (256K)` mit `Q8_0`/`Q4_0` KV, Radix Prefix-Sharing, YaRN RoPE & **NALA-Strata-RS NVMe Disk-Parking**.
- **Echtes Uncensoring (`RSI-MF-03` & `RSI-MF-06`):** Norm-Preserving Biprojected Refusal-Abliteration (`W' = W - α(W r_orth)r_orthᵀ`) direkt im Streaming-GGUF (`F32`/`F16`/`Q8_0`/`Q4_0`, `< 16 MB` RAM, atomarer Backup-Guard).
- **Multi-LoRA Stacker & Static Merger (`RSI-MF-07`):** Hänge 1..N LoRAs (`.gguf` / `.safetensors`) an, prüfe Spectral Drift und schmelze sie mit **0 ms Inferenz-Overhead** direkt ins Ziel-GGUF ein.
- **Native Apple Metal GPU (`RSI-MF-05`):** **`28,7×` Speedup / `6,66 GWeights/s`** (16,7 Mio. Gewichte in `2,52 ms`, `0` Bitfehler auf M4 Max!).
- **NALA-RAG (`:18700`) & NALA-MCP-cORe (28 Tools):** Live Evidence-Lock RAG-Bridge (`RSI-MF-08` 5-Zonen Corpus-Generator mit Zero-PII Redactor) + 28-Tool JSON-RPC 2.0 MCP Server (`--mcp-stdio`).

## 1-2-3 Schnellstart (Caveman)
1. **macOS:** `.dmg` öffnen → `NALA-ModelForge.app` nach `Programme` ziehen → öffnen!
2. **Ubuntu/Debian:** `sudo dpkg -i nala-modelforge-rs_1.5.0_amd64.deb && nala-modelforge-rs --open`
3. **Windows 11:** `.zip` entpacken → `NALA-ModelForge.exe` doppelklicken!
4. **MCP-cORe & IDEs verbinden:** `nala-modelforge-rs --register-mcp`
