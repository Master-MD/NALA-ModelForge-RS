# NALA-ModelForge-RS (`v1.5.0`) — Rumantsch 🇨🇭

**100 % Rust Nativ Segir + GPU Apple Metal Nativa (0,0 % Python · 0 Crates Externs · `#![forbid(unsafe_code)]`)**  
Il studio da precisiun Tri-OS (**macOS ARM64/Intel**, **Ubuntu/Linux x64/ARM64**, **Windows 11**) per quantisar, tagliar experts MoE, abliterar, fusiunar LoRAs e planisar KV-Cache multi-agents per models fin 1,5 TB.

## Funcziuns Principalas
- **Inspectur Drag & Drop Zero-Copy:** Tira `.gguf`, `.safetensors` u ordinaturs HuggingFace directamain en la fanestra (`< 25 ms`, nagina copia en RAM).
- **Quantisaziun Mista Selectiva & Pruning MoE:** Coding, Logica, Attention e Router (`FP16`) restan precis (`BF16`/`Q8_0`), il rest vegn comprimì cun **Rotaziun Hadamard (`H·D·x`)** sin **Bonsai 1-Bit (`1.125 bpw`)** (`-46 %` VRAM).
- **Planisader KV Asimetric Multi-Agents:** P. ex. `1× Utilisader (1M)` + `1× Orchestratur (1M)` + `14× Agents (256K)` cun KV `Q8_0` e Strata NVMe Disk-Parking.
- **Abliteraziun Reala (`RSI-MF-03` & `RSI-MF-06`):** Ortogonalisaziun normativa en streaming direct sin GGUF (`F32/F16/Q8_0/Q4_0`, `< 16 MB` RAM, protecziun atomara).
- **Multi-LoRA Stacker & Fusiun Statica (`RSI-MF-07`):** Fusiun statica cun `0 ms` overhead e Spectral-Drift Guard.
- **Punt GPU Apple Metal (`RSI-MF-05`):** **`28,7×` pli svelt / `6,66 GWeights/s`** (`0` errurs sin M4 Max).
- **NALA-RAG (`:18700`) & NALA-MCP-cORe (28 Utensils):** Generatur da datasets 5-Zonas Zero-PII (`RSI-MF-08`) e server MCP JSON-RPC 2.0.

## Cumenzament Svelt (1-2-3)
1. **macOS:** Avrir `.dmg` → tirar `NALA-ModelForge.app` en `Applicaziuns`.
2. **Ubuntu:** `sudo dpkg -i nala-modelforge-rs_1.5.0_amd64.deb && nala-modelforge-rs --open`
3. **Windows 11:** Extrair `.zip` → aviar `NALA-ModelForge.exe`.
4. **Colliar MCP-cORe:** `nala-modelforge-rs --register-mcp`
