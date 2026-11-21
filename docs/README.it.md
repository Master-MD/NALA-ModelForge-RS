# NALA-ModelForge-RS (`v1.5.0`) — Italiano 🇮🇹🇨🇭

**100 % Rust Nativo Sicuro + GPU Apple Metal Nativa (0,0 % Python · 0 Crate Esterni · `#![forbid(unsafe_code)]`)**  
Lo studio di precisione Tri-OS (**macOS ARM64/Intel**, **Ubuntu/Linux x64/ARM64**, **Windows 11**) per quantizzare, potare esperti MoE, abliterare, fondere LoRA e pianificare KV-Cache multi-agente fino a 1,5 TB.

## Funzionalità Principali
- **Ispettore Drag & Drop Zero-Copy:** Trascina `.gguf`, `.safetensors` o cartelle HuggingFace (`< 25 ms`, zero copia in RAM anche su 1,5 TB).
- **Quantizzazione Mista Selettiva & Pruning MoE:** Codice, Logica, Attention e Router (`FP16`) restano ad alta precisione (`BF16`/`Q8_0`), mentre il resto viene compresso con **Rotazione Hadamard (`H·D·x`)** in **Bonsai 1-Bit (`1.125 bpw`)** (`-46 %` VRAM).
- **Pianificatore KV Asimmetrico Multi-Agente:** Es. `1× Utente (1M)` + `1× Orchestratore (1M)` + `14× Agenti (256K)` con KV `Q8_0` e Disk-Parking NVMe Strata.
- **Abliterazione Reale (`RSI-MF-03` & `RSI-MF-06`):** Ortogonalizzazione normativa in streaming diretto su GGUF (`F32/F16/Q8_0/Q4_0`, `< 16 MB` RAM, protezione atomica).
- **Multi-LoRA Stacker & Fusione Statica (`RSI-MF-07`):** Fusione statica con `0 ms` di overhead e Spectral-Drift Guard.
- **Bridge GPU Apple Metal (`RSI-MF-05`):** **`28,7×` più veloce / `6,66 GWeights/s`** (`0` errori su M4 Max).
- **NALA-RAG (`:18700`) & NALA-MCP-cORe (28 Tool):** Generatore dataset 5-Zone Zero-PII (`RSI-MF-08`) e server MCP JSON-RPC 2.0.

## Avvio Rapido (1-2-3)
1. **macOS:** Apri `.dmg` → trascina `NALA-ModelForge.app` in `Applicazioni`.
2. **Ubuntu:** `sudo dpkg -i nala-modelforge-rs_1.5.0_amd64.deb && nala-modelforge-rs --open`
3. **Windows 11:** Estrai `.zip` → avvia `NALA-ModelForge.exe`.
4. **Collega MCP-cORe:** `nala-modelforge-rs --register-mcp`
