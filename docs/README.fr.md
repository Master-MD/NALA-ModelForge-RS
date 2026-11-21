# NALA-ModelForge-RS (`v1.5.0`) — Français 🇫🇷🇨🇭

**100 % Rust Natif Sûr + GPU Apple Metal Natif (0,0 % Python · 0 Crate Externe · `#![forbid(unsafe_code)]`)**  
Le studio de précision Tri-OS (**macOS ARM64/Intel**, **Ubuntu/Linux x64/ARM64**, **Windows 11**) pour quantifier, élaguer (MoE), ablitérer, fusionner des LoRAs et planifier le cache KV multi-agents de modèles jusqu'à 1,5 To.

## Fonctionnalités Clés
- **Inspecteur Drag & Drop Zero-Copy :** Glissez un `.gguf`, `.safetensors` ou dossier HuggingFace (`< 25 ms`, zéro copie RAM même sur 1,5 To).
- **Quantification Mixte Sélective & Élagage MoE :** Le cœur Code/Logique/Attention reste en haute précision (`BF16`/`Q8_0`), le reste est compressé avec **Rotation de Hadamard (`H·D·x`)** en **Bonsai 1-Bit (`1.125 bpw`)** ou élagué (`-46 %` VRAM).
- **Planificateur KV Asymétrique Multi-Agents :** Ex. `1× Humain (1M)` + `1× Orchestrateur (1M)` + `14× Agents (256K)` avec KV `Q8_0` et parking NVMe Strata.
- **Décensure Réelle (`RSI-MF-03` & `RSI-MF-06`) :** Ablitration orthogonale préservant la norme directement en streaming GGUF (`F32/F16/Q8_0/Q4_0`, `< 16 Mo` RAM, protection atomique).
- **Empileur & Fusion Statique Multi-LoRA (`RSI-MF-07`) :** Fusion statique (`0 ms` de latence) avec garde de dérive spectrale.
- **Pont GPU Apple Metal (`RSI-MF-05`) :** **`28,7×` plus rapide / `6,66 GWeights/s`** (`0` erreur binaire sur M4 Max).
- **NALA-RAG (`:18700`) & NALA-MCP-cORe (28 Outils) :** Générateur de datasets 5-Zones Zero-PII (`RSI-MF-08`) et serveur MCP JSON-RPC 2.0.

## Démarrage Rapide (1-2-3)
1. **macOS :** Ouvrir `.dmg` → glisser `NALA-ModelForge.app` dans `Applications`.
2. **Ubuntu :** `sudo dpkg -i nala-modelforge-rs_1.5.0_amd64.deb && nala-modelforge-rs --open`
3. **Windows 11 :** Dézipper `.zip` → lancer `NALA-ModelForge.exe`.
4. **Connecter MCP-cORe :** `nala-modelforge-rs --register-mcp`
