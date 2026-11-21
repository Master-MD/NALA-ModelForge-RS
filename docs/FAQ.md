# NALA-ModelForge-RS (`v1.5.0`) — 5-Language FAQ (`DE · EN · FR · IT · RM`)

---

## 🇩🇪 Deutsch — Häufige Fragen (FAQ)

1. **Ist wirklich 0 % Python in NALA-ModelForge-RS enthalten?**  
   **Ja — 100,0 % Pure Safe Rust (`#![forbid(unsafe_code)]`, 0 externe Crates) + Native Apple Swift/AppKit/Metal.** Keine `pip`, keine `venv`, keine `.py`-Dateien im gesamten Projekt.
2. **Kopiert Drag & Drop mein 1,5 TB Modell in den RAM?**  
   **Nein!** Native macOS AppKit `NSDraggingDestination` übergibt nur den Dateipfad. Rust liest via `BufReader + Seek` ausschliesslich den binären GGUF-/Safetensors-Header (`< 25 ms`, `< 16 MB` RAM).
3. **Wie füge ich LoRAs hinzu?**  
   Im Tab **`4. LoRA Stacker · NALA-RAG · Grid`** auf **`+ LoRA anhängen`** oder **`LoRA-Datei wählen`** klicken (`.gguf` oder `.safetensors`), Rank & Scale α einstellen und entweder per **`RSI-MF-07`** statisch mit `0 ms` Inferenz-Overhead direkt ins GGUF einschmelzen oder als `llama-server --lora` / `mlx_lm.fuse` exportieren.
4. **Wie funktioniert die Verbindung zu `NALA-MCP-cORe` und `NALA-RAG`?**  
   Ein Befehl `nala-modelforge-rs --register-mcp` registriert alle **28 MCP-Tools** in `NALA-MCP-cORe`, Antigravity, Gemini CLI & ZCode. Im Tab 4 prüft **`MCP-cORe Live-Check`** alle Einträge und **`NALA-RAG Live-Probe / Query`** verbindet direkt zu `http://127.0.0.1:18700` (inkl. 5-Zonen Tenant-Firewall in `RSI-MF-08`).
5. **Überschreibt ModelForge jemals mein Original-Modell?**  
   **Niemals!** Alle Schreib- und Abliteration-Module (`RSI-MF-03`, `RSI-MF-06`, `RSI-MF-07`) erzwingen serverseitig das **2-Klick Sicherheits-Gate** sowie atomares Schreiben (`*.tmp` → `rename`) in eine neue Zieldatei.

---

## 🇬🇧 English — Frequently Asked Questions

1. **Is there really 0.0% Python in NALA-ModelForge-RS?**  
   Yes: 100% Safe Rust (`#![forbid(unsafe_code)]`, 0 external crates) + Native macOS Swift/AppKit/Metal GPU Runner. Zero Python, zero Node.js runtime.
2. **Does Drag & Drop copy a 1.5 TB model into RAM?**  
   Never. Native AppKit `NSDraggingDestination` passes the zero-copy filesystem path; Rust streams only the GGUF/Safetensors header (`< 25 ms`, `< 16 MB` RAM).
3. **How do I attach or merge LoRAs?**  
   Open Tab **`4. LoRA Stacker · NALA-RAG · Grid`**, pick `.gguf`/`.safetensors` adapters, tune Rank & α, verify Frobenius Spectral Drift, and merge statically via **`RSI-MF-07`** (`0 ms` runtime overhead) or export `llama-server --lora` / `mlx_lm.fuse`.
4. **How do `NALA-MCP-cORe` & `NALA-RAG` connect?**  
   Run `nala-modelforge-rs --register-mcp` for instant **28-tool** wiring across `NALA-MCP-cORe`, Antigravity, Gemini CLI & ZCode. Use Tab 4 to live-probe `NALA-RAG (:18700)` and export 5-Zone Zero-PII datasets (`RSI-MF-08`).

---

## 🇫🇷 Français — Questions Fréquentes

1. **Y a-t-il vraiment 0,0 % de Python ?** Oui, 100 % Rust sûr (`#![forbid(unsafe_code)]`) + Swift/Metal natif sur macOS.
2. **Comment ajouter des LoRAs ?** Onglet **4 (LoRA Stacker)** : sélectionnez vos fichiers `.gguf`/`.safetensors`, réglez le rang et α, puis fusionnez statiquement via **`RSI-MF-07`** (`0 ms` de latence).
3. **Comment connecter NALA-MCP-cORe & NALA-RAG ?** Lancez `nala-modelforge-rs --register-mcp` (28 outils MCP) et testez `NALA-RAG (:18700)` en direct dans l'onglet 4.

---

## 🇮🇹 Italiano — Domande Frequenti

1. **Davvero 0,0 % Python?** Sì, 100 % Rust sicuro (`#![forbid(unsafe_code)]`) + Swift/Metal nativo su macOS.
2. **Come aggiungo i LoRA?** Scheda **4 (LoRA Stacker)**: aggiungi file `.gguf`/`.safetensors`, imposta Rank/α e fondi staticamente con **`RSI-MF-07`** (`0 ms` overhead).
3. **Come collego NALA-MCP-cORe & NALA-RAG?** Esegui `nala-modelforge-rs --register-mcp` (28 tool MCP) e verifica `NALA-RAG (:18700)` nella scheda 4.

---

## 🇨🇭 Rumantsch — Dumondas Frequentas

1. **Vairamain 0,0 % Python?** Gea, 100 % Rust segir (`#![forbid(unsafe_code)]`) + Swift/Metal nativ sin macOS.
2. **Co agiuntar LoRAs?** Tab **4 (LoRA Stacker)**: tscherner datotecas `.gguf`/`.safetensors`, reglar Rank/α e fusiunar staticamain cun **`RSI-MF-07`**.
3. **Co colliar NALA-MCP-cORe & NALA-RAG?** Exequir `nala-modelforge-rs --register-mcp` (28 utensils MCP) e testar `NALA-RAG (:18700)` en Tab 4.
