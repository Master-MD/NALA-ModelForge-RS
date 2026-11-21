# 🪨 NALA-ModelForge-RS (`v1.5.0`) — Caveman Install Guide (5 Languages)

> *No Python. No Docker. No pip. No headaches. 1 Binary. Drag. Click. Forge.*

---

## 🇩🇪 Deutsch — Caveman Kurzanleitung

### macOS (Apple Silicon M1–M4 & Intel)
1. Öffne `NALA-ModelForge-1.5.0-macOS-AppleSilicon-ARM64.dmg` *(oder `Intel-x86_64.dmg`)*.
2. Ziehe **`NALA-ModelForge.app`** in **`Programme`**.
3. Öffnen → `.gguf` / `.safetensors` / Modell-Ordner einfach **ins Fenster ziehen**!
4. Optional für MCP-cORe & Terminal:
   ```bash
   /Applications/NALA-ModelForge.app/Contents/MacOS/nala-modelforge-engine --register-mcp
   ```

### Ubuntu / Debian / Proxmox Linux (`amd64` & `arm64`)
```bash
sudo dpkg -i nala-modelforge-rs_1.5.0_amd64.deb
nala-modelforge-rs --self-test
nala-modelforge-rs --register-mcp
nala-modelforge-rs --open
```

### Windows 11 (`x86_64`)
1. Entpacke `NALA-ModelForge-1.5.0-windows-x86_64.zip`.
2. Starte `.\install.ps1` (oder direkt Doppelklick auf `NALA-ModelForge.exe`).
3. Für MCP-cORe: `.\NALA-ModelForge.exe --register-mcp`.

---

## 🇬🇧 English — Caveman Quickstart
- **macOS:** Open `.dmg` → drag `NALA-ModelForge.app` to `Applications` → drop any `.gguf` into window!
- **Ubuntu / Proxmox:** `sudo dpkg -i nala-modelforge-rs_1.5.0_amd64.deb && nala-modelforge-rs --open`
- **Windows 11:** Unzip → run `NALA-ModelForge.exe` (or `.\install.ps1`).
- **Connect 28 MCP Tools:** `nala-modelforge-rs --register-mcp`

---

## 🇫🇷 Français — Guide Rapide Caveman
- **macOS :** Ouvrir `.dmg` → glisser `NALA-ModelForge.app` dans `Applications` → déposer un `.gguf`.
- **Ubuntu / Proxmox :** `sudo dpkg -i nala-modelforge-rs_1.5.0_amd64.deb && nala-modelforge-rs --open`
- **Windows 11 :** Extraire `.zip` → lancer `NALA-ModelForge.exe`.

---

## 🇮🇹 Italiano — Guida Rapida Caveman
- **macOS:** Apri `.dmg` → trascina `NALA-ModelForge.app` in `Applicazioni` → trascina un `.gguf`.
- **Ubuntu / Proxmox:** `sudo dpkg -i nala-modelforge-rs_1.5.0_amd64.deb && nala-modelforge-rs --open`
- **Windows 11:** Estrai `.zip` → avvia `NALA-ModelForge.exe`.

---

## 🇨🇭 Rumantsch — Guida Svelta Caveman
- **macOS:** Avrir `.dmg` → tirar `NALA-ModelForge.app` en `Applicaziuns` → tirar in `.gguf`.
- **Ubuntu / Proxmox:** `sudo dpkg -i nala-modelforge-rs_1.5.0_amd64.deb && nala-modelforge-rs --open`
- **Windows 11:** Extrair `.zip` → aviar `NALA-ModelForge.exe`.
