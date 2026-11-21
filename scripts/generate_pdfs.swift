import Cocoa
import CoreGraphics

// NALA-ModelForge-RS — 100% Native Swift CoreGraphics A4 PDF Guide Generator (0.0% Python!)
// Renders crisp, dark-obsidian branded PDF guides in all 5 languages (DE, EN, FR, IT, RM)
// embedding the official NALA-ModelForge app icon.

let rootDir = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "."
let version = (try? String(contentsOfFile: "\(rootDir)/VERSION", encoding: .utf8))?
    .trimmingCharacters(in: .whitespacesAndNewlines) ?? "1.5.0"
let iconImg = NSImage(contentsOfFile: "\(rootDir)/assets/icon_1024.png")

struct PdfSpec {
    let lang: String
    let subtitle: String
    let sections: [(String, String)]
}

let guides: [PdfSpec] = [
    PdfSpec(
        lang: "DE",
        subtitle: "100% Native Safe Rust + Apple Metal GPU · 0.0% Python · Tri-OS Handbuch (v\(version))",
        sections: [
            ("1. Was ist NALA-ModelForge-RS?",
             "NALA-ModelForge-RS ist ein kompromissloses, 100 % Python-freies Rust-Studio fuer macOS (Apple Silicon & Intel), Ubuntu/Debian (x64 & ARM64) und Windows 11. Ziehe lokale .gguf-, .safetensors- oder HuggingFace-Modelle (bis 1,5 TB MoE) direkt ins Fenster: Zero-Copy Header-Analyse in <25 ms ohne RAM-Kopie."),
            ("2. Domain-Selective Mixed-Precision & MoE Pruning (Bonsai 1-Bit + Gyro-Coder)",
             "Halte Coding, Logik, Attention (attn_q/k/v/o), Embeddings und Router-Gates (FP16) auf maximaler Praezision (BF16/Q8_0/Q6_K), waehrend sekundaere Gewichte mit Randomized Walsh-Hadamard Rotation (H*D*x) auf Bonsai 1-Bit (1.125 bpw) oder BitNet 1.58-Bit komprimiert bzw. per RSI-MF-02 strukturell geprunt werden (-46% VRAM)."),
            ("3. Asymmetrischer Multi-User & Agenten KV-Cache Planer",
             "Plane heterogene Rollen-Fenster wie 1x Hauptuser (1M Tokens) + 1x Orchestrator (1M Tokens) + 14x Spezialisten-Agenten (je 256K Tokens) mit Q8_0/Q4_0 KV-Cache, Radix Prefix-Sharing, YaRN RoPE Scaling und NALA-Strata-RS NVMe Disk-Parking."),
            ("4. Echtes Uncensoring (RSI-MF-03 & RSI-MF-06) + Multi-LoRA Merger (RSI-MF-07)",
             "Norm-Preserving Biprojected Abliteration W' = W - alpha*(W*r_orth)*r_orth^T direkt im 4-MB Streaming-GGUF (F32/F16/Q8_0/Q4_0, >98% Refusal Suppression, >0.995 Coding Cosine). LoRAs per Tab 4 anhaengen und mit 0 ms Inferenz-Overhead statisch einschmelzen (Frobenius Drift-Guard + atomarer Backup-Guard)."),
            ("5. Apple Metal GPU (RSI-MF-05), NALA-RAG (:18700) & NALA-MCP-cORe (28 Tools)",
             "Nativer Apple M4 Max Metal GPU Runner erreicht 28.7x Speedup / 6.66 GWeights/s (0 Bitfehler). Ein Befehl 'nala-modelforge-rs --register-mcp' verbindet alle 28 MCP-Tools mit NALA-MCP-cORe, Antigravity, Gemini CLI & ZCode. RSI-MF-08 exportiert 5-Zonen Zero-PII Imatrix/LoRA Datasets direkt aus NALA-RAG."),
            ("6. Caveman 1-2-3 Installation",
             "macOS: .dmg oeffnen -> NALA-ModelForge.app nach Programme ziehen.\nUbuntu/Proxmox: sudo dpkg -i nala-modelforge-rs_\(version)_amd64.deb && nala-modelforge-rs --open\nWindows 11: .zip entpacken -> NALA-ModelForge.exe starten.")
        ]
    ),
    PdfSpec(
        lang: "EN",
        subtitle: "100% Native Safe Rust + Apple Metal GPU · 0.0% Python · Tri-OS Guide (v\(version))",
        sections: [
            ("1. Overview",
             "NALA-ModelForge-RS is a 100% zero-Python Safe Rust & native OS precision studio for macOS (Apple Silicon & Intel), Ubuntu/Debian (x64 & ARM64), and Windows 11. Drop local .gguf, .safetensors, or HuggingFace folders up to 1.5 TB for <25 ms zero-copy header inspection."),
            ("2. Domain-Selective Mixed-Precision & MoE Pruning",
             "Preserve Coding, Logic, Attention, Embeddings, and FP16 Router Gates at BF16/Q8_0/Q6_K while compressing secondary weights with Randomized Walsh-Hadamard rotation to Bonsai 1-Bit (1.125 bpw) or BitNet 1.58-Bit, or pruning non-coding MoE experts via RSI-MF-02 (-46% VRAM)."),
            ("3. Asymmetric Multi-User & Multi-Agent KV Planner",
             "Reserve heterogeneous context budgets (e.g., 1x Human 1M + 1x Orchestrator 1M + 14x Agents 256K) with Q8_0/Q4_0 KV, Radix Prefix Sharing, YaRN RoPE scaling, and NALA-Strata-RS NVMe Disk-Parking."),
            ("4. True Uncensoring (RSI-MF-03/06) & Static Multi-LoRA Merger (RSI-MF-07)",
             "Stream norm-preserving biprojected abliteration across F32/F16/Q8_0/Q4_0 GGUF tensors (<16 MB RAM, atomic source protection) and statically merge stacked LoRA adapters with 0 ms runtime overhead and Frobenius drift guards."),
            ("5. Apple Metal GPU (RSI-MF-05), NALA-RAG (:18700) & NALA-MCP-cORe (28 Tools)",
             "Native Apple Silicon Metal compute bridge delivers 28.7x speedup / 6.66 GWeights/s (0 bit mismatches). Run 'nala-modelforge-rs --register-mcp' to wire all 28 MCP tools into NALA-MCP-cORe, Antigravity, Gemini CLI & ZCode."),
            ("6. Caveman 1-2-3 Install",
             "macOS: Open .dmg -> drag NALA-ModelForge.app to Applications.\nUbuntu/Proxmox: sudo dpkg -i nala-modelforge-rs_\(version)_amd64.deb && nala-modelforge-rs --open\nWindows 11: Unzip -> launch NALA-ModelForge.exe.")
        ]
    ),
    PdfSpec(
        lang: "FR",
        subtitle: "100% Rust Natif Sur + GPU Apple Metal · 0.0% Python · Guide Tri-OS (v\(version))",
        sections: [
            ("1. Presentation",
             "NALA-ModelForge-RS est un studio 100% Rust natif sans Python pour macOS (Apple Silicon & Intel), Ubuntu/Debian (x64 & ARM64) et Windows 11. Glissez vos modeles .gguf ou .safetensors (jusqu'a 1,5 To) pour une inspection Zero-Copy en <25 ms."),
            ("2. Quantification Mixte Selective & Elagage MoE",
             "Conservez Code, Logique, Attention et Routeurs (FP16) en BF16/Q8_0 tout en compressant le reste avec Rotation de Hadamard en Bonsai 1-Bit (1.125 bpw) ou par elagage d'experts MoE (-46% VRAM)."),
            ("3. Cache KV Asymetrique Multi-Agents",
             "Planifiez 1x Humain (1M) + 1x Orchestrateur (1M) + 14x Agents (256K) avec KV Q8_0, partage de prefixes Radix et parking NVMe Strata."),
            ("4. Decensure Reelle (RSI-MF-03/06) & Fusion Statique Multi-LoRA (RSI-MF-07)",
             "Ablitration orthogonale en streaming GGUF (F32/F16/Q8_0/Q4_0, <16 Mo RAM, protection atomique) et fusion statique multi-LoRA a 0 ms de latence."),
            ("5. GPU Apple Metal (RSI-MF-05), NALA-RAG (:18700) & NALA-MCP-cORe (28 Outils)",
             "Pont Metal GPU natif: 28.7x plus rapide / 6.66 GWeights/s (0 erreur). Lancez 'nala-modelforge-rs --register-mcp' pour enregistrer les 28 outils MCP et connecter NALA-RAG."),
            ("6. Installation Caveman 1-2-3",
             "macOS: .dmg -> Applications | Ubuntu: sudo dpkg -i nala-modelforge-rs_\(version)_amd64.deb | Windows 11: .zip -> NALA-ModelForge.exe.")
        ]
    ),
    PdfSpec(
        lang: "IT",
        subtitle: "100% Rust Nativo Sicuro + GPU Apple Metal · 0.0% Python · Guida Tri-OS (v\(version))",
        sections: [
            ("1. Panoramica",
             "NALA-ModelForge-RS e uno studio 100% Rust nativo senza Python per macOS (Apple Silicon & Intel), Ubuntu/Debian (x64 & ARM64) e Windows 11. Trascina modelli .gguf o .safetensors fino a 1,5 TB per un'ispezione Zero-Copy in <25 ms."),
            ("2. Quantizzazione Mista Selettiva & Pruning MoE",
             "Mantieni Codice, Logica, Attention e Router (FP16) in BF16/Q8_0 mentre comprimi il resto con Rotazione Hadamard in Bonsai 1-Bit (1.125 bpw) o tramite pruning esperti MoE (-46% VRAM)."),
            ("3. Pianificatore KV Asimmetrico Multi-Agente",
             "Pianifica 1x Utente (1M) + 1x Orchestratore (1M) + 14x Agenti (256K) con KV Q8_0, condivisione prefissi Radix e Strata NVMe Disk-Parking."),
            ("4. Abliterazione Reale (RSI-MF-03/06) & Fusione Statica Multi-LoRA (RSI-MF-07)",
             "Abliterazione ortogonale in streaming GGUF (F32/F16/Q8_0/Q4_0, <16 MB RAM, guardia atomica) e fusione statica multi-LoRA a 0 ms di overhead."),
            ("5. GPU Apple Metal (RSI-MF-05), NALA-RAG (:18700) & NALA-MCP-cORe (28 Tool)",
             "Bridge Metal nativo: 28.7x piu veloce / 6.66 GWeights/s (0 errori). Esegui 'nala-modelforge-rs --register-mcp' per collegare tutti i 28 tool MCP e NALA-RAG."),
            ("6. Installazione Caveman 1-2-3",
             "macOS: .dmg -> Applicazioni | Ubuntu: sudo dpkg -i nala-modelforge-rs_\(version)_amd64.deb | Windows 11: .zip -> NALA-ModelForge.exe.")
        ]
    ),
    PdfSpec(
        lang: "RM",
        subtitle: "100% Rust Nativ Segir + GPU Apple Metal · 0.0% Python · Guida Tri-OS (v\(version))",
        sections: [
            ("1. Survista",
             "NALA-ModelForge-RS ei in studio 100% Rust nativ senza Python per macOS (Apple Silicon & Intel), Ubuntu/Debian (x64 & ARM64) e Windows 11. Tira models .gguf u .safetensors fin 1,5 TB per ina inspecziun Zero-Copy en <25 ms."),
            ("2. Quantisaziun Mista Selectiva & Pruning MoE",
             "Tegnair Coding, Logica, Attention e Router (FP16) en BF16/Q8_0 e comprimir il rest cun Rotaziun Hadamard sin Bonsai 1-Bit (1.125 bpw) u pruning d'experts MoE (-46% VRAM)."),
            ("3. Planisader KV Asimetric Multi-Agents",
             "Planisar 1x Utilisader (1M) + 1x Orchestratur (1M) + 14x Agents (256K) cun KV Q8_0, Radix Prefix-Sharing e Strata NVMe Disk-Parking."),
            ("4. Abliteraziun Reala (RSI-MF-03/06) & Fusiun Statica Multi-LoRA (RSI-MF-07)",
             "Abliteraziun ortogonala en streaming GGUF (F32/F16/Q8_0/Q4_0, <16 MB RAM, protecziun atomara) e fusiun statica multi-LoRA cun 0 ms overhead."),
            ("5. GPU Apple Metal (RSI-MF-05), NALA-RAG (:18700) & NALA-MCP-cORe (28 Utensils)",
             "Punt Metal nativa: 28.7x pli svelt / 6.66 GWeights/s (0 errurs). Exequir 'nala-modelforge-rs --register-mcp' per colliar ils 28 utensils MCP e NALA-RAG."),
            ("6. Installaziun Caveman 1-2-3",
             "macOS: .dmg -> Applicaziuns | Ubuntu: sudo dpkg -i nala-modelforge-rs_\(version)_amd64.deb | Windows 11: .zip -> NALA-ModelForge.exe.")
        ]
    )
]

let pageWidth: CGFloat = 595.0  // A4 portrait
let pageHeight: CGFloat = 842.0
var mediaBox = CGRect(x: 0, y: 0, width: pageWidth, height: pageHeight)

for g in guides {
    let outPath = "\(rootDir)/docs/NALA-ModelForge-Guide-\(g.lang).pdf"
    let url = URL(fileURLWithPath: outPath) as CFURL
    guard let ctx = CGContext(url, mediaBox: &mediaBox, nil) else {
        fputs("Failed creating PDF context for \(g.lang)\n", stderr)
        exit(1)
    }

    ctx.beginPDFPage(nil)
    let nsCtx = NSGraphicsContext(cgContext: ctx, flipped: false)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = nsCtx

    // Header dark obsidian banner
    NSColor(calibratedRed: 0.04, green: 0.055, blue: 0.08, alpha: 1.0).setFill()
    NSBezierPath(rect: NSRect(x: 0, y: pageHeight - 135, width: pageWidth, height: 135)).fill()

    // Emerald accent bar
    NSColor(calibratedRed: 0.06, green: 0.73, blue: 0.51, alpha: 1.0).setFill()
    NSBezierPath(rect: NSRect(x: 0, y: pageHeight - 139, width: pageWidth, height: 4)).fill()

    if let icon = iconImg {
        icon.draw(in: NSRect(x: 34, y: pageHeight - 115, width: 88, height: 88))
    }

    let titleAttrs: [NSAttributedString.Key: Any] = [
        .font: NSFont.systemFont(ofSize: 22, weight: .bold),
        .foregroundColor: NSColor.white
    ]
    let subAttrs: [NSAttributedString.Key: Any] = [
        .font: NSFont.systemFont(ofSize: 10.5, weight: .medium),
        .foregroundColor: NSColor(calibratedRed: 0.65, green: 0.95, blue: 0.82, alpha: 1.0)
    ]
    NSString(string: "NALA-ModelForge-RS  [\(g.lang)]").draw(at: NSPoint(x: 138, y: pageHeight - 66), withAttributes: titleAttrs)
    NSString(string: g.subtitle).draw(at: NSPoint(x: 138, y: pageHeight - 92), withAttributes: subAttrs)

    let secTitleAttrs: [NSAttributedString.Key: Any] = [
        .font: NSFont.systemFont(ofSize: 12.5, weight: .bold),
        .foregroundColor: NSColor(calibratedRed: 0.03, green: 0.48, blue: 0.34, alpha: 1.0)
    ]
    let bodyPara = NSMutableParagraphStyle()
    bodyPara.lineSpacing = 3.0
    let bodyAttrs: [NSAttributedString.Key: Any] = [
        .font: NSFont.systemFont(ofSize: 10.2, weight: .regular),
        .foregroundColor: NSColor(calibratedRed: 0.12, green: 0.14, blue: 0.18, alpha: 1.0),
        .paragraphStyle: bodyPara
    ]

    var y = pageHeight - 170
    for (heading, body) in g.sections {
        NSString(string: heading).draw(at: NSPoint(x: 36, y: y), withAttributes: secTitleAttrs)
        y -= 16
        let rect = NSRect(x: 36, y: y - 78, width: pageWidth - 72, height: 78)
        NSString(string: body).draw(in: rect, withAttributes: bodyAttrs)
        y -= 92
    }

    // Footer
    let footAttrs: [NSAttributedString.Key: Any] = [
        .font: NSFont.monospacedSystemFont(ofSize: 7.4, weight: .regular),
        .foregroundColor: NSColor.gray
    ]
    NSString(string: "NALA-ModelForge-RS v\(version) · 100% Native Safe Rust + Metal · Zero-Python · Zero-PII · Apache-2.0 · https://github.com/Master-MD/NALA-ModelForge-RS")
        .draw(at: NSPoint(x: 36, y: 24), withAttributes: footAttrs)

    NSGraphicsContext.restoreGraphicsState()
    ctx.endPDFPage()
    ctx.closePDF()
    print("  ✓ PDF generated: \(outPath)")
}
