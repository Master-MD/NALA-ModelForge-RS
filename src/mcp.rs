#![forbid(unsafe_code)]

use crate::jobs::{JobManager, build_multi_ide_prompt, dream_rsi_matrix, probe_nala_grid, probe_nala_rag};
use crate::json::{Json, parse_json};
use crate::model_parser::{builtin_presets, inspect_local_model, preset_by_id};
use crate::planner::{
    ForgePlanRequest, UserSlotGroup, builtin_hardware_profiles, compute_forge_plan,
    detect_local_hardware, hardware_by_id,
};
use crate::quant_math::{quant_benchmark_json, run_lora_stack_merge, run_norm_preserving_abliteration};
use crate::security::{APP_VERSION, emit_cyclonedx_sbom, sha256_bist};
use std::io::{self, BufRead, Write};

/// Run 16-Tool JSON-RPC 2.0 MCP Server over stdio (`--mcp-stdio`) for NALA-MCP-cORe, Antigravity, ZCode, Codex, OpenCode.
pub fn run_mcp_stdio(jobs: JobManager) {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines().map_while(Result::ok) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let req = match parse_json(trimmed) {
            Ok(v) => v,
            Err(e) => {
                let err = Json::obj(vec![
                    ("jsonrpc", Json::s("2.0")),
                    ("id", Json::Null),
                    (
                        "error",
                        Json::obj(vec![("code", Json::i(-32700)), ("message", Json::s(e))]),
                    ),
                ]);
                let _ = writeln!(out, "{}", err.to_string_compact());
                let _ = out.flush();
                continue;
            }
        };

        let id = req.get("id").cloned().unwrap_or(Json::Null);
        let method = req.get("method").and_then(|v| v.as_str()).unwrap_or("");
        if method == "notifications/initialized" || method == "notifications/cancelled" {
            continue;
        }

        let response = match method {
            "initialize" => Json::obj(vec![
                ("jsonrpc", Json::s("2.0")),
                ("id", id),
                (
                    "result",
                    Json::obj(vec![
                        ("protocolVersion", Json::s("2024-11-05")),
                        (
                            "capabilities",
                            Json::obj(vec![("tools", Json::obj(vec![]))]),
                        ),
                        (
                            "serverInfo",
                            Json::obj(vec![
                                ("name", Json::s("nala-modelforge-rs")),
                                ("version", Json::s(APP_VERSION)),
                            ]),
                        ),
                    ]),
                ),
            ]),
            "tools/list" => Json::obj(vec![
                ("jsonrpc", Json::s("2.0")),
                ("id", id),
                ("result", Json::obj(vec![("tools", mcp_tools_schema())])),
            ]),
            "tools/call" => {
                let params = req.get("params").cloned().unwrap_or(Json::Null);
                let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let args = params.get("arguments").cloned().unwrap_or(Json::obj(vec![]));
                match dispatch_tool(&jobs, name, &args) {
                    Ok(payload) => Json::obj(vec![
                        ("jsonrpc", Json::s("2.0")),
                        ("id", id),
                        (
                            "result",
                            Json::obj(vec![(
                                "content",
                                Json::Arr(vec![Json::obj(vec![
                                    ("type", Json::s("text")),
                                    ("text", Json::s(payload.to_string_compact())),
                                ])]),
                            )]),
                        ),
                    ]),
                    Err(err_msg) => Json::obj(vec![
                        ("jsonrpc", Json::s("2.0")),
                        ("id", id),
                        (
                            "result",
                            Json::obj(vec![
                                ("isError", Json::b(true)),
                                (
                                    "content",
                                    Json::Arr(vec![Json::obj(vec![
                                        ("type", Json::s("text")),
                                        ("text", Json::s(err_msg)),
                                    ])]),
                                ),
                            ]),
                        ),
                    ]),
                }
            }
            _ => Json::obj(vec![
                ("jsonrpc", Json::s("2.0")),
                ("id", id),
                (
                    "error",
                    Json::obj(vec![
                        ("code", Json::i(-32601)),
                        ("message", Json::s(format!("Method not found: {method}"))),
                    ]),
                ),
            ]),
        };

        let _ = writeln!(out, "{}", response.to_string_compact());
        let _ = out.flush();
    }
}

fn tool_def(name: &str, desc: &str, props: Vec<(&str, &str)>) -> Json {
    let prop_entries: Vec<(&str, Json)> = props
        .into_iter()
        .map(|(k, t)| (k, Json::obj(vec![("type", Json::s(t))])))
        .collect();
    Json::obj(vec![
        ("name", Json::s(name)),
        ("description", Json::s(desc)),
        (
            "inputSchema",
            Json::obj(vec![
                ("type", Json::s("object")),
                ("properties", Json::obj(prop_entries)),
            ]),
        ),
    ])
}

pub fn mcp_tools_schema() -> Json {
    Json::Arr(vec![
        tool_def("modelforge_health", "Check NALA-ModelForge-RS engine health, version and FIPS SHA-256 BIST.", vec![]),
        tool_def("modelforge_detect_hardware", "Detect local CPU, RAM, GPU accelerator and installed llama.cpp/MLX/Strata tools.", vec![]),
        tool_def("modelforge_list_presets", "List built-in Frontier LLM architecture presets (1.5TB MoE, Qwen3.8-Flash-Next, DeepSeek-V4.1, Coder390, Bonsai-8B).", vec![]),
        tool_def("modelforge_list_hardware_profiles", "List hardware profiles (Mac Studio 512GB, M4 Max 64GB, Dual Blackwell 48GB, Strix Halo 128GB, NALA-Grid 268GB).", vec![]),
        tool_def("modelforge_inspect_model", "Zero-copy inspect any local .gguf, .safetensors, HuggingFace dir, or preset:// model.", vec![("path", "string")]),
        tool_def("modelforge_compute_plan", "Solve domain-selective mixed-precision quantization (Coding FAT + Rest 1-Bit Bonsai) & asymmetric multi-user KV-cache budget.", vec![("model_id", "string"), ("hardware_id", "string"), ("target_format", "string"), ("coding_share_pct", "number"), ("secondary_quant", "string"), ("moe_prune_pct", "number")]),
        tool_def("modelforge_quant_benchmark", "Run live pure-Rust quantization benchmark (Q8_0 vs Q4_0 vs BitNet 1.58 vs Bonsai 1-Bit vs Gyro-Hadamard 1-Bit).", vec![("elements", "number")]),
        tool_def("modelforge_abliteration_verify", "Run norm-preserving biprojected refusal-direction orthogonalization kernel and verify coding-subspace preservation.", vec![("alpha", "number"), ("protect_coding", "boolean")]),
        tool_def("modelforge_lora_merge_verify", "Verify multi-LoRA static weight merge W + (alpha/r)*BA spectral drift.", vec![("lora_count", "number"), ("rank", "number")]),
        tool_def("modelforge_export_recipe_bundle", "Export complete SHA-256 signed surgical tensor_types.txt, build scripts, and multi-user server configs.", vec![("model_id", "string"), ("hardware_id", "string"), ("output_dir", "string")]),
        tool_def("modelforge_probe_nala_rag", "Probe local NALA-RAG Evidence-Lock daemon on 127.0.0.1:18700.", vec![]),
        tool_def("modelforge_probe_nala_grid", "Probe localhost accelerator and opt-in NALA-Grid cluster nodes.", vec![]),
        tool_def("modelforge_list_jobs", "List recent ModelForge export, quantization, and kernel verification jobs.", vec![]),
        tool_def("modelforge_dream_rsi_matrix", "Return 3-7 round Triple-Council Dream/REA/RSI evolution proposals & delta matrix.", vec![]),
        tool_def("modelforge_dream_prompt", "Generate Multi-IDE evolution prompt (Codex/Antigravity/ZCode/OpenCode) with mandatory TheStandart footer.", vec![("proposal_id", "string"), ("ide", "string")]),
        tool_def("modelforge_dream_execute", "Execute a 2-click confirmed Dream/REA/RSI proposal blueprint & SHA-256 package.", vec![("proposal_id", "string"), ("ide", "string"), ("confirmed_two_click", "boolean"), ("output_dir", "string")]),
        tool_def("modelforge_deslop_audit", "Run native Pure-Rust 5/5 Golden-DOM Anti-AI-Slop audit (zero Python).", vec![("path", "string")]),
        tool_def("modelforge_sync_skills", "Run native Pure-Rust Cross-IDE Skill-Bridge across Antigravity, Codex/ZCode & OpenCode (zero Python).", vec![]),
        tool_def("modelforge_rsi_mf01_bonsai_gpu", "RSI-MF-01: Run parallel SIMD + Metal/Vulkan shader verified 1-Bit Bonsai g128 + Hadamard kernel benchmark.", vec![("blocks", "number")]),
        tool_def("modelforge_rsi_mf02_expert_profiler", "RSI-MF-02: Read-only Zero-PII code repo scan + Gyro-Coder MoE expert-activation profiler (-48% VRAM).", vec![("repo_path", "string"), ("total_experts", "number"), ("keep_pct", "number")]),
        tool_def("modelforge_rsi_mf03_stream_abliterate", "RSI-MF-03: Streaming binary GGUF norm-preserving abliteration surgery (<16MB RAM, atomic source protection).", vec![("src_path", "string"), ("output_dir", "string"), ("alpha", "number"), ("apply_write", "boolean"), ("confirmed_two_click", "boolean")]),
        tool_def("modelforge_rsi_mf04_grid_4d", "RSI-MF-04: Roofline-weighted NALA-Grid 4D Imatrix & LoRA orchestrator with RFC 2104 HMAC-SHA256 seal.", vec![("model_id", "string"), ("total_chunks", "number"), ("lora_steps", "number")]),
        tool_def("modelforge_rsi_mf05_metal_gpu", "RSI-MF-05: Direct Apple Silicon Metal GPU Zero-Copy Shared Memory 1-Bit Bonsai & Hadamard compute dispatch (+12x).", vec![("blocks", "number")]),
        tool_def("modelforge_rsi_mf06_quant_abliterate", "RSI-MF-06: Streaming Q8_0/Q4_0/F16/F32 Dequant-Abliterate-Requant GGUF surgery (<16MB RAM, atomic source guard).", vec![("src_path", "string"), ("output_dir", "string"), ("alpha", "number"), ("apply_write", "boolean"), ("confirmed_two_click", "boolean")]),
        tool_def("modelforge_rsi_mf07_lora_static_merge", "RSI-MF-07: Live GGUF Multi-LoRA Static Weight Merger (0ms inference overhead, Frobenius drift guard).", vec![("src_path", "string"), ("output_dir", "string"), ("apply_write", "boolean"), ("confirmed_two_click", "boolean")]),
        tool_def("modelforge_rsi_mf08_rag_dataset", "RSI-MF-08: Direct NALA-RAG (:18700) 5-Zone Corpus-to-Imatrix & Coding-LoRA JSONL generator with Zero-PII redaction.", vec![("zone_id", "string"), ("fallback_repo", "string"), ("output_dir", "string"), ("confirmed_two_click", "boolean")]),
        tool_def("modelforge_emit_sbom", "Emit CycloneDX 1.5 SBOM + SLSA v1.0 provenance JSON (zero external crates).", vec![]),
        tool_def("modelforge_self_test", "Run full internal self-test of all mathematical, parser, and sHeriFF security gates.", vec![]),
    ])
}

fn dispatch_tool(jobs: &JobManager, name: &str, args: &Json) -> Result<Json, String> {
    match name {
        "modelforge_health" => Ok(Json::obj(vec![
            ("status", Json::s("ok")),
            ("engine", Json::s("nala-modelforge-rs")),
            ("version", Json::s(APP_VERSION)),
            ("sha256_bist", Json::b(sha256_bist())),
            ("zero_python", Json::b(true)),
            ("unsafe_code", Json::s("forbidden")),
        ])),
        "modelforge_detect_hardware" => Ok(detect_local_hardware()),
        "modelforge_list_presets" => Ok(Json::obj(vec![(
            "presets",
            Json::Arr(builtin_presets().into_iter().map(|p| p.to_json()).collect()),
        )])),
        "modelforge_list_hardware_profiles" => Ok(Json::obj(vec![(
            "profiles",
            Json::Arr(builtin_hardware_profiles().into_iter().map(|h| h.to_json()).collect()),
        )])),
        "modelforge_inspect_model" => {
            let p = args.get("path").and_then(|v| v.as_str()).unwrap_or("preset://frontier-1.5tb-moe");
            inspect_local_model(p).map(|m| m.to_json())
        }
        "modelforge_compute_plan" => {
            let mid = args.get("model_id").and_then(|v| v.as_str()).unwrap_or("frontier-1.5tb-moe");
            let hid = args.get("hardware_id").and_then(|v| v.as_str()).unwrap_or("mac-studio-512gb");
            let model = preset_by_id(mid).or_else(|| inspect_local_model(mid).ok()).ok_or("Model not found")?;
            let req = ForgePlanRequest {
                model,
                hardware: hardware_by_id(hid),
                target_format: args.get("target_format").and_then(|v| v.as_str()).unwrap_or("MLX").into(),
                coding_logic_share_pct: args.get("coding_share_pct").and_then(|v| v.as_f64()).unwrap_or(35.0),
                secondary_quant: args.get("secondary_quant").and_then(|v| v.as_str()).unwrap_or("BONSAI_1BIT").into(),
                use_hadamard_rotation: true,
                moe_prune_pct: args.get("moe_prune_pct").and_then(|v| v.as_f64()).unwrap_or(25.0),
                coding_precision_override: "AUTO".into(),
                enable_abliteration: true,
                abliteration_alpha: 1.0,
                protect_coding_subspace: true,
                enable_strata_nvme_parking: true,
                lora_count: 2,
                lora_rank: 32,
                enable_nala_rag: true,
                rag_reserved_tokens: 32_768,
                slots: UserSlotGroup::default_mike_swarm(),
            };
            Ok(compute_forge_plan(&req))
        }
        "modelforge_quant_benchmark" => {
            let elems = args.get("elements").and_then(|v| v.as_u64()).unwrap_or(8192) as usize;
            Ok(quant_benchmark_json(elems))
        }
        "modelforge_abliteration_verify" => {
            let alpha = args.get("alpha").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let protect = args.get("protect_coding").and_then(|v| v.as_bool()).unwrap_or(true);
            Ok(run_norm_preserving_abliteration(64, alpha, protect, true))
        }
        "modelforge_lora_merge_verify" => {
            let cnt = args.get("lora_count").and_then(|v| v.as_u64()).unwrap_or(2).min(8) as usize;
            let rank = args.get("rank").and_then(|v| v.as_u64()).unwrap_or(32) as u32;
            let list: Vec<(String, u32, f64)> = (0..cnt)
                .map(|i| (format!("Adapter-{}", i + 1), rank, 0.8))
                .collect();
            Ok(run_lora_stack_merge(&list))
        }
        "modelforge_export_recipe_bundle" => {
            let mid = args.get("model_id").and_then(|v| v.as_str()).unwrap_or("frontier-1.5tb-moe");
            let hid = args.get("hardware_id").and_then(|v| v.as_str()).unwrap_or("mac-studio-512gb");
            let out_dir = args.get("output_dir").and_then(|v| v.as_str()).unwrap_or("./modelforge-export");
            let model = preset_by_id(mid).or_else(|| inspect_local_model(mid).ok()).ok_or("Model not found")?;
            let req = ForgePlanRequest {
                model,
                hardware: hardware_by_id(hid),
                target_format: "GGUF".into(),
                coding_logic_share_pct: 35.0,
                secondary_quant: "BONSAI_1BIT".into(),
                use_hadamard_rotation: true,
                moe_prune_pct: 25.0,
                coding_precision_override: "AUTO".into(),
                enable_abliteration: true,
                abliteration_alpha: 1.0,
                protect_coding_subspace: true,
                enable_strata_nvme_parking: true,
                lora_count: 2,
                lora_rank: 32,
                enable_nala_rag: true,
                rag_reserved_tokens: 32_768,
                slots: UserSlotGroup::default_mike_swarm(),
            };
            jobs.export_recipe_bundle(&req, out_dir, &["Coding-Systems-LoRA".into(), "EfficientThink-LoRA".into()])
        }
        "modelforge_probe_nala_rag" => Ok(probe_nala_rag()),
        "modelforge_probe_nala_grid" => Ok(probe_nala_grid()),
        "modelforge_list_jobs" => Ok(jobs.list_json()),
        "modelforge_dream_rsi_matrix" => Ok(dream_rsi_matrix()),
        "modelforge_dream_prompt" => {
            let pid = args.get("proposal_id").and_then(|v| v.as_str()).unwrap_or("RSI-MF-01");
            let ide = args.get("ide").and_then(|v| v.as_str()).unwrap_or("Antigravity");
            build_multi_ide_prompt(pid, ide).map(|p| Json::obj(vec![("proposal_id", Json::s(pid)), ("ide", Json::s(ide)), ("prompt", Json::s(p))]))
        }
        "modelforge_dream_execute" => {
            if !args.get("confirmed_two_click").and_then(|v| v.as_bool()).unwrap_or(false) {
                return Err("2-Klick Gate: 'confirmed_two_click: true' required".into());
            }
            let pid = args.get("proposal_id").and_then(|v| v.as_str()).unwrap_or("RSI-MF-01");
            let ide = args.get("ide").and_then(|v| v.as_str()).unwrap_or("Antigravity");
            let out_dir = args.get("output_dir").and_then(|v| v.as_str()).unwrap_or("./modelforge-export");
            jobs.execute_dream_proposal(pid, ide, out_dir)
        }
        "modelforge_deslop_audit" => {
            let p = args.get("path").and_then(|v| v.as_str()).filter(|s| !s.trim().is_empty());
            crate::ops::run_golden_dom_deslop_audit(p)
        }
        "modelforge_sync_skills" => crate::ops::sync_cross_ide_skills(),
        "modelforge_rsi_mf01_bonsai_gpu" => {
            let blocks = args.get("blocks").and_then(|v| v.as_u64()).unwrap_or(4096) as usize;
            Ok(crate::rsi_wave1::run_rsi_mf01_bonsai_gpu_kernel(blocks))
        }
        "modelforge_rsi_mf02_expert_profiler" => {
            let dir = args.get("repo_path").and_then(|v| v.as_str()).unwrap_or(".");
            let experts = args.get("total_experts").and_then(|v| v.as_u64()).unwrap_or(512) as u32;
            let keep = args.get("keep_pct").and_then(|v| v.as_f64()).unwrap_or(50.0);
            Ok(crate::rsi_wave1::run_rsi_mf02_expert_profiler(dir, experts, keep))
        }
        "modelforge_rsi_mf03_stream_abliterate" => {
            let src = args.get("src_path").and_then(|v| v.as_str()).unwrap_or("");
            let out_dir = args.get("output_dir").and_then(|v| v.as_str()).unwrap_or("./modelforge-export");
            let alpha = args.get("alpha").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let apply = args.get("apply_write").and_then(|v| v.as_bool()).unwrap_or(false);
            let confirmed = args.get("confirmed_two_click").and_then(|v| v.as_bool()).unwrap_or(false);
            if apply && !confirmed {
                return Err("2-Klick Gate: 'confirmed_two_click: true' required for apply_write".into());
            }
            crate::rsi_wave1::run_rsi_mf03_stream_gguf_abliteration(src, out_dir, alpha, true, apply)
        }
        "modelforge_rsi_mf04_grid_4d" => {
            let mid = args.get("model_id").and_then(|v| v.as_str()).unwrap_or("frontier-1.5tb-moe");
            let chunks = args.get("total_chunks").and_then(|v| v.as_u64()).unwrap_or(128) as u32;
            let steps = args.get("lora_steps").and_then(|v| v.as_u64()).unwrap_or(400) as u32;
            Ok(crate::rsi_wave1::run_rsi_mf04_grid_4d_orchestrator(mid, chunks, steps))
        }
        "modelforge_rsi_mf05_metal_gpu" => {
            let blocks = args.get("blocks").and_then(|v| v.as_u64()).unwrap_or(131_072) as usize;
            Ok(crate::rsi_wave2::run_rsi_mf05_metal_gpu_bridge(blocks))
        }
        "modelforge_rsi_mf06_quant_abliterate" => {
            let src = args.get("src_path").and_then(|v| v.as_str()).unwrap_or("");
            let out_dir = args.get("output_dir").and_then(|v| v.as_str()).unwrap_or("./modelforge-export");
            let alpha = args.get("alpha").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let apply = args.get("apply_write").and_then(|v| v.as_bool()).unwrap_or(false);
            let confirmed = args.get("confirmed_two_click").and_then(|v| v.as_bool()).unwrap_or(false);
            if apply && !confirmed {
                return Err("2-Klick Gate: 'confirmed_two_click: true' required for apply_write".into());
            }
            crate::rsi_wave2::run_rsi_mf06_quantized_stream_abliteration(src, out_dir, alpha, true, apply)
        }
        "modelforge_rsi_mf07_lora_static_merge" => {
            let src = args.get("src_path").and_then(|v| v.as_str()).unwrap_or("");
            let out_dir = args.get("output_dir").and_then(|v| v.as_str()).unwrap_or("./modelforge-export");
            let apply = args.get("apply_write").and_then(|v| v.as_bool()).unwrap_or(false);
            let confirmed = args.get("confirmed_two_click").and_then(|v| v.as_bool()).unwrap_or(false);
            if apply && !confirmed {
                return Err("2-Klick Gate: 'confirmed_two_click: true' required for apply_write".into());
            }
            crate::rsi_wave2::run_rsi_mf07_gguf_lora_static_merger(
                src,
                out_dir,
                &[("Rust-Systems-LoRA".into(), 16, 0.85), ("EfficientThink-LoRA".into(), 8, 0.5)],
                12.0,
                apply,
            )
        }
        "modelforge_rsi_mf08_rag_dataset" => {
            if !args.get("confirmed_two_click").and_then(|v| v.as_bool()).unwrap_or(false) {
                return Err("2-Klick Gate: 'confirmed_two_click: true' required".into());
            }
            let zone = args.get("zone_id").and_then(|v| v.as_str()).unwrap_or("zone_engineering");
            let repo = args.get("fallback_repo").and_then(|v| v.as_str()).unwrap_or(".");
            let out_dir = args.get("output_dir").and_then(|v| v.as_str()).unwrap_or("./modelforge-rag-dataset");
            crate::rsi_wave2::run_rsi_mf08_nala_rag_dataset_generator(zone, repo, out_dir)
        }
        "modelforge_emit_sbom" => Ok(emit_cyclonedx_sbom()),
        "modelforge_self_test" => Ok(crate::run_internal_self_test()),
        other => Err(format!("Unknown MCP tool: {other}")),
    }
}
