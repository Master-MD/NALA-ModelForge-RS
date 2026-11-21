#![forbid(unsafe_code)]

use crate::jobs::{JobManager, build_multi_ide_prompt, dream_rsi_matrix, probe_nala_grid, probe_nala_rag};
use crate::json::{Json, parse_json};
use crate::model_parser::{builtin_presets, inspect_local_model, preset_by_id};
use crate::planner::{
    ForgePlanRequest, HardwareProfile, UserSlotGroup, builtin_hardware_profiles,
    compute_forge_plan, detect_local_hardware, hardware_by_id,
};
use crate::quant_math::{quant_benchmark_json, run_lora_stack_merge, run_norm_preserving_abliteration};
use crate::security::{APP_VERSION, MAX_HTTP_BODY_BYTES, emit_cyclonedx_sbom, sha256_bist};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

const INDEX_HTML: &str = include_str!("../ui/index.html");

pub fn run_http_server(port: u16, jobs: JobManager) -> Result<(), String> {
    // SEC-01: Strictly bind 127.0.0.1 only (never 0.0.0.0)
    let addr = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&addr)
        .map_err(|e| format!("Failed to bind http://{addr}: {e}"))?;
    eprintln!("[NALA-ModelForge-RS v{APP_VERSION}] Listening on http://{addr} (127.0.0.1 strict)");

    for incoming in listener.incoming() {
        if let Ok(stream) = incoming {
            let jm = jobs.clone();
            std::thread::spawn(move || {
                let _ = handle_client(stream, jm);
            });
        }
    }
    Ok(())
}

fn handle_client(mut stream: TcpStream, jobs: JobManager) -> Result<(), String> {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(15)));

    let mut raw = Vec::with_capacity(4096);
    let mut buf = [0u8; 4096];
    let header_end;
    loop {
        let n = stream.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            return Ok(());
        }
        raw.extend_from_slice(&buf[..n]);
        if let Some(pos) = find_subslice(&raw, b"\r\n\r\n") {
            header_end = pos + 4;
            break;
        }
        if raw.len() > 64 * 1024 {
            write_http(&mut stream, 431, "application/json", r#"{"error":"Headers too large"}"#)?;
            return Ok(());
        }
    }

    let head_str = String::from_utf8_lossy(&raw[..header_end]).into_owned();
    let mut lines = head_str.lines();
    let req_line = lines.next().unwrap_or("");
    let mut parts = req_line.split_whitespace();
    let method = parts.next().unwrap_or("GET");
    let full_path = parts.next().unwrap_or("/");
    let path = full_path.split('?').next().unwrap_or("/");

    let mut content_len = 0usize;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_len = v.trim().parse::<usize>().unwrap_or(0);
            }
        }
    }
    if content_len > MAX_HTTP_BODY_BYTES {
        write_http(&mut stream, 413, "application/json", r#"{"error":"SEC-04: Body exceeds 10 MiB limit"}"#)?;
        return Ok(());
    }

    let mut body = raw[header_end..].to_vec();
    while body.len() < content_len {
        let n = stream.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&buf[..n]);
        if body.len() > MAX_HTTP_BODY_BYTES {
            write_http(&mut stream, 413, "application/json", r#"{"error":"SEC-04: Body exceeds 10 MiB limit"}"#)?;
            return Ok(());
        }
    }
    let body_str = String::from_utf8_lossy(&body[..content_len.min(body.len())]).into_owned();

    match (method, path) {
        ("GET", "/") | ("GET", "/index.html") => {
            write_http(&mut stream, 200, "text/html; charset=utf-8", INDEX_HTML)?;
        }
        ("GET", "/api/health") => {
            let j = Json::obj(vec![
                ("status", Json::s("ok")),
                ("engine", Json::s("nala-modelforge-rs")),
                ("version", Json::s(APP_VERSION)),
                ("sha256_bist", Json::b(sha256_bist())),
                ("zero_python", Json::b(true)),
                ("unsafe_code", Json::s("forbidden")),
            ]);
            write_json(&mut stream, 200, &j)?;
        }
        ("GET", "/api/hardware") => {
            let j = Json::obj(vec![
                ("local", detect_local_hardware()),
                (
                    "profiles",
                    Json::Arr(builtin_hardware_profiles().into_iter().map(|h| h.to_json()).collect()),
                ),
            ]);
            write_json(&mut stream, 200, &j)?;
        }
        ("GET", "/api/presets") => {
            let j = Json::obj(vec![(
                "presets",
                Json::Arr(builtin_presets().into_iter().map(|p| p.to_json()).collect()),
            )]);
            write_json(&mut stream, 200, &j)?;
        }
        ("POST", "/api/inspect") => {
            match parse_json(&body_str)
                .and_then(|v| {
                    let p = v.get("path").and_then(|x| x.as_str()).ok_or("Missing 'path'")?;
                    inspect_local_model(p)
                })
            {
                Ok(model) => write_json(&mut stream, 200, &model.to_json())?,
                Err(e) => write_json(&mut stream, 400, &Json::obj(vec![("error", Json::s(e))]))?,
            }
        }
        ("POST", "/api/plan") => {
            match parse_plan_request(&body_str) {
                Ok(req) => write_json(&mut stream, 200, &compute_forge_plan(&req))?,
                Err(e) => write_json(&mut stream, 400, &Json::obj(vec![("error", Json::s(e))]))?,
            }
        }
        ("POST", "/api/quant-bench") => {
            let elems = parse_json(&body_str)
                .ok()
                .and_then(|v| v.get("elements").and_then(|x| x.as_u64()))
                .unwrap_or(8192) as usize;
            write_json(&mut stream, 200, &quant_benchmark_json(elems))?;
        }
        ("POST", "/api/abliterate-sim") => {
            let v = parse_json(&body_str).unwrap_or(Json::Null);
            let alpha = v.get("alpha").and_then(|x| x.as_f64()).unwrap_or(1.0);
            let protect = v.get("protect_coding").and_then(|x| x.as_bool()).unwrap_or(true);
            let norm_p = v.get("norm_preserving").and_then(|x| x.as_bool()).unwrap_or(true);
            write_json(&mut stream, 200, &run_norm_preserving_abliteration(64, alpha, protect, norm_p))?;
        }
        ("POST", "/api/lora-merge-sim") => {
            let v = parse_json(&body_str).unwrap_or(Json::Null);
            let mut adapters = Vec::new();
            if let Some(arr) = v.get("adapters").and_then(|x| x.as_arr()) {
                for a in arr.iter().take(12) {
                    let name = a.get("name").and_then(|x| x.as_str()).unwrap_or("LoRA").to_string();
                    let rank = a.get("rank").and_then(|x| x.as_u64()).unwrap_or(32) as u32;
                    let scale = a.get("scale").and_then(|x| x.as_f64()).unwrap_or(0.8);
                    adapters.push((name, rank, scale));
                }
            }
            if adapters.is_empty() {
                adapters.push(("Coding-Systems-LoRA".into(), 32, 0.85));
            }
            write_json(&mut stream, 200, &run_lora_stack_merge(&adapters))?;
        }
        ("POST", "/api/jobs/start") => {
            match handle_job_start(&jobs, &body_str) {
                Ok(res) => write_json(&mut stream, 200, &res)?,
                Err(e) => write_json(&mut stream, 400, &Json::obj(vec![("error", Json::s(e))]))?,
            }
        }
        ("GET", "/api/jobs/list") => write_json(&mut stream, 200, &jobs.list_json())?,
        ("GET", "/api/nala-rag/probe") => write_json(&mut stream, 200, &probe_nala_rag())?,
        ("GET", "/api/mcp/status") => write_json(&mut stream, 200, &crate::ops::check_mcp_core_status())?,
        ("POST", "/api/mcp/register") => {
            match crate::ops::register_mcp_and_app_registry() {
                Ok(res) => write_json(&mut stream, 200, &res)?,
                Err(e) => write_json(&mut stream, 400, &Json::obj(vec![("error", Json::s(e))]))?,
            }
        }
        ("GET", "/api/nala-grid/status") => write_json(&mut stream, 200, &probe_nala_grid())?,
        ("GET", "/api/dream/matrix") => write_json(&mut stream, 200, &dream_rsi_matrix())?,
        ("POST", "/api/dream/prompt") => {
            let v = parse_json(&body_str).unwrap_or(Json::Null);
            let pid = v.get("proposal_id").and_then(|x| x.as_str()).unwrap_or("RSI-MF-01");
            let ide = v.get("ide").and_then(|x| x.as_str()).unwrap_or("Antigravity");
            match build_multi_ide_prompt(pid, ide) {
                Ok(prompt) => write_json(&mut stream, 200, &Json::obj(vec![("proposal_id", Json::s(pid)), ("ide", Json::s(ide)), ("prompt", Json::s(prompt))]))?,
                Err(e) => write_json(&mut stream, 400, &Json::obj(vec![("error", Json::s(e))]))?,
            }
        }
        ("POST", "/api/dream/execute") => {
            let v = parse_json(&body_str).unwrap_or(Json::Null);
            if !v.get("confirmed_two_click").and_then(|x| x.as_bool()).unwrap_or(false) {
                write_json(&mut stream, 400, &Json::obj(vec![("error", Json::s("2-Klick Safety Gate: 'confirmed_two_click: true' erforderlich."))]))?;
            } else {
                let pid = v.get("proposal_id").and_then(|x| x.as_str()).unwrap_or("RSI-MF-01");
                let ide = v.get("ide").and_then(|x| x.as_str()).unwrap_or("Antigravity");
                let default_out = std::env::var("HOME")
                    .map(|h| format!("{h}/Documents/NALA-ModelForge-Exports"))
                    .unwrap_or_else(|_| "./NALA-ModelForge-Exports".into());
                let out_dir = v.get("output_dir").and_then(|x| x.as_str()).filter(|s| !s.trim().is_empty()).unwrap_or(&default_out);
                match jobs.execute_dream_proposal(pid, ide, out_dir) {
                    Ok(res) => write_json(&mut stream, 200, &res)?,
                    Err(e) => write_json(&mut stream, 400, &Json::obj(vec![("error", Json::s(e))]))?,
                }
            }
        }
        ("POST", "/api/dream/run-council") => {
            match jobs.run_live_dream_council() {
                Ok(res) => write_json(&mut stream, 200, &res)?,
                Err(e) => write_json(&mut stream, 409, &Json::obj(vec![("error", Json::s(e))]))?,
            }
        }
        ("POST", "/api/rsi/mf01") => {
            let v = parse_json(&body_str).unwrap_or(Json::Null);
            let blocks = v.get("blocks").and_then(|x| x.as_u64()).unwrap_or(4096) as usize;
            write_json(&mut stream, 200, &crate::rsi_wave1::run_rsi_mf01_bonsai_gpu_kernel(blocks))?;
        }
        ("POST", "/api/rsi/mf02") => {
            let v = parse_json(&body_str).unwrap_or(Json::Null);
            let dir = v.get("repo_path").and_then(|x| x.as_str()).unwrap_or(".");
            let experts = v.get("total_experts").and_then(|x| x.as_u64()).unwrap_or(512) as u32;
            let keep_pct = v.get("keep_pct").and_then(|x| x.as_f64()).unwrap_or(50.0);
            write_json(&mut stream, 200, &crate::rsi_wave1::run_rsi_mf02_expert_profiler(dir, experts, keep_pct))?;
        }
        ("POST", "/api/rsi/mf03") => {
            let v = parse_json(&body_str).unwrap_or(Json::Null);
            let src = v.get("src_path").and_then(|x| x.as_str()).unwrap_or("");
            let apply = v.get("apply_write").and_then(|x| x.as_bool()).unwrap_or(false);
            let confirmed = v.get("confirmed_two_click").and_then(|x| x.as_bool()).unwrap_or(false);
            if apply && !confirmed {
                write_json(&mut stream, 400, &Json::obj(vec![("error", Json::s("2-Klick Safety Gate: 'confirmed_two_click: true' fuer Schreib-Surgery erforderlich."))]))?;
            } else {
                let default_out = std::env::var("HOME")
                    .map(|h| format!("{h}/Documents/NALA-ModelForge-Exports"))
                    .unwrap_or_else(|_| "./NALA-ModelForge-Exports".into());
                let out_dir = v.get("output_dir").and_then(|x| x.as_str()).filter(|s| !s.trim().is_empty()).unwrap_or(&default_out);
                let alpha = v.get("alpha").and_then(|x| x.as_f64()).unwrap_or(1.0);
                let protect = v.get("protect_coding").and_then(|x| x.as_bool()).unwrap_or(true);
                match crate::rsi_wave1::run_rsi_mf03_stream_gguf_abliteration(src, out_dir, alpha, protect, apply) {
                    Ok(res) => write_json(&mut stream, 200, &res)?,
                    Err(e) => write_json(&mut stream, 400, &Json::obj(vec![("error", Json::s(e))]))?,
                }
            }
        }
        ("POST", "/api/rsi/mf04") => {
            let v = parse_json(&body_str).unwrap_or(Json::Null);
            let mid = v.get("model_id").and_then(|x| x.as_str()).unwrap_or("frontier-1.5tb-moe");
            let chunks = v.get("total_chunks").and_then(|x| x.as_u64()).unwrap_or(128) as u32;
            let steps = v.get("lora_steps").and_then(|x| x.as_u64()).unwrap_or(400) as u32;
            write_json(&mut stream, 200, &crate::rsi_wave1::run_rsi_mf04_grid_4d_orchestrator(mid, chunks, steps))?;
        }
        ("POST", "/api/rsi/mf05") => {
            let v = parse_json(&body_str).unwrap_or(Json::Null);
            let blocks = v.get("blocks").and_then(|x| x.as_u64()).unwrap_or(131_072) as usize;
            write_json(&mut stream, 200, &crate::rsi_wave2::run_rsi_mf05_metal_gpu_bridge(blocks))?;
        }
        ("POST", "/api/rsi/mf06") => {
            let v = parse_json(&body_str).unwrap_or(Json::Null);
            let src = v.get("src_path").and_then(|x| x.as_str()).unwrap_or("");
            let apply = v.get("apply_write").and_then(|x| x.as_bool()).unwrap_or(false);
            let confirmed = v.get("confirmed_two_click").and_then(|x| x.as_bool()).unwrap_or(false);
            if apply && !confirmed {
                write_json(&mut stream, 400, &Json::obj(vec![("error", Json::s("2-Klick Safety Gate: 'confirmed_two_click: true' erforderlich."))]))?;
            } else {
                let default_out = std::env::var("HOME")
                    .map(|h| format!("{h}/Documents/NALA-ModelForge-Exports"))
                    .unwrap_or_else(|_| "./NALA-ModelForge-Exports".into());
                let out_dir = v.get("output_dir").and_then(|x| x.as_str()).filter(|s| !s.trim().is_empty()).unwrap_or(&default_out);
                let alpha = v.get("alpha").and_then(|x| x.as_f64()).unwrap_or(1.0);
                let protect = v.get("protect_coding").and_then(|x| x.as_bool()).unwrap_or(true);
                match crate::rsi_wave2::run_rsi_mf06_quantized_stream_abliteration(src, out_dir, alpha, protect, apply) {
                    Ok(res) => write_json(&mut stream, 200, &res)?,
                    Err(e) => write_json(&mut stream, 400, &Json::obj(vec![("error", Json::s(e))]))?,
                }
            }
        }
        ("POST", "/api/rsi/mf07") => {
            let v = parse_json(&body_str).unwrap_or(Json::Null);
            let src = v.get("src_path").and_then(|x| x.as_str()).unwrap_or("");
            let apply = v.get("apply_write").and_then(|x| x.as_bool()).unwrap_or(false);
            let confirmed = v.get("confirmed_two_click").and_then(|x| x.as_bool()).unwrap_or(false);
            if apply && !confirmed {
                write_json(&mut stream, 400, &Json::obj(vec![("error", Json::s("2-Klick Safety Gate: 'confirmed_two_click: true' erforderlich."))]))?;
            } else {
                let default_out = std::env::var("HOME")
                    .map(|h| format!("{h}/Documents/NALA-ModelForge-Exports"))
                    .unwrap_or_else(|_| "./NALA-ModelForge-Exports".into());
                let out_dir = v.get("output_dir").and_then(|x| x.as_str()).filter(|s| !s.trim().is_empty()).unwrap_or(&default_out);
                let max_drift = v.get("max_drift_pct").and_then(|x| x.as_f64()).unwrap_or(12.0);
                let mut adapters = Vec::new();
                if let Some(arr) = v.get("adapters").and_then(|x| x.as_arr()) {
                    for a in arr.iter().take(8) {
                        let nm = a.get("name").and_then(|x| x.as_str()).unwrap_or("LoRA").to_string();
                        let rk = a.get("rank").and_then(|x| x.as_u64()).unwrap_or(16) as u32;
                        let sc = a.get("scale").and_then(|x| x.as_f64()).unwrap_or(0.8);
                        adapters.push((nm, rk, sc));
                    }
                }
                if adapters.is_empty() {
                    adapters.push(("Rust-Systems-LoRA".into(), 16, 0.85));
                }
                match crate::rsi_wave2::run_rsi_mf07_gguf_lora_static_merger(src, out_dir, &adapters, max_drift, apply) {
                    Ok(res) => write_json(&mut stream, 200, &res)?,
                    Err(e) => write_json(&mut stream, 400, &Json::obj(vec![("error", Json::s(e))]))?,
                }
            }
        }
        ("POST", "/api/rsi/mf08") => {
            let v = parse_json(&body_str).unwrap_or(Json::Null);
            if !v.get("confirmed_two_click").and_then(|x| x.as_bool()).unwrap_or(false) {
                write_json(&mut stream, 400, &Json::obj(vec![("error", Json::s("2-Klick Safety Gate: 'confirmed_two_click: true' erforderlich."))]))?;
            } else {
                let zone = v.get("zone_id").and_then(|x| x.as_str()).unwrap_or("zone_engineering");
                let repo = v.get("fallback_repo").and_then(|x| x.as_str()).unwrap_or(".");
                let default_out = std::env::var("HOME")
                    .map(|h| format!("{h}/Documents/NALA-ModelForge-Exports/rag-dataset"))
                    .unwrap_or_else(|_| "./NALA-ModelForge-Exports/rag-dataset".into());
                let out_dir = v.get("output_dir").and_then(|x| x.as_str()).filter(|s| !s.trim().is_empty()).unwrap_or(&default_out);
                match crate::rsi_wave2::run_rsi_mf08_nala_rag_dataset_generator(zone, repo, out_dir) {
                    Ok(res) => write_json(&mut stream, 200, &res)?,
                    Err(e) => write_json(&mut stream, 400, &Json::obj(vec![("error", Json::s(e))]))?,
                }
            }
        }
        ("GET", "/api/sbom") => write_json(&mut stream, 200, &emit_cyclonedx_sbom())?,
        _ => write_json(&mut stream, 404, &Json::obj(vec![("error", Json::s("Not found"))]))?,
    }
    Ok(())
}

fn parse_plan_request(body_str: &str) -> Result<ForgePlanRequest, String> {
    let v = parse_json(body_str)?;
    let model_path = v
        .get("model_path")
        .and_then(|x| x.as_str())
        .unwrap_or("preset://frontier-1.5tb-moe");
    let model = inspect_local_model(model_path)
        .or_else(|_| preset_by_id(model_path).ok_or_else(|| format!("Unknown model: {model_path}")))?;

    let hw_id = v
        .get("hardware_id")
        .and_then(|x| x.as_str())
        .unwrap_or("mac-studio-512gb");
    let mut hw: HardwareProfile = if hw_id == "custom" {
        let total = v.get("custom_ram_gb").and_then(|x| x.as_f64()).unwrap_or(128.0).clamp(4.0, 8192.0);
        let bw = v.get("custom_bw_gbs").and_then(|x| x.as_f64()).unwrap_or(500.0).clamp(20.0, 20000.0);
        HardwareProfile {
            id: "custom".into(),
            name: format!("Custom Target · {:.0} GB ({:.0} GB/s)", total, bw),
            vendor: v.get("custom_vendor").and_then(|x| x.as_str()).unwrap_or("apple_mlx").into(),
            total_mem_gb: total,
            usable_mem_gb: total * 0.88,
            bandwidth_gbs: bw,
            default_format: "GGUF".into(),
            notes: "Benutzerdefinierte Hardware-Konfiguration".into(),
        }
    } else {
        hardware_by_id(hw_id)
    };
    if let Some(ov) = v.get("override_usable_gb").and_then(|x| x.as_f64()) {
        if ov >= 4.0 && ov <= 8192.0 {
            hw.usable_mem_gb = ov;
        }
    }

    let mut slots = Vec::new();
    if let Some(arr) = v.get("slots").and_then(|x| x.as_arr()) {
        for s in arr.iter().take(16) {
            slots.push(UserSlotGroup {
                label: s.get("label").and_then(|x| x.as_str()).unwrap_or("User").into(),
                role: s.get("role").and_then(|x| x.as_str()).unwrap_or("agent").into(),
                count: s.get("count").and_then(|x| x.as_u64()).unwrap_or(1).clamp(1, 256) as u32,
                context_tokens: s
                    .get("context_tokens")
                    .and_then(|x| x.as_u64())
                    .unwrap_or(65_536)
                    .clamp(2048, 4_194_304) as u32,
                kv_quant: s.get("kv_quant").and_then(|x| x.as_str()).unwrap_or("Q8_0").into(),
                active_concurrent: s
                    .get("active_concurrent")
                    .and_then(|x| x.as_u64())
                    .unwrap_or(1)
                    .clamp(1, 256) as u32,
                prefix_share_pct: s
                    .get("prefix_share_pct")
                    .and_then(|x| x.as_f64())
                    .unwrap_or(35.0),
            });
        }
    }
    if slots.is_empty() {
        slots = UserSlotGroup::default_mike_swarm();
    }

    Ok(ForgePlanRequest {
        model,
        hardware: hw,
        target_format: v.get("target_format").and_then(|x| x.as_str()).unwrap_or("MLX").into(),
        coding_logic_share_pct: v.get("coding_logic_share_pct").and_then(|x| x.as_f64()).unwrap_or(35.0),
        secondary_quant: v.get("secondary_quant").and_then(|x| x.as_str()).unwrap_or("BONSAI_1BIT").into(),
        use_hadamard_rotation: v.get("use_hadamard_rotation").and_then(|x| x.as_bool()).unwrap_or(true),
        moe_prune_pct: v.get("moe_prune_pct").and_then(|x| x.as_f64()).unwrap_or(25.0),
        coding_precision_override: v.get("coding_precision_override").and_then(|x| x.as_str()).unwrap_or("AUTO").into(),
        enable_abliteration: v.get("enable_abliteration").and_then(|x| x.as_bool()).unwrap_or(true),
        abliteration_alpha: v.get("abliteration_alpha").and_then(|x| x.as_f64()).unwrap_or(1.0),
        protect_coding_subspace: v.get("protect_coding_subspace").and_then(|x| x.as_bool()).unwrap_or(true),
        enable_strata_nvme_parking: v.get("enable_strata_nvme_parking").and_then(|x| x.as_bool()).unwrap_or(true),
        lora_count: v.get("lora_count").and_then(|x| x.as_u64()).unwrap_or(2) as u32,
        lora_rank: v.get("lora_rank").and_then(|x| x.as_u64()).unwrap_or(32) as u32,
        enable_nala_rag: v.get("enable_nala_rag").and_then(|x| x.as_bool()).unwrap_or(true),
        rag_reserved_tokens: v.get("rag_reserved_tokens").and_then(|x| x.as_u64()).unwrap_or(32_768) as u32,
        slots,
    })
}

fn handle_job_start(jobs: &JobManager, body_str: &str) -> Result<Json, String> {
    let v = parse_json(body_str)?;
    // Enforce 2-Click Confirmation Gate server-side!
    let confirmed = v.get("confirmed_two_click").and_then(|x| x.as_bool()).unwrap_or(false);
    if !confirmed {
        return Err("2-Klick Safety Gate: 'confirmed_two_click: true' erforderlich.".into());
    }
    let action = v.get("action").and_then(|x| x.as_str()).unwrap_or("export_bundle");
    let default_out = std::env::var("HOME")
        .map(|h| format!("{h}/Documents/NALA-ModelForge-Exports"))
        .unwrap_or_else(|_| "./NALA-ModelForge-Exports".into());
    let out_dir = v
        .get("output_dir")
        .and_then(|x| x.as_str())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(&default_out);

    match action {
        "export_bundle" => {
            let req = parse_plan_request(body_str)?;
            let mut lora_names = Vec::new();
            if let Some(arr) = v.get("lora_names").and_then(|x| x.as_arr()) {
                for n in arr {
                    if let Some(s) = n.as_str() {
                        lora_names.push(s.to_string());
                    }
                }
            }
            if lora_names.is_empty() {
                lora_names.push("Rust-Swift-Systems-LoRA".into());
                lora_names.push("EfficientThink-AntiLoop-LoRA".into());
            }
            jobs.export_recipe_bundle(&req, out_dir, &lora_names)
        }
        "quantize_gguf_live" => {
            let req = parse_plan_request(body_str)?;
            let plan = compute_forge_plan(&req);
            let ttypes = plan
                .get("recipes")
                .and_then(|r| r.get("tensor_types_txt"))
                .and_then(|s| s.as_str())
                .unwrap_or("");
            let base_q = plan
                .get("chosen_coding_quant")
                .and_then(|s| s.as_str())
                .unwrap_or("Q4_K_M");
            jobs.start_live_gguf_quantize(&req.model, ttypes, base_q, out_dir)
        }
        "rust_kernel_suite" => Ok(jobs.run_kernel_verification_job()),
        other => Err(format!("Unknown job action: {other}")),
    }
}

fn write_json(stream: &mut TcpStream, status: u16, val: &Json) -> Result<(), String> {
    write_http(stream, status, "application/json; charset=utf-8", &val.to_string_compact())
}

fn write_http(stream: &mut TcpStream, status: u16, ctype: &str, body: &str) -> Result<(), String> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        413 => "Payload Too Large",
        431 => "Request Header Fields Too Large",
        _ => "Error",
    };
    let resp = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: {ctype}\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         X-Content-Type-Options: nosniff\r\n\
         X-Frame-Options: SAMEORIGIN\r\n\
         Referrer-Policy: no-referrer\r\n\
         X-NALA-Version: {APP_VERSION}\r\n\r\n\
         {body}",
        body.len()
    );
    stream.write_all(resp.as_bytes()).map_err(|e| e.to_string())
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}
