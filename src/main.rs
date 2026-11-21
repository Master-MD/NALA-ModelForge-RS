#![forbid(unsafe_code)]

mod jobs;
mod json;
mod mcp;
mod model_parser;
mod ops;
mod planner;
mod quant_math;
mod rsi_wave1;
mod rsi_wave2;
mod security;
mod server;

use crate::jobs::JobManager;
use crate::json::Json;
use crate::model_parser::{inspect_local_model, preset_by_id};
use crate::planner::{ForgePlanRequest, UserSlotGroup, compute_forge_plan, hardware_by_id};
use crate::quant_math::{quant_benchmark_json, run_lora_stack_merge, run_norm_preserving_abliteration};
use crate::security::{APP_VERSION, emit_cyclonedx_sbom, sanitize_output_dir, sha256_bist, validate_endpoint_url};
use std::process::Command;

pub fn run_internal_self_test() -> Json {
    let bist_ok = sha256_bist();
    let ssrf_blocked = validate_endpoint_url("http://169.254.169.254/latest/meta-data", true).is_err();
    let traversal_blocked = sanitize_output_dir("/tmp/../etc/passwd").is_err();

    // Verify Hadamard + Bonsai 1-Bit SQNR rescue
    let qb = quant_benchmark_json(4096);
    let rows: Vec<Json> = qb
        .get("results")
        .and_then(|v| v.as_arr())
        .map(|s| s.to_vec())
        .unwrap_or_default();
    let raw_sqnr = rows
        .iter()
        .find(|r| r.get("method").and_then(|m| m.as_str()).unwrap_or("").contains("Unrotated"))
        .and_then(|r| r.get("sqnr_db").and_then(|x| x.as_f64()))
        .unwrap_or(-99.0);
    let gyro_sqnr = rows
        .iter()
        .find(|r| r.get("method").and_then(|m| m.as_str()).unwrap_or("").contains("Gyro-Hadamard"))
        .and_then(|r| r.get("sqnr_db").and_then(|x| x.as_f64()))
        .unwrap_or(-99.0);
    let hadamard_ok = gyro_sqnr > raw_sqnr;

    // Verify Norm-Preserving Abliteration
    let ab = run_norm_preserving_abliteration(48, 1.0, true, true);
    let ab_ok = ab.get("refusal_suppression_pct").and_then(|v| v.as_f64()).unwrap_or(0.0) > 99.9
        && ab.get("coding_subspace_cosine").and_then(|v| v.as_f64()).unwrap_or(0.0) > 0.999;

    // Verify Multi-LoRA Merge
    let lr = run_lora_stack_merge(&[("Coder-LoRA".into(), 32, 0.8), ("Think-LoRA".into(), 16, 0.5)]);
    let lora_ok = lr.get("adapter_count").and_then(|v| v.as_u64()).unwrap_or(0) == 2;

    // Verify 1.5 TB Frontier MoE fits Mac Studio 512 GB with 16 Multi-User Asymmetric Slots!
    let plan = compute_forge_plan(&ForgePlanRequest {
        model: preset_by_id("frontier-1.5tb-moe").unwrap(),
        hardware: hardware_by_id("mac-studio-512gb"),
        target_format: "MLX".into(),
        coding_logic_share_pct: 32.0,
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
    });
    let fit_status = plan.get("fit_status").and_then(|v| v.as_str()).unwrap_or("");
    let planner_ok = fit_status == "PERFECT_FIT" || fit_status == "TIGHT_FIT_VERIFIED";
    let deslop_ok = ops::run_golden_dom_deslop_audit(None)
        .ok()
        .and_then(|v| v.get("all_passed").and_then(|b| b.as_bool()))
        .unwrap_or(false);
    let mf01_ok = rsi_wave1::run_rsi_mf01_bonsai_gpu_kernel(512)
        .get("status")
        .and_then(|v| v.as_str())
        == Some("VERIFIED_BIT_EXACT");
    let hmac_ok = rsi_wave1::hmac_sha256_hex(b"Jefe", b"what do ya want for nothing?")
        == "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843";
    let f16_ok = (rsi_wave2::f16_to_f32(rsi_wave2::f32_to_f16(-2.5)) + 2.5).abs() < 0.01;
    let pii_ok = rsi_wave2::redact_pii_zero_leak("test@example.com sk-test99999").1 == 2;

    let all_pass = bist_ok
        && ssrf_blocked
        && traversal_blocked
        && hadamard_ok
        && ab_ok
        && lora_ok
        && planner_ok
        && deslop_ok
        && mf01_ok
        && hmac_ok
        && f16_ok
        && pii_ok;

    Json::obj(vec![
        ("status", Json::s(if all_pass { "PASS_ALL_GATES" } else { "FAILED" })),
        ("version", Json::s(APP_VERSION)),
        ("zero_python", Json::b(true)),
        ("fips_sha256_bist", Json::b(bist_ok)),
        ("rfc4231_hmac_sha256_bist", Json::b(hmac_ok)),
        ("ieee_f16_quant_codec", Json::b(f16_ok)),
        ("five_zone_pii_redactor", Json::b(pii_ok)),
        ("sheriff_ssrf_guard", Json::b(ssrf_blocked)),
        ("sheriff_path_traversal_guard", Json::b(traversal_blocked)),
        ("hadamard_bonsai_1bit_sqnr_rescue", Json::b(hadamard_ok)),
        ("rsi_mf01_parallel_bonsai_bit_exact", Json::b(mf01_ok)),
        ("norm_preserving_abliteration_verified", Json::b(ab_ok)),
        ("multi_lora_merge_verified", Json::b(lora_ok)),
        ("frontier_15tb_mac_studio_512gb_fit", Json::b(planner_ok)),
        ("golden_dom_deslop_5_of_5", Json::b(deslop_ok)),
    ])
}

fn open_browser(url: &str) {
    #[cfg(target_os = "macos")]
    let _ = Command::new("open").arg(url).spawn();
    #[cfg(target_os = "linux")]
    let _ = Command::new("xdg-open").arg(url).spawn();
    #[cfg(target_os = "windows")]
    let _ = Command::new("cmd").args(["/C", "start", "", url]).spawn();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let jobs = JobManager::new();

    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("nala-modelforge-rs {APP_VERSION}");
        return;
    }

    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "NALA-ModelForge-RS v{APP_VERSION} — 100% Native Safe Rust Tri-OS LLM Quantizer, Abliteration & Multi-Agent Studio\n\
             Usage:\n\
               nala-modelforge-rs [--serve] [--open] [--headless] [--port 18795]\n\
               nala-modelforge-rs --inspect <path/to/model.gguf|dir|preset://id>\n\
               nala-modelforge-rs --quant-bench\n\
               nala-modelforge-rs --abliterate-verify\n\
               nala-modelforge-rs --mcp-stdio       (20-tool JSON-RPC 2.0 MCP Server)\n\
               nala-modelforge-rs --register-mcp    (Native Rust NALA-MCP-cORe & IDE Registrar)\n\
               nala-modelforge-rs --deslop-audit [f](Native Rust 5/5 Golden-DOM Anti-Slop Gate)\n\
               nala-modelforge-rs --sync-skills     (Native Rust Cross-IDE Skill-Bridge)\n\
               nala-modelforge-rs --pack-deb D OUT  (Native Rust POSIX ar .deb Packager)\n\
               nala-modelforge-rs --emit-sbom       (CycloneDX 1.5 SBOM + SLSA v1.0)\n\
               nala-modelforge-rs --self-test       (Verify all 8 math, deslop & sHeriFF gates)"
        );
        return;
    }

    if args.iter().any(|a| a == "--register-mcp") {
        match ops::register_mcp_and_app_registry() {
            Ok(j) => println!("{}", j.to_string_compact()),
            Err(e) => {
                eprintln!("register-mcp error: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    if let Some(pos) = args.iter().position(|a| a == "--deslop-audit") {
        let target = args.get(pos + 1).map(|s| s.as_str());
        match ops::run_golden_dom_deslop_audit(target) {
            Ok(j) => {
                println!("{}", j.to_string_compact());
                if j.get("all_passed").and_then(|v| v.as_bool()) != Some(true) {
                    std::process::exit(1);
                }
            }
            Err(e) => {
                eprintln!("deslop-audit error: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    if args.iter().any(|a| a == "--sync-skills") {
        match ops::sync_cross_ide_skills() {
            Ok(j) => println!("{}", j.to_string_compact()),
            Err(e) => {
                eprintln!("sync-skills error: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    if let Some(pos) = args.iter().position(|a| a == "--pack-deb") {
        let work = args.get(pos + 1).map(|s| s.as_str()).unwrap_or("");
        let out = args.get(pos + 2).map(|s| s.as_str()).unwrap_or("");
        match ops::pack_debian_ar(work, out) {
            Ok(j) => println!("{}", j.to_string_compact()),
            Err(e) => {
                eprintln!("pack-deb error: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    if args.iter().any(|a| a == "--self-test") {
        let res = run_internal_self_test();
        println!("{}", res.to_string_compact());
        if res.get("status").and_then(|v| v.as_str()) != Some("PASS_ALL_GATES") {
            std::process::exit(1);
        }
        return;
    }

    if args.iter().any(|a| a == "--emit-sbom") {
        println!("{}", emit_cyclonedx_sbom().to_string_compact());
        return;
    }

    if args.iter().any(|a| a == "--quant-bench") {
        println!("{}", quant_benchmark_json(8192).to_string_compact());
        return;
    }

    if args.iter().any(|a| a == "--abliterate-verify") {
        println!(
            "{}",
            run_norm_preserving_abliteration(64, 1.0, true, true).to_string_compact()
        );
        return;
    }

    if let Some(pos) = args.iter().position(|a| a == "--inspect") {
        let target = args.get(pos + 1).map(|s| s.as_str()).unwrap_or("preset://frontier-1.5tb-moe");
        match inspect_local_model(target) {
            Ok(m) => println!("{}", m.to_json().to_string_compact()),
            Err(e) => {
                eprintln!("Inspect error: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    if args.iter().any(|a| a == "--rsi-mf01") {
        println!("{}", rsi_wave1::run_rsi_mf01_bonsai_gpu_kernel(4096).to_string_compact());
        return;
    }

    if let Some(pos) = args.iter().position(|a| a == "--rsi-mf02") {
        let dir = args.get(pos + 1).map(|s| s.as_str()).unwrap_or(".");
        println!("{}", rsi_wave1::run_rsi_mf02_expert_profiler(dir, 512, 50.0).to_string_compact());
        return;
    }

    if args.iter().any(|a| a == "--rsi-mf04") {
        println!("{}", rsi_wave1::run_rsi_mf04_grid_4d_orchestrator("frontier-1.5tb-moe", 128, 400).to_string_compact());
        return;
    }

    if args.iter().any(|a| a == "--rsi-mf05") {
        println!("{}", rsi_wave2::run_rsi_mf05_metal_gpu_bridge(131_072).to_string_compact());
        return;
    }

    if let Some(pos) = args.iter().position(|a| a == "--rsi-mf08") {
        let out_d = args.get(pos + 1).map(|s| s.as_str()).unwrap_or("./nala-rag-dataset");
        match rsi_wave2::run_rsi_mf08_nala_rag_dataset_generator("zone_engineering", ".", out_d) {
            Ok(j) => println!("{}", j.to_string_compact()),
            Err(e) => {
                eprintln!("rsi-mf08 error: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    if args.iter().any(|a| a == "--mcp-stdio" || a == "--stdio-mcp") {
        mcp::run_mcp_stdio(jobs);
        return;
    }

    let mut port: u16 = std::env::var("NALA_MODELFORGE_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(18795);
    if let Some(pos) = args.iter().position(|a| a == "--port") {
        if let Some(p) = args.get(pos + 1).and_then(|v| v.parse::<u16>().ok()) {
            port = p;
        }
    }

    let headless = args.iter().any(|a| a == "--headless");
    let explicit_open = args.iter().any(|a| a == "--open");
    if explicit_open || (!headless && args.len() == 1) {
        let url = format!("http://127.0.0.1:{port}/");
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(350));
            open_browser(&url);
        });
    }

    if let Err(e) = server::run_http_server(port, jobs) {
        eprintln!("Fatal server error: {e}");
        std::process::exit(1);
    }
}
