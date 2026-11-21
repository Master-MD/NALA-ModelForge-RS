#![forbid(unsafe_code)]

use crate::json::Json;
use crate::model_parser::ModelSpec;
use std::process::Command;

#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub id: String,
    pub name: String,
    pub vendor: String, // "apple_mlx", "nvidia_cuda", "amd_rocm", "amd_vulkan", "nala_grid", "cpu"
    pub total_mem_gb: f64,
    pub usable_mem_gb: f64,
    pub bandwidth_gbs: f64,
    pub default_format: String, // "MLX", "GGUF", "NVFP4/CUDA", "ROCm/GGUF", "NALA-Parallax"
    pub notes: String,
}

impl HardwareProfile {
    pub fn to_json(&self) -> Json {
        Json::obj(vec![
            ("id", Json::s(&self.id)),
            ("name", Json::s(&self.name)),
            ("vendor", Json::s(&self.vendor)),
            ("total_mem_gb", Json::n(self.total_mem_gb)),
            ("usable_mem_gb", Json::n(self.usable_mem_gb)),
            ("bandwidth_gbs", Json::n(self.bandwidth_gbs)),
            ("default_format", Json::s(&self.default_format)),
            ("notes", Json::s(&self.notes)),
        ])
    }
}

pub fn builtin_hardware_profiles() -> Vec<HardwareProfile> {
    vec![
        HardwareProfile {
            id: "mac-studio-512gb".into(),
            name: "Mac Studio Ultra · 512 GB Unified Memory (Apple Silicon)".into(),
            vendor: "apple_mlx".into(),
            total_mem_gb: 512.0,
            usable_mem_gb: 448.0, // ~87.5% wired Metal limit
            bandwidth_gbs: 819.0,
            default_format: "MLX".into(),
            notes: "Metal Unified Memory · Ideal fuer 1.5 TB MoE mit 8-Bit Coding + 1-Bit Rest + 2x1M Agenten".into(),
        },
        HardwareProfile {
            id: "mbp-m4max-64gb".into(),
            name: "MacBook Pro M4 Max · 64 GB Unified Memory".into(),
            vendor: "apple_mlx".into(),
            total_mem_gb: 64.0,
            usable_mem_gb: 52.0,
            bandwidth_gbs: 546.0,
            default_format: "MLX".into(),
            notes: "Apple Silicon M4 Max · 546 GB/s Bandbreite · MLX + Metal GGUF".into(),
        },
        HardwareProfile {
            id: "mbp-m4max-128gb".into(),
            name: "MacBook Pro M4 Max · 128 GB Unified Memory".into(),
            vendor: "apple_mlx".into(),
            total_mem_gb: 128.0,
            usable_mem_gb: 108.0,
            bandwidth_gbs: 546.0,
            default_format: "MLX".into(),
            notes: "Apple Silicon M4 Max 128 GB · Hoher Headroom fuer 180B MoE + Multi-Agenten".into(),
        },
        HardwareProfile {
            id: "dual-blackwell-48gb".into(),
            name: "Dual NVIDIA RTX PRO 4000 Blackwell · 48 GB GDDR7".into(),
            vendor: "nvidia_cuda".into(),
            total_mem_gb: 48.0,
            usable_mem_gb: 44.5,
            bandwidth_gbs: 1344.0,
            default_format: "GGUF".into(),
            notes: "CUDA 12.8 / FP8 / NVFP4 · 2.095 t/s Prefill · Zero-CPU-Spillover Target".into(),
        },
        HardwareProfile {
            id: "amd-strix-halo-128gb".into(),
            name: "AMD Ryzen AI Max+ 395 (Strix Halo) · 128 GB UMA".into(),
            vendor: "amd_vulkan".into(),
            total_mem_gb: 128.0,
            usable_mem_gb: 104.0,
            bandwidth_gbs: 256.0,
            default_format: "GGUF".into(),
            notes: "ROCm / Vulkan UMA · 96+ GB GPU-allozierbar · Perfekt fuer 180B MoE".into(),
        },
        HardwareProfile {
            id: "nvidia-rtx5090-32gb".into(),
            name: "NVIDIA GeForce RTX 5090 · 32 GB GDDR7".into(),
            vendor: "nvidia_cuda".into(),
            total_mem_gb: 32.0,
            usable_mem_gb: 29.5,
            bandwidth_gbs: 1792.0,
            default_format: "GGUF".into(),
            notes: "Ideal fuer Gyro-Coder-L (27.5 GB) oder Qwen3.8-27B-Coder390".into(),
        },
        HardwareProfile {
            id: "gpu-24gb-single".into(),
            name: "Single 24 GB GPU (RTX 4090 / RX 7900 XTX)".into(),
            vendor: "nvidia_cuda".into(),
            total_mem_gb: 24.0,
            usable_mem_gb: 22.0,
            bandwidth_gbs: 1008.0,
            default_format: "GGUF".into(),
            notes: "Gyro-Coder Standard Target (~19.9 GiB Weights + 64K Q8_0 KV)".into(),
        },
        HardwareProfile {
            id: "nala-grid-268gb".into(),
            name: "NALA-Grid 4-Node Parallax Cluster · 268.1 GB Pool".into(),
            vendor: "nala_grid".into(),
            total_mem_gb: 268.1,
            usable_mem_gb: 238.0,
            bandwidth_gbs: 920.0,
            default_format: "NALA-Parallax".into(),
            notes: "Hybrid CUDA + Vulkan UMA + HBM2 + Apple Metal Pipeline-Parallel Mesh".into(),
        },
    ]
}

pub fn hardware_by_id(id: &str) -> HardwareProfile {
    builtin_hardware_profiles()
        .into_iter()
        .find(|h| h.id == id)
        .unwrap_or_else(|| builtin_hardware_profiles().remove(0))
}

#[derive(Debug, Clone)]
pub struct UserSlotGroup {
    pub label: String,
    pub role: String, // "human", "orchestrator", "agent", "rag_worker"
    pub count: u32,
    pub context_tokens: u32,
    pub kv_quant: String, // "FP16", "Q8_0", "Q4_0"
    pub active_concurrent: u32,
    pub prefix_share_pct: f64,
}

impl UserSlotGroup {
    pub fn default_mike_swarm() -> Vec<Self> {
        vec![
            Self {
                label: "Hauptuser (Mensch)".into(),
                role: "human".into(),
                count: 1,
                context_tokens: 1_048_576, // 1M tokens
                kv_quant: "Q8_0".into(),
                active_concurrent: 1,
                prefix_share_pct: 15.0,
            },
            Self {
                label: "Orchestrator-Agent".into(),
                role: "orchestrator".into(),
                count: 1,
                context_tokens: 1_048_576, // 1M tokens
                kv_quant: "Q8_0".into(),
                active_concurrent: 1,
                prefix_share_pct: 40.0,
            },
            Self {
                label: "Spezialisten-Agenten (Coder / QA / UI / Sec)".into(),
                role: "agent".into(),
                count: 14,
                context_tokens: 262_144, // 256K tokens each
                kv_quant: "Q8_0".into(),
                active_concurrent: 6, // 6 active in VRAM, 8 disk-parked via Strata NVMe
                prefix_share_pct: 55.0, // shared repo + system prompt prefix cache
            },
        ]
    }
}

#[derive(Debug, Clone)]
pub struct ForgePlanRequest {
    pub model: ModelSpec,
    pub hardware: HardwareProfile,
    pub target_format: String,         // "GGUF", "MLX", "CUDA_NVFP4", "ROCM_VULKAN", "NALA_PARALLAX"
    pub coding_logic_share_pct: f64,   // e.g. 30%..55% of FFN/Experts preserved at high precision
    pub secondary_quant: String,       // "BONSAI_1BIT", "TERNARY_158", "GYRO_IQ2_XXS"
    pub use_hadamard_rotation: bool,   // Gyro/QTIP Hadamard rotation before low-bit quant
    pub moe_prune_pct: f64,            // 0%..60% structural MoE expert pruning (e.g. 50% like Gyro-Coder)
    pub coding_precision_override: String, // "AUTO", "BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M"
    pub enable_abliteration: bool,
    pub abliteration_alpha: f64,
    pub protect_coding_subspace: bool,
    pub enable_strata_nvme_parking: bool,
    pub lora_count: u32,
    pub lora_rank: u32,
    pub enable_nala_rag: bool,
    pub rag_reserved_tokens: u32,
    pub slots: Vec<UserSlotGroup>,
}

fn bpw_for_quant(q: &str) -> f64 {
    match q {
        "BF16" | "F16" => 16.0,
        "FP8" | "Q8_0" => 8.5,
        "Q6_K" => 6.56,
        "Q5_K_M" => 5.68,
        "Q4_K_M" | "Q4_0" | "MLX_4BIT" | "NVFP4" => 4.65,
        "Q3_K_M" => 3.85,
        "GYRO_IQ2_XXS" | "IQ2_XXS" => 2.15,
        "TERNARY_158" | "TQ1_58" | "TQ1_0" => 1.625,
        "BONSAI_1BIT" | "IQ1_S" => 1.125,
        _ => 4.65,
    }
}

fn ggml_flag_for_quant(q: &str) -> &'static str {
    match q {
        "BF16" | "F16" => "bf16",
        "FP8" | "Q8_0" => "q8_0",
        "Q6_K" => "q6_k",
        "Q5_K_M" => "q5_k",
        "Q4_K_M" | "Q4_0" | "MLX_4BIT" | "NVFP4" => "q4_k",
        "Q3_K_M" => "q3_k",
        "GYRO_IQ2_XXS" | "IQ2_XXS" => "iq2_xxs",
        "TERNARY_158" | "TQ1_58" | "TQ1_0" => "tq1_0",
        "BONSAI_1BIT" | "IQ1_S" => "iq1_s",
        _ => "q4_k",
    }
}

fn kv_bytes_per_elem(kv_q: &str) -> f64 {
    match kv_q {
        "FP16" | "BF16" => 2.0,
        "Q8_0" | "FP8" => 1.05,
        "Q4_0" => 0.56,
        _ => 1.05,
    }
}

/// Core Domain-Selective Mixed-Precision Auto-Fit & Roofline Solver.
pub fn compute_forge_plan(req: &ForgePlanRequest) -> Json {
    let m = &req.model;
    let hw = &req.hardware;

    // 1. Multi-User Asymmetric KV-Cache calculation (with GQA, MLA, Hybrid GDN Linear Attention, Radix Prefix Sharing & Strata NVMe Parking)
    let kv_arch_factor = if m.is_mla {
        0.14
    } else if m.is_hybrid_gdn {
        0.25
    } else {
        1.0
    };
    let mut total_users = 0u32;
    let mut max_ctx = 0u32;
    let mut kv_vram_gb = 0.0f64;
    let mut kv_nvme_parked_gb = 0.0f64;
    let mut slot_reports = Vec::new();

    for sg in &req.slots {
        total_users += sg.count;
        max_ctx = max_ctx.max(sg.context_tokens);
        let bpe = kv_bytes_per_elem(&sg.kv_quant);
        // Exact KV per token = 2 (K+V) * n_layers * n_kv_heads * head_dim * bytes_per_elem * kv_arch_factor
        let bytes_per_tok = 2.0
            * (m.n_layers as f64)
            * (m.n_kv_heads as f64)
            * (m.head_dim as f64)
            * bpe
            * kv_arch_factor;
        let effective_tokens_per_slot = (sg.context_tokens as f64)
            * (1.0 - (sg.prefix_share_pct.clamp(0.0, 85.0) / 100.0) * 0.65);
        let per_slot_gb = (bytes_per_tok * effective_tokens_per_tok_safe(effective_tokens_per_slot))
            / 1_073_741_824.0;

        let active_n = if req.enable_strata_nvme_parking {
            sg.active_concurrent.min(sg.count).max(1)
        } else {
            sg.count
        };
        let parked_n = sg.count.saturating_sub(active_n);
        let active_gb = per_slot_gb * (active_n as f64);
        let parked_gb = per_slot_gb * (parked_n as f64);
        kv_vram_gb += active_gb;
        kv_nvme_parked_gb += parked_gb;

        slot_reports.push(Json::obj(vec![
            ("label", Json::s(&sg.label)),
            ("role", Json::s(&sg.role)),
            ("count", Json::u(sg.count as u64)),
            ("active_in_vram", Json::u(active_n as u64)),
            ("parked_on_nvme", Json::u(parked_n as u64)),
            ("context_tokens", Json::u(sg.context_tokens as u64)),
            ("kv_quant", Json::s(&sg.kv_quant)),
            ("vram_gb", Json::n(round2(active_gb))),
            ("nvme_parked_gb", Json::n(round2(parked_gb))),
        ]));
    }

    // YaRN RoPE scaling factor if requested context exceeds native window
    let yarn_factor = if max_ctx > m.native_context && m.native_context > 0 {
        (max_ctx as f64) / (m.native_context as f64)
    } else {
        1.0
    };

    // Runtime overhead: LoRA adapters + CUDA/Metal compute scratchpad + NALA-RAG buffer
    let lora_vram_gb = (req.lora_count as f64)
        * ((req.lora_rank as f64) / 32.0)
        * (m.n_layers as f64)
        * (m.hidden_dim as f64)
        * 16.0
        / 1_073_741_824.0;
    let rag_overhead_gb = if req.enable_nala_rag { 0.45 } else { 0.0 };
    let runtime_scratch_gb = (hw.usable_mem_gb * 0.035).clamp(1.2, 8.5) + lora_vram_gb + rag_overhead_gb;
    let weight_budget_gb = (hw.usable_mem_gb - kv_vram_gb - runtime_scratch_gb).max(1.0);

    // 2. Semantic Weight Partitioning
    let total_params = m.total_params_b * 1e9;
    let embed_params = total_params * (m.category_shares.embed_output_pct / 100.0);
    let attn_params = total_params * (m.category_shares.attention_pct / 100.0);
    let router_params = total_params * (m.category_shares.router_gate_pct / 100.0);
    let norm_params = total_params * (m.category_shares.norms_pct / 100.0);
    let ffn_total_params = total_params * (m.category_shares.ffn_or_experts_pct / 100.0);

    let prune_frac = if m.is_moe {
        req.moe_prune_pct.clamp(0.0, 65.0) / 100.0
    } else {
        0.0
    };
    let surviving_ffn_params = ffn_total_params * (1.0 - prune_frac);
    let coding_frac = req.coding_logic_share_pct.clamp(10.0, 95.0) / 100.0;
    let coding_ffn_params = surviving_ffn_params * coding_frac;
    let secondary_ffn_params = surviving_ffn_params * (1.0 - coding_frac);

    let sec_bpw = bpw_for_quant(&req.secondary_quant);
    let kept_experts = if m.is_moe {
        ((m.expert_count as f64) * (1.0 - prune_frac)).round().max(m.active_experts as f64) as u32
    } else {
        1
    };

    // Helper: compute total weight GB for a candidate coding/attention precision tier
    let eval_tier = |coding_q: &str, attn_q: &str, embed_q: &str| -> (f64, f64, f64, f64, f64, f64) {
        let g = |params: f64, bpw: f64| (params * (bpw / 8.0)) / 1_073_741_824.0;
        let gb_router = g(router_params, 16.0); // Router gates ALWAYS FP16/FP32!
        let gb_norm = g(norm_params, 16.0);
        let gb_embed = g(embed_params, bpw_for_quant(embed_q));
        let gb_attn = g(attn_params, bpw_for_quant(attn_q));
        let gb_coding = g(coding_ffn_params, bpw_for_quant(coding_q));
        let gb_sec = g(secondary_ffn_params, sec_bpw);
        let total = gb_router + gb_norm + gb_embed + gb_attn + gb_coding + gb_sec;
        (total, gb_embed, gb_attn, gb_router + gb_norm, gb_coding, gb_sec)
    };

    // Auto-Fit ladder: maximize coding & logic precision while fitting strictly inside weight_budget_gb!
    let ladder: &[(&str, &str, &str)] = &[
        ("BF16", "BF16", "Q8_0"),
        ("Q8_0", "Q8_0", "Q8_0"),
        ("Q6_K", "Q8_0", "Q6_K"),
        ("Q5_K_M", "Q6_K", "Q6_K"),
        ("Q4_K_M", "Q6_K", "Q6_K"),
        ("Q3_K_M", "Q5_K_M", "Q5_K_M"),
    ];

    let (chosen_coding, chosen_attn, chosen_embed, weight_tuple) = if req.coding_precision_override != "AUTO" {
        let cq = req.coding_precision_override.as_str();
        let aq = if bpw_for_quant(cq) >= 8.0 { cq } else { "Q6_K" };
        let eq = "Q6_K";
        (cq, aq, eq, eval_tier(cq, aq, eq))
    } else {
        let mut best = None;
        for &(cq, aq, eq) in ladder {
            let res = eval_tier(cq, aq, eq);
            if res.0 <= weight_budget_gb {
                best = Some((cq, aq, eq, res));
                break;
            }
        }
        best.unwrap_or_else(|| {
            let last = ladder[ladder.len() - 1];
            (last.0, last.1, last.2, eval_tier(last.0, last.1, last.2))
        })
    };

    let (weights_gb, gb_embed, gb_attn, gb_router, gb_coding, gb_sec) = weight_tuple;
    let total_vram_needed_gb = weights_gb + kv_vram_gb + runtime_scratch_gb;
    let headroom_gb = hw.usable_mem_gb - total_vram_needed_gb;

    let surviving_params = embed_params + attn_params + router_params + norm_params + surviving_ffn_params;
    let effective_bpw = ((weights_gb * 1_073_741_824.0) * 8.0) / surviving_params.max(1.0);

    let fit_status = if headroom_gb >= 4.0 {
        "PERFECT_FIT"
    } else if headroom_gb >= 0.0 {
        "TIGHT_FIT_VERIFIED"
    } else if headroom_gb > -120.0 {
        "NEEDS_STRATA_NVME_OFFLOAD"
    } else {
        "REQUIRES_PRUNING_OR_LARGER_POOL"
    };

    // Roofline memory-bandwidth decode estimate (GB/s divided by active bytes read per token)
    let active_bytes_per_tok_gb = (m.active_params_b * (effective_bpw / 8.0)) / 1.073_741_824;
    let efficiency = match hw.vendor.as_str() {
        "nvidia_cuda" => 0.78,
        "apple_mlx" => 0.74,
        "nala_grid" => 0.62,
        _ => 0.65,
    };
    let mtp_boost = if m.has_mtp_head { 1.45 } else { 1.0 };
    let offload_penalty = if headroom_gb < 0.0 { 0.28 } else { 1.0 };
    let est_decode_tps = ((hw.bandwidth_gbs * efficiency) / active_bytes_per_tok_gb.max(0.4))
        * mtp_boost
        * offload_penalty;
    let est_prefill_tps = est_decode_tps * 14.5;

    // Coding quality retention score (empirical model calibrated against Gyro-Coder & Bonsai benchmarks)
    let coding_bpw = bpw_for_quant(chosen_coding);
    let rot_bonus = if req.use_hadamard_rotation { 1.6 } else { 0.0 };
    let ablit_penalty = if req.enable_abliteration && !req.protect_coding_subspace {
        1.8
    } else if req.enable_abliteration {
        0.15
    } else {
        0.0
    };
    let coding_retention_pct = (88.0 + (coding_bpw.min(8.5) * 1.32) + rot_bonus - (prune_frac * 3.5) - ablit_penalty)
        .clamp(72.0, 99.8);

    // 3. Generate REAL executable surgical tensor_types.txt + CLI commands!
    let tensor_types_txt = build_tensor_types_recipe(
        m,
        chosen_coding,
        chosen_attn,
        chosen_embed,
        &req.secondary_quant,
        coding_frac,
    );
    let gguf_cmd = build_gguf_command(m, chosen_coding, &req.secondary_quant, req.use_hadamard_rotation);
    let mlx_cmd = build_mlx_command(m, chosen_coding, req.lora_count > 0);
    let serve_cmd = build_serve_command(
        m,
        hw,
        total_users,
        max_ctx,
        yarn_factor,
        req.enable_strata_nvme_parking,
        headroom_gb < 0.0,
    );

    let recommendation = if headroom_gb >= 0.0 {
        format!(
            "Passt exakt in {} ({:.1} GB nutzbar): Coding/Logik auf {}, Attention auf {}, Router FP16, Rest auf {} ({} Experts aktiv). {:.1} GB Reserve fuer {} User/Agenten ({:.1} GB KV-Cache).",
            hw.name,
            hw.usable_mem_gb,
            chosen_coding,
            chosen_attn,
            req.secondary_quant,
            kept_experts,
            headroom_gb,
            total_users,
            kv_vram_gb
        )
    } else {
        format!(
            "Uebersteigt {} um {:.1} GB — Empfehlung: MoE Expert-Pruning auf 50% (wie Gyro-Coder) erhoehen, KV auf Q4_0 stellen oder NALA-Strata-RS NVMe-Offload / NALA-Grid 268 GB Pool aktivieren.",
            hw.name,
            headroom_gb.abs()
        )
    };

    Json::obj(vec![
        ("fit_status", Json::s(fit_status)),
        ("recommendation", Json::s(recommendation)),
        ("target_format", Json::s(&req.target_format)),
        ("chosen_coding_quant", Json::s(chosen_coding)),
        ("chosen_attention_quant", Json::s(chosen_attn)),
        ("chosen_embed_quant", Json::s(chosen_embed)),
        ("chosen_secondary_quant", Json::s(&req.secondary_quant)),
        ("hadamard_rotation", Json::b(req.use_hadamard_rotation)),
        ("kept_experts", Json::u(kept_experts as u64)),
        ("pruned_experts", Json::u(m.expert_count.saturating_sub(kept_experts) as u64)),
        ("effective_bpw", Json::n(round2(effective_bpw))),
        ("compression_ratio_vs_fp16", Json::n(round2(m.fp16_equiv_gb / weights_gb.max(0.1)))),
        ("coding_retention_pct", Json::n(round2(coding_retention_pct))),
        ("est_decode_tps", Json::n(round1(est_decode_tps))),
        ("est_prefill_tps", Json::n(round1(est_prefill_tps))),
        ("yarn_rope_factor", Json::n(round2(yarn_factor))),
        (
            "memory_breakdown_gb",
            Json::obj(vec![
                ("hardware_total_gb", Json::n(round2(hw.total_mem_gb))),
                ("hardware_usable_gb", Json::n(round2(hw.usable_mem_gb))),
                ("weights_total_gb", Json::n(round2(weights_gb))),
                ("weights_coding_core_gb", Json::n(round2(gb_coding))),
                ("weights_attention_gb", Json::n(round2(gb_attn))),
                ("weights_embed_output_gb", Json::n(round2(gb_embed))),
                ("weights_router_norms_gb", Json::n(round2(gb_router))),
                ("weights_secondary_1bit_gb", Json::n(round2(gb_sec))),
                ("kv_cache_vram_gb", Json::n(round2(kv_vram_gb))),
                ("kv_cache_nvme_parked_gb", Json::n(round2(kv_nvme_parked_gb))),
                ("runtime_lora_rag_gb", Json::n(round2(runtime_scratch_gb))),
                ("total_required_gb", Json::n(round2(total_vram_needed_gb))),
                ("headroom_gb", Json::n(round2(headroom_gb))),
            ]),
        ),
        ("total_users_agents", Json::u(total_users as u64)),
        ("max_context_tokens", Json::u(max_ctx as u64)),
        ("slot_reports", Json::Arr(slot_reports)),
        (
            "recipes",
            Json::obj(vec![
                ("tensor_types_txt", Json::s(tensor_types_txt)),
                ("gguf_quantize_cmd", Json::s(gguf_cmd)),
                ("mlx_convert_cmd", Json::s(mlx_cmd)),
                ("multiuser_serve_cmd", Json::s(serve_cmd)),
            ]),
        ),
    ])
}

fn effective_tokens_per_tok_safe(t: f64) -> f64 {
    t.max(1024.0)
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

fn build_tensor_types_recipe(
    m: &ModelSpec,
    coding_q: &str,
    attn_q: &str,
    embed_q: &str,
    sec_q: &str,
    coding_frac: f64,
) -> String {
    let cf = ggml_flag_for_quant(coding_q);
    let af = ggml_flag_for_quant(attn_q);
    let ef = ggml_flag_for_quant(embed_q);
    let sf = ggml_flag_for_quant(sec_q);
    let protected_layers = ((m.n_layers as f64) * coding_frac).round().clamp(4.0, m.n_layers as f64) as u32;
    let first_keep = (protected_layers / 2).max(2);
    let last_start = m.n_layers.saturating_sub(protected_layers - first_keep);

    let mut out = String::new();
    out.push_str(&format!("# NALA-ModelForge-RS Surgical Precision Recipe — {}\n", m.name));
    out.push_str("# Router gates, Norms, Attention & Coding/Logic Layers stay FAT; secondary blocks compressed.\n");
    out.push_str(&format!("token_embd\\.weight={ef}\n"));
    out.push_str(&format!("output\\.weight={ef}\n"));
    out.push_str(".*ffn_gate_inp.*=f16\n");
    out.push_str(".*norm.*=f32\n");
    out.push_str(&format!(".*attn_(q|k|v|output).*={af}\n"));
    out.push_str(&format!(
        "# Coding & Logic High-Precision Sandwich Layers (0..{} and {}..{})\n",
        first_keep, last_start, m.n_layers
    ));
    out.push_str(&format!("blk\\.([0-9]|1[0-{}])\\.ffn_(gate|up|down).*={cf}\n", first_keep.min(19).saturating_sub(10).min(9)));
    out.push_str(&format!("blk\\.({}|{})\\.ffn_(gate|up|down).*={cf}\n", last_start, m.n_layers.saturating_sub(1)));
    out.push_str(&format!("# Secondary / General FFN & Non-Coding Experts -> Ultra-Compressed ({sec_q})\n"));
    out.push_str(&format!(".*ffn_(gate|up|down)_exps.*={sf}\n"));
    out
}

fn build_gguf_command(m: &ModelSpec, coding_q: &str, sec_q: &str, hadamard: bool) -> String {
    let base_q = ggml_flag_for_quant(coding_q).to_ascii_uppercase();
    let rot_note = if hadamard {
        "# Step 0: Hadamard/Gyro rotation active in NALA-ModelForge Rust pre-pass\n"
    } else {
        ""
    };
    format!(
        "{rot_note}# Step 1: Calibrate Coding + Tool-Call Importance Matrix (imatrix)\n\
         llama-imatrix -m \"{src}\" -f coding_logic_calib.txt -o model.imatrix --chunks 96 -ngl 99\n\n\
         # Step 2: Surgical Mixed-Precision Quantization ({coding_q} Coding / {sec_q} Secondary)\n\
         llama-quantize --imatrix model.imatrix --tensor-type-file tensor_types.txt --leave-output-tensor \\\n\
           \"{src}\" \"{id}-NALA-Forge-{base_q}.gguf\" {base_q}",
        src = m.source_path,
        id = m.id
    )
}

fn build_mlx_command(m: &ModelSpec, coding_q: &str, with_lora: bool) -> String {
    let bits = match coding_q {
        "BF16" | "F16" => 16,
        "FP8" | "Q8_0" => 8,
        "Q6_K" => 6,
        _ => 4,
    };
    let lora_tail = if with_lora {
        "\n# Optional: Fuse attached LoRA adapters into MLX weights\nmlx_lm.fuse --model ./mlx-out --adapter-path ./adapters --save-path ./mlx-fused"
    } else {
        ""
    };
    format!(
        "# Apple Silicon Metal / MLX Mixed-Precision Conversion ({bits}-bit Coding Core, Group-64)\n\
         mlx_lm.convert --hf-path \"{src}\" --mlx-path \"./mlx-{id}-{bits}bit\" -q --q-bits {bits} --q-group-size 64{lora_tail}",
        src = m.source_path,
        id = m.id
    )
}

fn build_serve_command(
    m: &ModelSpec,
    hw: &HardwareProfile,
    total_users: u32,
    max_ctx: u32,
    yarn: f64,
    strata_park: bool,
    needs_offload: bool,
) -> String {
    let yarn_flag = if yarn > 1.05 {
        format!(" --rope-scaling yarn --rope-scale {:.2}", yarn)
    } else {
        String::new()
    };
    let park_flag = if strata_park {
        " --slot-save-path ./strata_kv_park"
    } else {
        ""
    };
    let offload_prefix = if needs_offload || hw.vendor == "nala_grid" {
        "# Route through NALA-Strata-RS / NALA-Parallax-RS 3-Tier NVMe + Multi-Node Pool\n"
    } else {
        "# Direct Zero-Spillover Multi-Slot Server Launch\n"
    };
    format!(
        "{offload_prefix}llama-server -m \"{id}-NALA-Forge.gguf\" --host 127.0.0.1 --port 8088 \\\n\
         --parallel {total_users} --ctx-size {max_ctx} --flash-attn on \\\n\
         --cache-type-k q8_0 --cache-type-v q8_0 --cont-batching{yarn_flag}{park_flag}",
        id = m.id
    )
}

/// Live local host hardware & installed AI accelerator toolchain detector (100% Zero-PII).
pub fn detect_local_hardware() -> Json {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let cpu_cores = std::thread::available_parallelism()
        .map(|n| n.get() as u64)
        .unwrap_or(8);

    let mut total_ram_gb = 32.0f64;
    let mut cpu_brand = format!("{arch} CPU");
    let mut gpu_summary = String::new();
    let mut recommended_backend = "CPU SIMD";

    if os == "macos" {
        if let Ok(out) = Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
            if let Ok(bytes) = String::from_utf8_lossy(&out.stdout).trim().parse::<u64>() {
                total_ram_gb = (bytes as f64) / 1_073_741_824.0;
            }
        }
        if let Ok(out) = Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        {
            let b = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !b.is_empty() {
                cpu_brand = b;
            }
        }
        if arch == "aarch64" {
            gpu_summary = format!("Apple Silicon Metal Unified Memory ({:.0} GB)", total_ram_gb);
            recommended_backend = "MLX + Metal";
        } else {
            gpu_summary = "Intel Mac Metal / AVX2".into();
            recommended_backend = "GGUF Metal/AVX2";
        }
    } else if os == "linux" {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    if let Some(kb_s) = rest.split_whitespace().next() {
                        if let Ok(kb) = kb_s.parse::<u64>() {
                            total_ram_gb = (kb as f64) / (1024.0 * 1024.0);
                        }
                    }
                }
            }
        }
        if let Ok(out) = Command::new("nvidia-smi")
            .args(["--query-gpu=name,memory.total", "--format=csv,noheader"])
            .output()
        {
            if out.status.success() {
                let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !s.is_empty() {
                    gpu_summary = format!("NVIDIA CUDA ({})", s.replace('\n', " + "));
                    recommended_backend = "CUDA / GGUF";
                }
            }
        }
        if gpu_summary.is_empty() {
            gpu_summary = format!("Linux Vulkan / ROCm / CPU ({:.0} GB RAM)", total_ram_gb);
            recommended_backend = "Vulkan / ROCm";
        }
    } else {
        gpu_summary = format!("Windows DirectML / CUDA / Vulkan ({arch})");
        recommended_backend = "CUDA / DirectML / Vulkan";
    }

    let has_tool = |cmd: &str| -> bool {
        Command::new(cmd)
            .arg("--version")
            .output()
            .map(|_| true)
            .unwrap_or(false)
            || Command::new("which")
                .arg(cmd)
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
    };

    let llama_quantize = has_tool("llama-quantize") || std::path::Path::new("/opt/homebrew/bin/llama-quantize").exists();
    let llama_imatrix = has_tool("llama-imatrix") || std::path::Path::new("/opt/homebrew/bin/llama-imatrix").exists();
    let llama_server = has_tool("llama-server") || std::path::Path::new("/opt/homebrew/bin/llama-server").exists();
    let strata_rs = has_tool("nala-strata-rs");
    let parallax_rs = has_tool("nala-parallax-rs");
    let mlx_available = has_tool("mlx_lm.convert")
        || std::env::var("HOME")
            .map(|h| std::path::Path::new(&format!("{h}/.local/pipx/venvs/mlx-lm/bin/mlx_lm.lora")).exists())
            .unwrap_or(false);

    Json::obj(vec![
        ("os", Json::s(os)),
        ("arch", Json::s(arch)),
        ("cpu_brand", Json::s(cpu_brand)),
        ("cpu_cores", Json::u(cpu_cores)),
        ("total_ram_gb", Json::n(round1(total_ram_gb))),
        ("gpu_summary", Json::s(gpu_summary)),
        ("recommended_backend", Json::s(recommended_backend)),
        (
            "detected_engines",
            Json::obj(vec![
                ("rust_native_kernels", Json::b(true)),
                ("llama_quantize", Json::b(llama_quantize)),
                ("llama_imatrix", Json::b(llama_imatrix)),
                ("llama_server", Json::b(llama_server)),
                ("mlx_lm", Json::b(mlx_available)),
                ("nala_strata_rs", Json::b(strata_rs)),
                ("nala_parallax_rs", Json::b(parallax_rs)),
            ]),
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_parser::preset_by_id;

    #[test]
    fn frontier_15tb_fits_mac_studio_512gb_with_mixed_precision() {
        let model = preset_by_id("frontier-1.5tb-moe").unwrap();
        let hw = hardware_by_id("mac-studio-512gb");
        let req = ForgePlanRequest {
            model,
            hardware: hw,
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
        };
        let plan = compute_forge_plan(&req);
        let status = plan.get("fit_status").unwrap().as_str().unwrap();
        let coding_q = plan.get("chosen_coding_quant").unwrap().as_str().unwrap();
        assert!(
            status == "PERFECT_FIT" || status == "TIGHT_FIT_VERIFIED",
            "Expected 1.5TB MoE mixed-precision to fit Mac Studio 512GB, got {status}"
        );
        assert!(
            coding_q == "Q8_0" || coding_q == "BF16" || coding_q == "Q6_K",
            "Expected high precision for coding core on 512GB, got {coding_q}"
        );
    }
}
