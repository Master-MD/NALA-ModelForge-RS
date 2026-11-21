#![forbid(unsafe_code)]

use crate::json::{Json, parse_json};
use crate::security::sanitize_existing_path;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct TensorSummary {
    pub name: String,
    pub category: String, // "router", "attention", "moe_expert", "dense_ffn", "embed_output", "norm"
    pub ggml_type: String,
    pub elements: u64,
    pub bytes_est: u64,
}

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub id: String,
    pub name: String,
    pub source_format: String, // "gguf", "safetensors", "hf_dir", "preset"
    pub source_path: String,
    pub architecture: String,
    pub total_params_b: f64,
    pub active_params_b: f64,
    pub file_size_gb: f64,
    pub fp16_equiv_gb: f64,
    pub n_layers: u32,
    pub hidden_dim: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub native_context: u32,
    pub is_moe: bool,
    pub expert_count: u32,
    pub active_experts: u32,
    pub is_mla: bool,        // DeepSeek Multi-head Latent Attention KV compression (~7x)
    pub is_hybrid_gdn: bool, // Qwen3.8-Next / Frontier 3:1 Gated DeltaNet Hybrid Linear Attention (~4x KV savings)
    pub has_mtp_head: bool,
    pub is_uncensored_base: bool,
    pub tensor_count: u64,
    pub category_shares: CategoryShares,
    pub sample_tensors: Vec<TensorSummary>,
}

#[derive(Debug, Clone)]
pub struct CategoryShares {
    pub embed_output_pct: f64,
    pub attention_pct: f64,
    pub router_gate_pct: f64,
    pub ffn_or_experts_pct: f64,
    pub norms_pct: f64,
}

impl ModelSpec {
    pub fn to_json(&self) -> Json {
        let samples = self
            .sample_tensors
            .iter()
            .take(24)
            .map(|t| {
                Json::obj(vec![
                    ("name", Json::s(&t.name)),
                    ("category", Json::s(&t.category)),
                    ("dtype", Json::s(&t.ggml_type)),
                    ("elements", Json::u(t.elements)),
                    ("mb", Json::n((t.bytes_est as f64) / (1024.0 * 1024.0))),
                ])
            })
            .collect();
        Json::obj(vec![
            ("id", Json::s(&self.id)),
            ("name", Json::s(&self.name)),
            ("source_format", Json::s(&self.source_format)),
            ("source_path", Json::s(&self.source_path)),
            ("architecture", Json::s(&self.architecture)),
            ("total_params_b", Json::n(self.total_params_b)),
            ("active_params_b", Json::n(self.active_params_b)),
            ("file_size_gb", Json::n(self.file_size_gb)),
            ("fp16_equiv_gb", Json::n(self.fp16_equiv_gb)),
            ("n_layers", Json::u(self.n_layers as u64)),
            ("hidden_dim", Json::u(self.hidden_dim as u64)),
            ("n_heads", Json::u(self.n_heads as u64)),
            ("n_kv_heads", Json::u(self.n_kv_heads as u64)),
            ("head_dim", Json::u(self.head_dim as u64)),
            ("native_context", Json::u(self.native_context as u64)),
            ("is_moe", Json::b(self.is_moe)),
            ("expert_count", Json::u(self.expert_count as u64)),
            ("active_experts", Json::u(self.active_experts as u64)),
            ("is_mla", Json::b(self.is_mla)),
            ("is_hybrid_gdn", Json::b(self.is_hybrid_gdn)),
            ("has_mtp_head", Json::b(self.has_mtp_head)),
            ("is_uncensored_base", Json::b(self.is_uncensored_base)),
            ("tensor_count", Json::u(self.tensor_count)),
            (
                "category_shares",
                Json::obj(vec![
                    ("embed_output_pct", Json::n(self.category_shares.embed_output_pct)),
                    ("attention_pct", Json::n(self.category_shares.attention_pct)),
                    ("router_gate_pct", Json::n(self.category_shares.router_gate_pct)),
                    ("ffn_or_experts_pct", Json::n(self.category_shares.ffn_or_experts_pct)),
                    ("norms_pct", Json::n(self.category_shares.norms_pct)),
                ]),
            ),
            ("sample_tensors", Json::Arr(samples)),
        ])
    }
}

pub fn builtin_presets() -> Vec<ModelSpec> {
    vec![
        ModelSpec {
            id: "frontier-1.5tb-moe".into(),
            name: "Frontier-750B/1T MoE (1.5 TB FP16 Original · 512 Experts)".into(),
            source_format: "preset".into(),
            source_path: "preset://frontier-1.5tb-moe".into(),
            architecture: "qwen4moe".into(),
            total_params_b: 750.0,
            active_params_b: 32.0,
            file_size_gb: 1500.0,
            fp16_equiv_gb: 1500.0,
            n_layers: 96,
            hidden_dim: 8192,
            n_heads: 64,
            n_kv_heads: 8,
            head_dim: 128,
            native_context: 262_144,
            is_moe: true,
            expert_count: 512,
            active_experts: 16,
            is_mla: false,
            is_hybrid_gdn: true,
            has_mtp_head: true,
            is_uncensored_base: false,
            tensor_count: 1446,
            category_shares: CategoryShares {
                embed_output_pct: 1.8,
                attention_pct: 5.4,
                router_gate_pct: 0.35,
                ffn_or_experts_pct: 92.35,
                norms_pct: 0.10,
            },
            sample_tensors: synthetic_moe_tensors(96, 512),
        },
        ModelSpec {
            id: "qwen38-flash-next-180b".into(),
            name: "Qwen3.8-Flash-Next 180B MoE (512 Experts · Gyro-Coder Target)".into(),
            source_format: "preset".into(),
            source_path: "preset://qwen38-flash-next-180b".into(),
            architecture: "qwen4exp".into(),
            total_params_b: 180.0,
            active_params_b: 6.0,
            file_size_gb: 360.0,
            fp16_equiv_gb: 360.0,
            n_layers: 64,
            hidden_dim: 5120,
            n_heads: 40,
            n_kv_heads: 8,
            head_dim: 128,
            native_context: 262_144,
            is_moe: true,
            expert_count: 512,
            active_experts: 10,
            is_mla: false,
            is_hybrid_gdn: true,
            has_mtp_head: true,
            is_uncensored_base: true,
            tensor_count: 966,
            category_shares: CategoryShares {
                embed_output_pct: 2.2,
                attention_pct: 6.1,
                router_gate_pct: 0.40,
                ffn_or_experts_pct: 91.15,
                norms_pct: 0.15,
            },
            sample_tensors: synthetic_moe_tensors(64, 512),
        },
        ModelSpec {
            id: "deepseek-v41-671b".into(),
            name: "DeepSeek-V4.1 671B MoE (256 Experts · MLA Ultra-Low KV)".into(),
            source_format: "preset".into(),
            source_path: "preset://deepseek-v41-671b".into(),
            architecture: "deepseek2".into(),
            total_params_b: 671.0,
            active_params_b: 37.0,
            file_size_gb: 1342.0,
            fp16_equiv_gb: 1342.0,
            n_layers: 61,
            hidden_dim: 7168,
            n_heads: 128,
            n_kv_heads: 4, // MLA compressed latent equivalent
            head_dim: 128,
            native_context: 1_048_576,
            is_moe: true,
            expert_count: 256,
            active_experts: 8,
            is_mla: true,
            is_hybrid_gdn: false,
            has_mtp_head: true,
            is_uncensored_base: false,
            tensor_count: 1024,
            category_shares: CategoryShares {
                embed_output_pct: 1.9,
                attention_pct: 4.8,
                router_gate_pct: 0.30,
                ffn_or_experts_pct: 92.88,
                norms_pct: 0.12,
            },
            sample_tensors: synthetic_moe_tensors(61, 256),
        },
        ModelSpec {
            id: "qwen38-27b-coder390".into(),
            name: "Qwen3.8-27B-Coder390-EfficientThink (Dense · SFT+RLOO+MTP+DFlash2)".into(),
            source_format: "preset".into(),
            source_path: "preset://qwen38-27b-coder390".into(),
            architecture: "qwen3".into(),
            total_params_b: 27.0,
            active_params_b: 27.0,
            file_size_gb: 54.0,
            fp16_equiv_gb: 54.0,
            n_layers: 64,
            hidden_dim: 5120,
            n_heads: 40,
            n_kv_heads: 8,
            head_dim: 128,
            native_context: 262_144,
            is_moe: false,
            expert_count: 1,
            active_experts: 1,
            is_mla: false,
            is_hybrid_gdn: false,
            has_mtp_head: true,
            is_uncensored_base: false,
            tensor_count: 710,
            category_shares: CategoryShares {
                embed_output_pct: 5.5,
                attention_pct: 21.5,
                router_gate_pct: 0.0,
                ffn_or_experts_pct: 72.7,
                norms_pct: 0.3,
            },
            sample_tensors: synthetic_dense_tensors(64),
        },
        ModelSpec {
            id: "llama4-maverick-400b".into(),
            name: "Llama-4-Maverick 400B MoE (128 Experts · 1M Native Context)".into(),
            source_format: "preset".into(),
            source_path: "preset://llama4-maverick-400b".into(),
            architecture: "llama4".into(),
            total_params_b: 400.0,
            active_params_b: 17.0,
            file_size_gb: 800.0,
            fp16_equiv_gb: 800.0,
            n_layers: 48,
            hidden_dim: 5120,
            n_heads: 40,
            n_kv_heads: 8,
            head_dim: 128,
            native_context: 1_048_576,
            is_moe: true,
            expert_count: 128,
            active_experts: 2,
            is_mla: false,
            is_hybrid_gdn: false,
            has_mtp_head: false,
            is_uncensored_base: false,
            tensor_count: 640,
            category_shares: CategoryShares {
                embed_output_pct: 2.1,
                attention_pct: 5.8,
                router_gate_pct: 0.25,
                ffn_or_experts_pct: 91.70,
                norms_pct: 0.15,
            },
            sample_tensors: synthetic_moe_tensors(48, 128),
        },
        ModelSpec {
            id: "prism-bonsai-8b-1bit".into(),
            name: "PrismML Bonsai-8B (Native 1-Bit g128 Qwen3 Backbone Reference)".into(),
            source_format: "preset".into(),
            source_path: "preset://prism-bonsai-8b-1bit".into(),
            architecture: "qwen3_bonsai1b".into(),
            total_params_b: 8.0,
            active_params_b: 8.0,
            file_size_gb: 1.15,
            fp16_equiv_gb: 16.0,
            n_layers: 36,
            hidden_dim: 4096,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            native_context: 131_072,
            is_moe: false,
            expert_count: 1,
            active_experts: 1,
            is_mla: false,
            is_hybrid_gdn: false,
            has_mtp_head: false,
            is_uncensored_base: false,
            tensor_count: 398,
            category_shares: CategoryShares {
                embed_output_pct: 12.0,
                attention_pct: 22.0,
                router_gate_pct: 0.0,
                ffn_or_experts_pct: 65.5,
                norms_pct: 0.5,
            },
            sample_tensors: synthetic_dense_tensors(36),
        },
    ]
}

fn synthetic_moe_tensors(n_layers: u32, experts: u32) -> Vec<TensorSummary> {
    vec![
        TensorSummary { name: "token_embd.weight".into(), category: "embed_output".into(), ggml_type: "BF16".into(), elements: 151_936 * 5120, bytes_est: 1_555_824_640 },
        TensorSummary { name: "blk.0.attn_q.weight".into(), category: "attention".into(), ggml_type: "BF16".into(), elements: 5120 * 5120, bytes_est: 52_428_800 },
        TensorSummary { name: "blk.0.attn_k.weight".into(), category: "attention".into(), ggml_type: "BF16".into(), elements: 5120 * 1024, bytes_est: 10_485_760 },
        TensorSummary { name: "blk.0.attn_v.weight".into(), category: "attention".into(), ggml_type: "BF16".into(), elements: 5120 * 1024, bytes_est: 10_485_760 },
        TensorSummary { name: "blk.0.attn_output.weight".into(), category: "attention".into(), ggml_type: "BF16".into(), elements: 5120 * 5120, bytes_est: 52_428_800 },
        TensorSummary { name: "blk.0.ffn_gate_inp.weight".into(), category: "router".into(), ggml_type: "F32".into(), elements: (experts as u64) * 5120, bytes_est: (experts as u64) * 5120 * 4 },
        TensorSummary { name: format!("blk.0.ffn_gate_exps.weight [{} experts]", experts), category: "moe_expert".into(), ggml_type: "BF16".into(), elements: (experts as u64) * 5120 * 1536, bytes_est: (experts as u64) * 5120 * 1536 * 2 },
        TensorSummary { name: format!("blk.0.ffn_up_exps.weight [{} experts]", experts), category: "moe_expert".into(), ggml_type: "BF16".into(), elements: (experts as u64) * 5120 * 1536, bytes_est: (experts as u64) * 5120 * 1536 * 2 },
        TensorSummary { name: format!("blk.0.ffn_down_exps.weight [{} experts]", experts), category: "moe_expert".into(), ggml_type: "BF16".into(), elements: (experts as u64) * 1536 * 5120, bytes_est: (experts as u64) * 5120 * 1536 * 2 },
        TensorSummary { name: format!("blk.{}.ffn_down_exps.weight [last]", n_layers.saturating_sub(1)), category: "moe_expert".into(), ggml_type: "BF16".into(), elements: (experts as u64) * 1536 * 5120, bytes_est: (experts as u64) * 5120 * 1536 * 2 },
        TensorSummary { name: "output.weight".into(), category: "embed_output".into(), ggml_type: "BF16".into(), elements: 151_936 * 5120, bytes_est: 1_555_824_640 },
    ]
}

fn synthetic_dense_tensors(n_layers: u32) -> Vec<TensorSummary> {
    vec![
        TensorSummary { name: "token_embd.weight".into(), category: "embed_output".into(), ggml_type: "BF16".into(), elements: 151_936 * 5120, bytes_est: 1_555_824_640 },
        TensorSummary { name: "blk.0.attn_q.weight".into(), category: "attention".into(), ggml_type: "BF16".into(), elements: 5120 * 5120, bytes_est: 52_428_800 },
        TensorSummary { name: "blk.0.attn_output.weight".into(), category: "attention".into(), ggml_type: "BF16".into(), elements: 5120 * 5120, bytes_est: 52_428_800 },
        TensorSummary { name: "blk.0.ffn_gate.weight".into(), category: "dense_ffn".into(), ggml_type: "BF16".into(), elements: 5120 * 17408, bytes_est: 178_257_920 },
        TensorSummary { name: "blk.0.ffn_down.weight".into(), category: "dense_ffn".into(), ggml_type: "BF16".into(), elements: 17408 * 5120, bytes_est: 178_257_920 },
        TensorSummary { name: format!("blk.{}.ffn_down.weight", n_layers.saturating_sub(1)), category: "dense_ffn".into(), ggml_type: "BF16".into(), elements: 17408 * 5120, bytes_est: 178_257_920 },
        TensorSummary { name: "output.weight".into(), category: "embed_output".into(), ggml_type: "BF16".into(), elements: 151_936 * 5120, bytes_est: 1_555_824_640 },
    ]
}

pub fn preset_by_id(id: &str) -> Option<ModelSpec> {
    builtin_presets().into_iter().find(|p| p.id == id)
}

pub fn classify_tensor_name(name: &str) -> &'static str {
    let l = name.to_ascii_lowercase();
    if l.contains("gate_inp") || l.contains(".router.") || l.contains("mlp.gate.weight") {
        "router"
    } else if l.contains("token_embd")
        || l.contains("embed_tokens")
        || l == "output.weight"
        || l.contains("lm_head")
    {
        "embed_output"
    } else if l.contains("attn_") || l.contains("self_attn") || l.contains("q_proj") || l.contains("k_proj") || l.contains("v_proj") || l.contains("o_proj") {
        "attention"
    } else if l.contains("_exps") || l.contains(".experts.") || l.contains("expert_") {
        "moe_expert"
    } else if l.contains("norm") || l.contains("ln_") {
        "norm"
    } else {
        "dense_ffn"
    }
}

/// Inspect a real local file or folder (.gguf, Ollama blob, .safetensors, or HF model dir)
/// using zero-copy streaming header reads (never loads weights into RAM!).
pub fn inspect_local_model(raw_path: &str) -> Result<ModelSpec, String> {
    if let Some(pid) = raw_path.strip_prefix("preset://") {
        return preset_by_id(pid).ok_or_else(|| format!("Unknown preset: {pid}"));
    }
    let canon = sanitize_existing_path(raw_path)?;
    if canon.is_dir() {
        return inspect_hf_directory(&canon);
    }
    // Peek first 8 bytes to detect magic ("GGUF" vs Safetensors header)
    let mut f = File::open(&canon).map_err(|e| format!("Cannot open file: {e}"))?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)
        .map_err(|e| format!("File too small to inspect: {e}"))?;
    drop(f);
    if &magic == b"GGUF" {
        return parse_gguf_file(&canon);
    }
    if canon
        .extension()
        .and_then(|s| s.to_str())
        .map(|e| e.eq_ignore_ascii_case("safetensors"))
        .unwrap_or(false)
    {
        return parse_safetensors_file(&canon);
    }
    Err(format!(
        "Unsupported file format at {} (expected GGUF magic 'GGUF', .safetensors, or HF model directory)",
        canon.display()
    ))
}

fn ggml_type_info(code: u32) -> (&'static str, f64) {
    match code {
        0 => ("F32", 4.0),
        1 => ("F16", 2.0),
        2 => ("Q4_0", 0.5625),
        3 => ("Q4_1", 0.625),
        6 => ("Q5_0", 0.6875),
        7 => ("Q5_1", 0.75),
        8 => ("Q8_0", 1.0625),
        9 => ("Q8_1", 1.125),
        10 => ("Q2_K", 0.328),
        11 => ("Q3_K", 0.43),
        12 => ("Q4_K", 0.57),
        13 => ("Q5_K", 0.69),
        14 => ("Q6_K", 0.82),
        15 => ("Q8_K", 1.08),
        16 => ("IQ2_XXS", 0.258),
        17 => ("IQ2_XS", 0.29),
        18 => ("IQ3_XXS", 0.382),
        19 => ("IQ1_S", 0.195),
        20 => ("IQ4_NL", 0.5625),
        21 => ("IQ3_S", 0.43),
        22 => ("IQ2_S", 0.3125),
        23 => ("IQ4_XS", 0.531),
        24 => ("I8", 1.0),
        25 => ("I16", 2.0),
        26 => ("I32", 4.0),
        27 => ("I64", 8.0),
        28 => ("F64", 8.0),
        29 => ("IQ1_M", 0.22),
        30 => ("BF16", 2.0),
        34 => ("TQ1_0", 0.21),
        35 => ("TQ2_0", 0.26),
        _ => ("CUSTOM_QUANT", 0.55),
    }
}

fn read_u32_le<R: Read>(r: &mut R) -> Result<u32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| format!("GGUF EOF u32: {e}"))?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64_le<R: Read>(r: &mut R) -> Result<u64, String> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|e| format!("GGUF EOF u64: {e}"))?;
    Ok(u64::from_le_bytes(b))
}

fn read_gguf_string<R: Read>(r: &mut R) -> Result<String, String> {
    let len = read_u64_le(r)?;
    if len > 4 * 1024 * 1024 {
        return Err(format!("GGUF string length suspiciously large: {len}"));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf).map_err(|e| format!("GGUF EOF str: {e}"))?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn skip_gguf_value<R: Read + Seek>(r: &mut R, vtype: u32) -> Result<Option<String>, String> {
    match vtype {
        0 => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b).map_err(|e| e.to_string())?;
            Ok(Some(b[0].to_string()))
        }
        1 => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b).map_err(|e| e.to_string())?;
            Ok(Some((b[0] as i8).to_string()))
        }
        2 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b).map_err(|e| e.to_string())?;
            Ok(Some(u16::from_le_bytes(b).to_string()))
        }
        3 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b).map_err(|e| e.to_string())?;
            Ok(Some(i16::from_le_bytes(b).to_string()))
        }
        4 => Ok(Some(read_u32_le(r)?.to_string())),
        5 => {
            let mut b = [0u8; 4];
            r.read_exact(&mut b).map_err(|e| e.to_string())?;
            Ok(Some(i32::from_le_bytes(b).to_string()))
        }
        6 => {
            let mut b = [0u8; 4];
            r.read_exact(&mut b).map_err(|e| e.to_string())?;
            Ok(Some(f32::from_le_bytes(b).to_string()))
        }
        7 => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b).map_err(|e| e.to_string())?;
            Ok(Some(if b[0] != 0 { "true" } else { "false" }.into()))
        }
        8 => {
            let len = read_u64_le(r)?;
            if len <= 256 {
                let mut buf = vec![0u8; len as usize];
                r.read_exact(&mut buf).map_err(|e| e.to_string())?;
                Ok(Some(String::from_utf8_lossy(&buf).into_owned()))
            } else {
                r.seek(SeekFrom::Current(len as i64)).map_err(|e| e.to_string())?;
                Ok(None)
            }
        }
        9 => {
            let elem_type = read_u32_le(r)?;
            let count = read_u64_le(r)?;
            if count > 50_000_000 {
                return Err("GGUF array count out of bounds".into());
            }
            let fixed_size: Option<u64> = match elem_type {
                0 | 1 | 7 => Some(1),
                2 | 3 => Some(2),
                4 | 5 | 6 => Some(4),
                10 | 11 | 12 => Some(8),
                _ => None,
            };
            if let Some(sz) = fixed_size {
                let skip_bytes = count.checked_mul(sz).ok_or("GGUF array size overflow")?;
                r.seek(SeekFrom::Current(skip_bytes as i64))
                    .map_err(|e| e.to_string())?;
            } else {
                for _ in 0..count {
                    skip_gguf_value(r, elem_type)?;
                }
            }
            Ok(None)
        }
        10 => Ok(Some(read_u64_le(r)?.to_string())),
        11 => {
            let mut b = [0u8; 8];
            r.read_exact(&mut b).map_err(|e| e.to_string())?;
            Ok(Some(i64::from_le_bytes(b).to_string()))
        }
        12 => {
            let mut b = [0u8; 8];
            r.read_exact(&mut b).map_err(|e| e.to_string())?;
            Ok(Some(f64::from_le_bytes(b).to_string()))
        }
        other => Err(format!("Unknown GGUF value type {other}")),
    }
}

/// Real binary GGUF v2/v3 zero-copy streaming parser.
pub fn parse_gguf_file(path: &Path) -> Result<ModelSpec, String> {
    let meta = std::fs::metadata(path).map_err(|e| e.to_string())?;
    let file_size_gb = (meta.len() as f64) / 1_073_741_824.0;
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut r = BufReader::with_capacity(64 * 1024, file);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != b"GGUF" {
        return Err("Invalid GGUF magic header".into());
    }
    let version = read_u32_le(&mut r)?;
    if !(2..=3).contains(&version) {
        return Err(format!("Unsupported GGUF version {version}"));
    }
    let tensor_count = read_u64_le(&mut r)?;
    let kv_count = read_u64_le(&mut r)?;
    if kv_count > 100_000 || tensor_count > 500_000 {
        return Err("GGUF header counts exceed safety bounds".into());
    }

    let mut kv: BTreeMap<String, String> = BTreeMap::new();
    for _ in 0..kv_count {
        let key = read_gguf_string(&mut r)?;
        let vtype = read_u32_le(&mut r)?;
        if let Some(val) = skip_gguf_value(&mut r, vtype)? {
            if key.len() < 128 && val.len() <= 256 {
                kv.insert(key, val);
            }
        }
    }

    let arch = kv
        .get("general.architecture")
        .cloned()
        .unwrap_or_else(|| "llama".into());
    let name = kv
        .get("general.name")
        .cloned()
        .unwrap_or_else(|| {
            path.file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "Local GGUF Model".into())
        });

    let get_u32 = |suffix: &str, default: u32| -> u32 {
        kv.get(&format!("{arch}.{suffix}"))
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(default)
    };

    let n_layers = get_u32("block_count", 32);
    let hidden_dim = get_u32("embedding_length", 4096);
    let n_heads = get_u32("attention.head_count", 32);
    let n_kv_heads = get_u32("attention.head_count_kv", n_heads.min(8));
    let head_dim = get_u32(
        "attention.key_length",
        if n_heads > 0 { hidden_dim / n_heads } else { 128 },
    );
    let native_context = get_u32("context_length", 32_768);
    let expert_count = get_u32("expert_count", 1);
    let active_experts = get_u32("expert_used_count", if expert_count > 1 { 2 } else { 1 });

    // Read tensor descriptors (streaming, zero weight data read!)
    let mut total_elements: u64 = 0;
    let mut cat_elems: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut sample_tensors = Vec::new();
    let mut has_mtp_head = false;

    for idx in 0..tensor_count {
        let tname = read_gguf_string(&mut r)?;
        let n_dims = read_u32_le(&mut r)?;
        if n_dims > 8 {
            return Err("GGUF tensor n_dims out of bounds".into());
        }
        let mut elems: u64 = 1;
        for _ in 0..n_dims {
            let d = read_u64_le(&mut r)?;
            elems = elems.saturating_mul(d);
        }
        let dtype_code = read_u32_le(&mut r)?;
        let _offset = read_u64_le(&mut r)?;
        let (dtype_name, bpe) = ggml_type_info(dtype_code);
        let cat = classify_tensor_name(&tname);
        if tname.to_ascii_lowercase().contains("mtp") {
            has_mtp_head = true;
        }
        total_elements = total_elements.saturating_add(elems);
        *cat_elems.entry(cat).or_insert(0) += elems;

        if sample_tensors.len() < 24 || idx + 1 == tensor_count {
            if sample_tensors.len() == 24 {
                sample_tensors.pop();
            }
            sample_tensors.push(TensorSummary {
                name: tname,
                category: cat.into(),
                ggml_type: dtype_name.into(),
                elements: elems,
                bytes_est: (elems as f64 * bpe) as u64,
            });
        }
    }

    let tot = (total_elements.max(1)) as f64;
    let pct = |k: &str| (*cat_elems.get(k).unwrap_or(&0) as f64 / tot) * 100.0;
    let total_params_b = tot / 1e9;
    let is_moe = expert_count > 1 || *cat_elems.get("moe_expert").unwrap_or(&0) > 0;
    let moe_share = pct("moe_expert") / 100.0;
    let active_params_b = if is_moe && expert_count > 0 {
        total_params_b * ((1.0 - moe_share) + moe_share * (active_experts as f64 / expert_count as f64))
    } else {
        total_params_b
    };

    let lower_name = name.to_ascii_lowercase();
    let is_uncensored_base = lower_name.contains("abliterated")
        || lower_name.contains("uncensored")
        || lower_name.contains("heretic")
        || lower_name.contains("base");

    Ok(ModelSpec {
        id: format!("local-gguf-{}", path.file_name().unwrap_or_default().to_string_lossy()),
        name,
        source_format: format!("gguf-v{version}"),
        source_path: path.display().to_string(),
        architecture: arch.clone(),
        total_params_b,
        active_params_b,
        file_size_gb,
        fp16_equiv_gb: (tot * 2.0) / 1_073_741_824.0,
        n_layers,
        hidden_dim,
        n_heads,
        n_kv_heads,
        head_dim,
        native_context,
        is_moe,
        expert_count,
        active_experts,
        is_mla: arch.contains("deepseek"),
        is_hybrid_gdn: arch.contains("qwen4") || arch.contains("next") || arch.contains("gdn"),
        has_mtp_head,
        is_uncensored_base,
        tensor_count,
        category_shares: CategoryShares {
            embed_output_pct: pct("embed_output"),
            attention_pct: pct("attention"),
            router_gate_pct: pct("router"),
            ffn_or_experts_pct: pct("moe_expert") + pct("dense_ffn"),
            norms_pct: pct("norm"),
        },
        sample_tensors,
    })
}

/// Real zero-copy streaming Safetensors header parser (reads only 8-byte header size + JSON index).
pub fn parse_safetensors_file(path: &Path) -> Result<ModelSpec, String> {
    let meta = std::fs::metadata(path).map_err(|e| e.to_string())?;
    let file_size_gb = (meta.len() as f64) / 1_073_741_824.0;
    let mut f = File::open(path).map_err(|e| e.to_string())?;
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf).map_err(|e| e.to_string())?;
    let header_len = u64::from_le_bytes(len_buf);
    if header_len == 0 || header_len > 32 * 1024 * 1024 {
        return Err(format!("Invalid safetensors header length: {header_len}"));
    }
    let mut header_bytes = vec![0u8; header_len as usize];
    f.read_exact(&mut header_bytes).map_err(|e| e.to_string())?;
    let header_str = String::from_utf8(header_bytes)
        .map_err(|_| "Safetensors header is not valid UTF-8".to_string())?;
    let root = parse_json(&header_str)?;
    let obj = root
        .as_obj()
        .ok_or_else(|| "Safetensors header must be a JSON object".to_string())?;

    let mut total_elements: u64 = 0;
    let mut cat_elems: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut sample_tensors = Vec::new();
    let mut tensor_count = 0u64;
    let mut max_layer = 0u32;

    for (k, v) in obj {
        if k == "__metadata__" {
            continue;
        }
        tensor_count += 1;
        let dtype = v.get("dtype").and_then(|d| d.as_str()).unwrap_or("BF16");
        let shape = v.get("shape").and_then(|s| s.as_arr()).unwrap_or(&[]);
        let mut elems: u64 = 1;
        for dim in shape {
            elems = elems.saturating_mul(dim.as_u64().unwrap_or(1));
        }
        let bpe = match dtype {
            "F32" | "I32" => 4.0,
            "F16" | "BF16" | "I16" => 2.0,
            "F8_E4M3" | "F8_E5M2" | "I8" | "U8" => 1.0,
            _ => 2.0,
        };
        let cat = classify_tensor_name(k);
        total_elements = total_elements.saturating_add(elems);
        *cat_elems.entry(cat).or_insert(0) += elems;
        if let Some(rest) = k.split("layers.").nth(1) {
            if let Some(num_s) = rest.split('.').next() {
                if let Ok(n) = num_s.parse::<u32>() {
                    max_layer = max_layer.max(n + 1);
                }
            }
        }
        if sample_tensors.len() < 24 {
            sample_tensors.push(TensorSummary {
                name: k.clone(),
                category: cat.into(),
                ggml_type: dtype.into(),
                elements: elems,
                bytes_est: (elems as f64 * bpe) as u64,
            });
        }
    }

    let tot = total_elements.max(1) as f64;
    let pct = |k: &str| (*cat_elems.get(k).unwrap_or(&0) as f64 / tot) * 100.0;
    let is_moe = *cat_elems.get("moe_expert").unwrap_or(&0) > 0;
    let total_params_b = tot / 1e9;

    Ok(ModelSpec {
        id: format!("local-st-{}", path.file_name().unwrap_or_default().to_string_lossy()),
        name: path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Safetensors Model".into()),
        source_format: "safetensors".into(),
        source_path: path.display().to_string(),
        architecture: "safetensors".into(),
        total_params_b,
        active_params_b: if is_moe { total_params_b * 0.12 } else { total_params_b },
        file_size_gb,
        fp16_equiv_gb: (tot * 2.0) / 1_073_741_824.0,
        n_layers: max_layer.max(32),
        hidden_dim: 4096,
        n_heads: 32,
        n_kv_heads: 8,
        head_dim: 128,
        native_context: 131_072,
        is_moe,
        expert_count: if is_moe { 64 } else { 1 },
        active_experts: if is_moe { 8 } else { 1 },
        is_mla: false,
        is_hybrid_gdn: false,
        has_mtp_head: false,
        is_uncensored_base: false,
        tensor_count,
        category_shares: CategoryShares {
            embed_output_pct: pct("embed_output"),
            attention_pct: pct("attention"),
            router_gate_pct: pct("router"),
            ffn_or_experts_pct: pct("moe_expert") + pct("dense_ffn"),
            norms_pct: pct("norm"),
        },
        sample_tensors,
    })
}

/// Inspect a HuggingFace model directory (config.json + safetensors shards or GGUF files).
pub fn inspect_hf_directory(dir: &Path) -> Result<ModelSpec, String> {
    let entries = std::fs::read_dir(dir).map_err(|e| e.to_string())?;
    let mut total_bytes = 0u64;
    let mut first_gguf = None;
    let mut first_st = None;
    let mut config_path = None;

    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_file() {
            if let Ok(m) = std::fs::metadata(&p) {
                total_bytes = total_bytes.saturating_add(m.len());
            }
            if let Some(fname) = p.file_name().and_then(|s| s.to_str()) {
                if fname == "config.json" {
                    config_path = Some(p.clone());
                } else if fname.ends_with(".gguf") && first_gguf.is_none() {
                    first_gguf = Some(p.clone());
                } else if fname.ends_with(".safetensors") && first_st.is_none() {
                    first_st = Some(p.clone());
                }
            }
        }
    }

    if let Some(gg) = first_gguf {
        let mut spec = parse_gguf_file(&gg)?;
        spec.file_size_gb = (total_bytes as f64) / 1_073_741_824.0;
        spec.source_path = dir.display().to_string();
        return Ok(spec);
    }

    if let Some(cfg_p) = config_path {
        let raw = std::fs::read_to_string(&cfg_p).map_err(|e| e.to_string())?;
        let cfg = parse_json(&raw)?;
        let arch = cfg
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("transformer")
            .to_string();
        let n_layers = cfg.get("num_hidden_layers").and_then(|v| v.as_u64()).unwrap_or(32) as u32;
        let hidden_dim = cfg.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(4096) as u32;
        let n_heads = cfg.get("num_attention_heads").and_then(|v| v.as_u64()).unwrap_or(32) as u32;
        let n_kv_heads = cfg
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .unwrap_or(n_heads as u64) as u32;
        let head_dim = cfg
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(if n_heads > 0 { (hidden_dim / n_heads) as u64 } else { 128 }) as u32;
        let native_context = cfg
            .get("max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(131_072) as u32;
        let expert_count = cfg
            .get("num_experts")
            .or_else(|| cfg.get("n_routed_experts"))
            .or_else(|| cfg.get("num_local_experts"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as u32;
        let active_experts = cfg
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .unwrap_or(if expert_count > 1 { 8 } else { 1 }) as u32;
        let is_moe = expert_count > 1;
        let file_size_gb = (total_bytes as f64) / 1_073_741_824.0;
        let est_params_b = (file_size_gb / 2.0).max(1.0);
        let active_params_b = if is_moe {
            est_params_b * (0.08 + 0.92 * (active_experts as f64 / expert_count as f64))
        } else {
            est_params_b
        };
        let samples = if let Some(st) = first_st {
            parse_safetensors_file(&st).map(|s| s.sample_tensors).unwrap_or_default()
        } else if is_moe {
            synthetic_moe_tensors(n_layers, expert_count)
        } else {
            synthetic_dense_tensors(n_layers)
        };
        return Ok(ModelSpec {
            id: format!("local-hf-{}", dir.file_name().unwrap_or_default().to_string_lossy()),
            name: dir
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "HuggingFace Model Dir".into()),
            source_format: "hf_dir".into(),
            source_path: dir.display().to_string(),
            architecture: arch.clone(),
            total_params_b: est_params_b,
            active_params_b,
            file_size_gb,
            fp16_equiv_gb: est_params_b * 2.0,
            n_layers,
            hidden_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            native_context,
            is_moe,
            expert_count,
            active_experts,
            is_mla: arch.contains("deepseek"),
            is_hybrid_gdn: arch.contains("qwen4") || arch.contains("next") || arch.contains("gdn"),
            has_mtp_head: cfg.get("num_nextn_predict_layers").is_some(),
            is_uncensored_base: false,
            tensor_count: (n_layers as u64) * 14,
            category_shares: if is_moe {
                CategoryShares {
                    embed_output_pct: 2.0,
                    attention_pct: 5.5,
                    router_gate_pct: 0.35,
                    ffn_or_experts_pct: 92.0,
                    norms_pct: 0.15,
                }
            } else {
                CategoryShares {
                    embed_output_pct: 6.0,
                    attention_pct: 22.0,
                    router_gate_pct: 0.0,
                    ffn_or_experts_pct: 71.7,
                    norms_pct: 0.3,
                }
            },
            sample_tensors: samples,
        });
    }

    if let Some(st) = first_st {
        return parse_safetensors_file(&st);
    }
    Err("Directory contains no .gguf, .safetensors, or config.json".into())
}
