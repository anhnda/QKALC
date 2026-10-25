from src.attention.geometry import AttnGeometry, expand_kv, get_attn_geometry, project_heads
from src.attention.qk_diag import attention_kl_diagnostics
from src.attention.qk_stats import SeqCache, build_seq_cache, compute_seq_stats, query_mean
from src.attention.rope import compute_rope_tables, rope_apply, rope_apply_T, rope_diag_var, rotate_half

__all__ = [
    "AttnGeometry",
    "expand_kv",
    "get_attn_geometry",
    "project_heads",
    "attention_kl_diagnostics",
    "SeqCache",
    "build_seq_cache",
    "compute_seq_stats",
    "query_mean",
    "compute_rope_tables",
    "rope_apply",
    "rope_apply_T",
    "rope_diag_var",
    "rotate_half",
]
