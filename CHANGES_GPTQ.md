# Thay đổi: thêm GPTQ làm base quantizer

## File mới

- **`src/quantization/gptq.py`**
  - `GPTQConfig` và `gptq_quantize(w, H, bits, group_size, cfg)`, trả về `IntegerQuantizedTensorState`.
  - Tính bằng fp32, có lazy-batch blocks, act-order, và hai chế độ group (static/dynamic).
  - `pre_round` = giá trị đã bù lỗi; `float_weights` = W gốc.
  - Damping tự tăng ×10 nếu Cholesky lỗi.
  - Kênh input chết (năng lượng bằng 0) giữ nguyên weight.
- **`tests/test_gptq.py`** gồm các test:
  - H chéo ⇒ kết quả trùng RTN từng bit;
  - khớp bản OBS tuần tự chạy ngây thơ;
  - kết quả không đổi theo blocksize;
  - các bất biến của lattice;
  - output error thấp hơn RTN;
  - GPTQ + CLC làm giảm mean-shift thật.
- **`scripts/gptq/run_gptq_ablation.sh`**: các arm GPTQ cho 1B/8B.

## File sửa

- **`src/quantization/base.py`**
  - `ActStats(hessian=True)` cộng dồn XᵀX và có `second_moment()`.
  - `quantize_linear(..., method="gptq", gptq_cfg=...)`.
- **`src/pipeline/sequential.py`**
  - `PipelineConfig` có thêm `gptq_*` và `true_sequential`.
  - Gom Linear thành các nhóm `[qkv] → (diag + QK) → [o] → [gate,up] → [down]`.
  - Với GPTQ, Linear dùng chung input thì dùng chung Hessian.
  - Bước diag/QK được tách thành `_attention_stage`.
  - RTN/AWQ chạy một nhóm như cũ, kết quả không đổi.
- **`run_qk.py`**
  - Thêm `--base gptq`, `--gptq-damp`, `--gptq-blocksize`, `--[no-]gptq-act-order`, `--[no-]gptq-static-groups`, `--[no-]gptq-true-sequential`.
  - Thêm `--diag-split {test,validation}`, mặc định `test` như trước.
  - Tên run có hậu tố `aoXsgXtsX`.
  - `summary.json` có thêm `gptq_loss_total` và `gptq_damp_max`.
- **`scripts/llama3_8b/run_8b.sh`**: thêm `PHASE=gptq` và `PHASE=gsmoke`.
- **`scripts/llama3_8b/run_8b_final.sh`**:
  - `STEP=6`: GPTQ W3/W4, seeds `42 1 2`, có PPL 8K.
  - `STEP=7`: tổng hợp seed và paired diff.
- **`README.md`**: thêm mục "Thí nghiệm: GPTQ làm base (T3)".

## Thứ tự chạy gợi ý

1. `python -m pytest tests/ -v`
2. `PHASE=gsmoke bash scripts/llama3_8b/run_8b.sh`
3. `MODEL_PATH=meta-llama/Llama-3.2-1B BITS=3 EXTRA_ARGS="--diag-split validation" bash scripts/gptq/run_gptq_ablation.sh`
4. `STEP=6 bash scripts/llama3_8b/run_8b_final.sh`, sau đó `STEP=7`.
