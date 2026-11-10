# QK-ALC trên nền GPTQ

Tài liệu này tóm tắt phần **GPTQ base** mới thêm vào QK-ALC: đã cài những gì, chạy thế nào, và còn những việc gì phải làm. Mục tiêu là trả lời câu hỏi T3 trong bài: *trên một base mạnh hơn RTN/AWQ, Q-pass/K-pass có còn bổ sung được không?*

> **Trạng thái:** code mới chỉ được kiểm tra bằng cách đọc lại, **chưa chạy lần nào**, kể cả test. Hãy chạy theo đúng thứ tự ở mục 3.

---

## 1. Những gì đã cài

### 1.1. GPTQ trên cùng lattice (`src/quantization/gptq.py`)

`gptq_quantize(w, H, bits, group_size, cfg)` trả về một `IntegerQuantizedTensorState`, giống hệt kiểu RTN/AWQ trả về. Vì vậy CLC, Q-pass và K-pass chạy trên GPTQ mà không phải sửa gì.

- **Thuật toán:** GPTQ chuẩn: Cholesky của (H + λI)⁻¹, xử lý theo lazy-batch blocks, act-order (theo thứ tự diag(H) giảm dần), group động hoặc tĩnh. Mọi phép tính ở fp32.
- **Lưới lượng tử:** min-max bất đối xứng theo group, **cùng công thức** với `groupwise_quantize` của RTN/AWQ. Khác GPTQ gốc ở chỗ range không bị ép chứa số 0, để ba base dùng chung một định nghĩa lattice.
- **Damping:** λ = `damp · mean(diag H)`. Nếu Cholesky lỗi thì tự tăng ×10, tối đa 6 lần.
- **Kênh input chết** (năng lượng calib bằng 0): **giữ nguyên weight**, chỉ làm tròn kiểu RTN và không lan lỗi sang cột khác. GPTQ gốc thì đặt weight của các kênh này về 0, làm model thay đổi khi kênh đó khác 0 lúc eval.
- **Group động nằm vắt qua ranh giới block:** được dựng từ các cột đã cập nhật chính xác, kể cả phần lazy update còn treo.

Ý nghĩa các trường của lattice:

| Trường | GPTQ | RTN / AWQ |
|---|---|---|
| `pre_round` | w′/scale + zp, với w′ là weight **đã được bù lỗi** tại lúc làm tròn cột đó | w/scale + zp |
| `integer_weights` | mức gần nhất với `pre_round` | như vậy |
| `float_weights` | **W gốc** | W (AWQ: đã nhân input scale) |
| `scale`, `zero_point` | lưu theo từng phần tử (act-order + group động thì group không liền nhau, deploy cần `g_idx`) | như vậy |
| `in_scales` | 1 | 1 (RTN) / scale AWQ |

Hệ quả:

- **Flip bù** (Q-pass, K-pass, CLC) là chuyển sang mức kề bên kia của `pre_round`, giữ đúng ngữ nghĩa như với RTN/AWQ.
- **CLC trên GPTQ** sửa đúng mean-shift thật μᵀ(W − Ŵ) ở đầu ra, vì nó lấy `float_weights` = W gốc.
- **Flip sau GPTQ không cập nhật lại phần bù** GPTQ đã tính cho các cột sau. Với W_Q, đầu ra chỉ đi vào logit và J được tính chính xác trên toàn bộ Ŵ, nên vẫn hợp lý. Với CLC trên mọi layer thì đây là một điểm yếu, cần ghi rõ khi báo cáo.

### 1.2. Thống kê activation (`src/quantization/base.py`)

- `ActStats(hessian=True)` cộng dồn XᵀX ở fp32. `second_moment()` trả về E[xxᵀ].
- `quantize_linear(..., method="gptq", gptq_cfg=GPTQConfig(...))`. Khi có CLC, mean activation được chia cho `in_scales` (bằng 1 với GPTQ), như cũ.

### 1.3. Pipeline (`src/pipeline/sequential.py`)

- `PipelineConfig` có thêm các trường `gptq_damp`, `gptq_blocksize`, `gptq_act_order`, `gptq_static_groups`, `true_sequential`.
- **True-sequential** (mặc định bật cho GPTQ). Trong mỗi block:
  ```
  [q,k,v] → diag_before → QK correction (Q/K pass) → diag_after
          → [o_proj] → [gate,up] → [down] → propagate
  ```
  Input của từng nhóm được thu lại qua phần block đã lượng tử, nên `o_proj` được lượng tử trên attention **đã sửa QK**.
- **Hessian dùng chung:** q/k/v dùng chung một E[xxᵀ], gate/up cũng vậy. Chỉ áp dụng khi base là GPTQ.
- **RTN/AWQ** chạy một nhóm như cũ. Hook vẫn kích hoạt theo thứ tự thực thi, nên RNG của AWQ không đổi và kết quả cũ **tái lập được bit-for-bit**.
- Phần diag và QK được tách thành `_attention_stage()`, logic không đổi.

### 1.4. CLI (`run_qk.py`)

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--base gptq` | – | Dùng GPTQ làm base |
| `--gptq-damp` | 0.01 | Damping tương đối |
| `--gptq-blocksize` | 128 | Kích thước lazy-batch block |
| `--[no-]gptq-act-order` | bật | Xử lý cột theo diag(H) giảm dần |
| `--[no-]gptq-static-groups` | tắt | Tham số group lấy từ W gốc, group liền nhau (kernel-friendly) |
| `--[no-]gptq-true-sequential` | bật | Lượng tử theo thứ tự [qkv]→QK→[o]→[gate,up]→[down] |
| `--diag-split {test,validation}` | test | Split WikiText-2 dùng cho diagnostics KL. **Dùng `validation` khi sweep** |

- Tên run tự sinh có thêm hậu tố `aoXsgXtsX`.
- `summary.json → layers` có thêm `gptq_loss_total` và `gptq_damp_max`.
- `layers.json` lưu cho từng Linear: `gptq_loss`, `gptq_damp`, `gptq_dead`.

### 1.5. Test (`tests/test_gptq.py`, chạy trên CPU)

| Test | Kiểm tra |
|---|---|
| `test_diagonal_hessian_reduces_to_rtn` | H chéo ⇒ codes, scale, zp trùng RTN từng bit |
| `test_matches_naive_obs_reference` | Khớp bản OBS tuần tự cài trực tiếp, có và không có act-order |
| `test_lazy_batch_blocksize_invariance` | Đổi blocksize 1/7/16/64 không làm đổi kết quả |
| `test_gptq_beats_rtn_on_correlated_inputs` | Output error thấp hơn RTN, với 5 biến thể |
| `test_lattice_invariants` | codes = round(pre_round), nằm trong range, float_weights = W gốc, cấu trúc group đúng |
| `test_dead_channels_keep_weights_and_are_rtn_rounded` | Kênh chết được xử lý đúng |
| `test_non_divisible_in_features` | in_features không chia hết cho group size |
| `test_actstats_second_moment` | XᵀX/n đúng |
| `test_quantize_linear_gptq_and_clc` | weight của module bằng weight hiệu dụng của lattice; CLC giảm \|μᵀ(W−Ŵ)\| trên từng hàng |

### 1.6. Script

| Script | Nội dung |
|---|---|
| `scripts/gptq/run_gptq_ablation.sh` | Arm GPTQ cho một model bất kỳ (1B để dev) |
| `scripts/llama3_8b/run_8b.sh` | `PHASE=gsmoke` (layer 0–1) và `PHASE=gptq` (W3/W4) |
| `scripts/llama3_8b/run_8b_final.sh` | `STEP=6`: GPTQ nhiều seed + 8K; `STEP=7`: tổng hợp seed và paired diff |

Các arm chính:

| Arm | Nội dung |
|---|---|
| `gptq` | Base |
| `gptq_clcq` | CLC trên W_Q |
| `gptq_qk` | Q-pass |
| `gptq_qkq` | Q→K→Q |
| `gptq_clc` | CLC trên mọi Linear |
| `gptq_clcnq_qk` | CLC\q + Q |
| `gptq_clcnq_qkq` | CLC\q + QKQ |
| `gptq_qk_noacc`, `gptq_qk_f02` | Ablation (chỉ có trong script ablation) |

---

## 2. Cài đặt

```bash
unzip QKALC-gptq.zip && cd QKALC-gptq
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124   # đổi theo CUDA
echo "HF_TOKEN=hf_xxx" > .env
```

---

## 3. Chạy (theo thứ tự)

```bash
# 1. Test toán (CPU, vài giây)
python -m pytest tests/ -v

# 2. Smoke 8B: GPTQ + CLC\q + QKQ, layer 0-1, không tính PPL
#    Cần thấy: không lỗi, mỗi layer in gptq_loss, KL_full giảm
PHASE=gsmoke bash scripts/llama3_8b/run_8b.sh

# 3. Dev trên 1B: KL đo trên validation, không đụng test
MODEL_PATH=meta-llama/Llama-3.2-1B BITS=3 \
EXTRA_ARGS="--diag-split validation --diag-seqs 8" \
bash scripts/gptq/run_gptq_ablation.sh
#    (BITS=4 cho W4; ARMS=gptq,gptq_clcq,gptq_qk để chạy một phần)

# 4. Bảng chính 8B: W3/W4, seeds 42 1 2, có PPL 8K
STEP=6 bash scripts/llama3_8b/run_8b_final.sh
STEP=7 bash scripts/llama3_8b/run_8b_final.sh      # mean ± std + paired diff

# 5. lm-eval zero-shot cho các arm GPTQ (W3, seed 42)
SEED=42 ROOT_OUT=results/llama3_8b_final/lmeval BITS_LIST=3 PHASE=gptq \
ARMS=gptq,gptq_clcq,gptq_qk,gptq_clc,gptq_clcnq_qk,gptq_clcnq_qkq \
LM_EVAL_TASKS="arc_easy arc_challenge hellaswag piqa winogrande" \
EXTRA_ARGS="--lm-eval-batch-size 16" \
bash scripts/llama3_8b/run_8b.sh
```

Chạy lẻ một lần để debug:

```bash
python run_qk.py --model-path meta-llama/Meta-Llama-3-8B --base gptq --bits 3 \
  --clc-targets all_but_q --qk-correct full --qk-schedule QKQ --qk-strata 1 \
  --diag --diag-seqs 8 --ppl-extra-lengths 8192 --run-name gptq_clcnq_qkq_test
```

Biến môi trường hay dùng:

| Biến | Ví dụ | Tác dụng |
|---|---|---|
| `EXTRA_ARGS` | `"--inps-device cpu"` | GPU < 40 GB |
| `GPTQ_ARGS` | `"--gptq-static-groups"`, `"--no-gptq-act-order"` | Đổi biến thể GPTQ |
| `GPTQ_SEEDS` | `"42"` | Seed cho STEP=6/7 (mặc định `42 1 2`) |
| `GPTQ_BITS` | `"3"` | Số bit cho STEP=6/7 (mặc định `3 4`) |
| `ARMS` | `gptq,gptq_qk` | Chỉ chạy các arm này |

Ghi chú:

- `run_8b.sh` tự thêm `--qk-strata 1 --diag-seqs 8` cho 8B.
- Kết quả nằm ở `results/llama3_8b_final/s<seed>/gptq_w<bits>g128/<arm>/summary.json`, cùng `summary.csv` tổng hợp trong thư mục cha.
- **Bộ nhớ trên 8B:** E[xxᵀ] của `down_proj` khoảng 0.8 GB fp32, Cholesky cần thêm 2–3 GB tạm. Ở chế độ true-sequential, mỗi lúc chỉ giữ Hessian của một nhóm.

---

## 4. TODO

### A. Chạy và kiểm tra GPTQ (làm ngay)
- [ ] Chạy `pytest`, sửa lỗi nếu có. Code chưa từng được chạy.
- [ ] `PHASE=gsmoke`: kiểm tra API transformers 4.45, OOM, `gptq_loss` hợp lý, KL giảm.
- [ ] **Sanity check base:** so PPL của `gptq` W4/W3 trên Llama-3-8B với số GPTQ đã công bố, sau khi quy về cùng protocol (stride, dtype, g128, act-order). Lệch nhiều thì xem lại damping, range không chứa 0 và true-sequential.
- [ ] Dev trên 1B (`--diag-split validation`): GPTQ+Q có còn hơn GPTQ+CLC-q không? Mất bao nhiêu flip? Tỉ lệ nhận batch ra sao?
- [ ] 8B nhiều seed (STEP=6/7), rồi lm-eval (bước 5).
- [ ] Ablation biến thể GPTQ: `--gptq-static-groups`, `--no-gptq-act-order`, `--no-gptq-true-sequential` (QK chạy sau khi đã lượng tử cả block).

### B. Quy trình đánh giá (từ phần rà soát với CLC)
- [ ] PPL đang dùng **stride 2048, eval bf16**, còn CLC dùng **stride 512, eval fp16** (lưu bf16 rồi load lại fp16). Cần thêm `--eval-dtype fp16` (gọi `model.to(torch.float16)` trước khi eval) và chạy thêm một bộ với `--ppl-stride 512` nếu muốn so với số của CLC.
- [ ] Bật `--eval-c4` và preset task `extended` của CLC (9 task) nếu muốn bảng có cùng định dạng với CLC.
- [ ] Kiểm tra lại các hiệu W4 nhỏ (cỡ 0.003–0.009) dưới eval fp16, vì nhiễu bf16 cùng cỡ.

### C. Dữ liệu và độ tin cậy thống kê
- [ ] Mọi sweep hay lựa chọn cấu hình chuyển sang `--diag-split validation`. Ghi trong bài rằng cấu hình 1B trước đây được chọn trên WikiText-2 test.
- [ ] Ghi rõ: seed chỉ đổi **offset** trong cùng 128 document đầu của shard C4 `00000`, không đổi tập document. Có thể sửa loader để bốc document ngẫu nhiên theo seed.
- [ ] Tập chấp nhận được dùng lại hàng trăm lần và cả AWQ/CLC/GPTQ đều đã thấy nó, nên `J_acc`, ρ và tỉ lệ nhận bị lệch lạc quan. Thêm khoảng 16 chuỗi C4 nằm ngoài pool calib để đo J sạch.
- [ ] Mở `config.json` của các run 8B cũ để xác nhận `qk_opt_seqs=64` và `qk_strata=1` đúng như bài ghi.
- [ ] Kiểm định ghép cặp cho lm-eval (McNemar / paired bootstrap) và thêm seed (T1).

### D. Hướng nghiên cứu tiếp
- [ ] **So với BoA / TurboBoA** trên cùng protocol. Đây là phép so quyết định chỗ đứng của bài.
- [ ] **Cách 2: đưa objective QK vào chính GPTQ** cho W_Q, thay vì sửa sau:
  - Với K=1, H ≈ s²(C̄ ⊗ M). Bước bù của GPTQ không đổi (C̄ triệt tiêu), chỉ đổi **cách làm tròn các hàng trong cùng cột**: một bài toán lattice cỡ d_h với metric C̄ (Babai hoặc làm tròn tuần tự theo Cholesky của C̄).
  - Số hạng tuyến tính G (bù lỗi của K đã lượng tử) đưa vào bằng cách dịch target: W* = W − (C̄⁻¹ ⊗ M⁻¹)g. Cần lượng tử k_proj trước q_proj.
  - Đọc kỹ BoA trước khi làm, vì hướng này gần với BoA.
- [ ] Re-GPTQ phần còn lại sau mỗi batch flip được nhận, để giữ phần bù của GPTQ. Có thể làm cho W_Q: chạy lại GPTQ trên các cột chưa bị khoá.
- [ ] Mở rộng sang Mistral-7B-v0.3 (model tham chiếu của CLC), Llama-3.1-8B, Qwen2.5-7B (T4).
- [ ] AWQ deploy được: fold scale vào norm, Q/K/V dùng chung scale (T9).
