# QK-ALC: Attention-aware Lattice Correction cho Q projection

Project này phát triển từ repo CLC. Phần lõi của CLC được giữ nguyên làm baseline, và bổ sung bước **sửa lattice của `q_proj` theo proxy KL của softmax** (Q-pass, K cố định), đúng như phương pháp đã chốt.

## Cấu trúc

```
run_qk.py                     entrypoint: base quant -> [CLC] -> [QK correction] -> diag -> PPL / lm-eval
summarize.py                  gom results/*/summary.json thành bảng (markdown/CSV)
src/
  calibration.py              (giữ từ repo) C4 / WikiText-2 calibration, có cache
  io_utils.py                 (giữ từ repo)
  quantization/
    state.py                  (giữ từ repo) IntegerQuantizedTensorState
    base.py                   group-wise RTN + AWQ search (tách từ repo awq.py), ActStats, quantize_linear
  post_correction/
    clc.py                    (giữ nguyên từ repo) CLC gốc: baseline
    qk_alc.py                 MỚI: QKLatticeCorrector, KronCurvature (PSD), AcceptCache
  attention/
    rope.py                   RoPE kiểu rotate_half: P_t, P_t^T, diag(P^T V P), bảng cos/sin của model
    geometry.py               số head / GQA / scaling, kiểm tra phạm vi (không QK-norm, không partial RoPE)
    qk_stats.py               thống kê đủ theo từng query: c_t, v_t, C_t, J, entropy
    qk_diag.py                KL attention trên tập held-out: full / Q-only / K-only / MQ / QF / proxy
  pipeline/sequential.py      pipeline block-sequential (input của mỗi layer lấy từ prefix đã quantize)
  evaluation/                 (giữ từ repo) sliding-window PPL, lm-eval runner
scripts/
  common.sh                   biến chung (MODEL_PATH, BITS, FLIP, ...)
  rtn/run_rtn_ablation.sh     các arm RTN (chạy trước)
  awq/run_awq_ablation.sh     các arm AWQ (chạy sau)
tests/test_qk_math.py         test chính xác (float64, CPU, kích thước nhỏ)
```

Những phần đã bỏ: FlatQuant (`flatquant/`, `src/quantization/flatquant*.py`), bias correction, drift eval, CLI `main.py` cùng các bash wrapper cũ, thư mục `datasets/`.

## Phương pháp (tóm tắt những gì code làm)

Với mỗi layer, sau khi base quantizer chạy xong và giữ K cố định:

- **Objective.** $J=\tfrac12\,\mathbb E_t\,\mathrm{Var}_{s\sim p_t}(\varepsilon_{ts})$, trong đó $p_t$ là attention của teacher FP (Q, K FP trên cùng input $x$), còn $\varepsilon_{ts}=s\,[(P_t\hat q_t)\cdot(P_s\hat k_s)-(P_tq_t)\cdot(P_sk_s)]$. Query ở vị trí $< \text{exclude\_first}$ (mặc định là vị trí 0) không được lấy mẫu.
- **Thống kê đủ** (`qk_stats.py`). Với $Y_{ts}=P_t^\top P_s\hat k_s$:
  - $c_t=\mathrm{Cov}_{p_t}(\varepsilon,Y)$
  - $v_t=\mathrm{diag}\,\mathrm{Cov}_{p_t}(Y)$
  - $C_t=\mathrm{Cov}_{p_t}(Y)$, chỉ tính cho một tập con query.
- **Chấm điểm một flip** $(o,i,\delta)$: $\Delta J=\delta G_{oi}+\tfrac12\delta^2H_{oi}$, chính xác vì $J$ bậc hai theo $W_Q$. Ở đây $G=s\,\mathbb E[x\,c^\top]$ và $H=s^2\,\mathbb E[x^2 v^\top]$.
- **Tương tác giữa các flip.** Dùng Kronecker phân tầng theo entropy của $p_t$: $s^2\sum_k\pi_kM_k\otimes\bar C_k$ (damping trên $M_k$). Ma trận này được hiệu chỉnh đường chéo thành $D K D$ để đường chéo khớp đúng $H$ mà vẫn PSD (`KronCurvature`).
- **Chấp nhận.** Batch flip của mỗi head được tính $\Delta J$ **chính xác** trên một tập chấp nhận độc lập, dùng cache $\{x_t,c_t,C_t\}$ (`AcceptCache`). Batch được nhận khi $\Delta J_{\text{pred}}<0$ và $\Delta J_{\mathcal A}<-\text{margin}\cdot J_{\mathcal A}$. Nếu bị bác bỏ thì chia đôi batch; khi batch còn 1 flip mà vẫn bị bác bỏ thì chặn flip đó.
- **Làm mới.** $G$ được tính lại chính xác sau mỗi `--qk-refresh-every` vòng.
- **Ràng buộc lattice.** Chỉ có flip bù sang mức kề bên kia của `pre_round`, mỗi weight tối đa một lần, có budget theo hàng (`--qk-max-flip-percent`). Code luôn nằm trong range, nên định dạng integer và kernel không đổi.

**K-pass** (`src/post_correction/k_alc.py`) dùng cùng objective, nhưng giữ Q cố định và sửa lattice của `k_proj`:

- Với $u_t=P_t\hat q_t$ và $Z_{ts}=P_s^\top u_t$, flip tại $W_K[(g,r),i]$ thay đổi $\varepsilon_{ts}$ một lượng $\delta\,s\,x_{s,i}Z_{ts,r}$, ở **mọi** head thuộc nhóm KV $g$ (GQA).
- Gradient được tính chính xác: $G=\tfrac{s}{N}A_g^\top X$, trong đó $A_s=P_s^\top\sum_t w_{ts}u_t$, cộng qua các head trong nhóm.
- Đường chéo Hessian $H=E_1-E_2$:
  - $E_1$ tính chính xác;
  - $E_2=\sum_t(\sum_s p\,x\,Z)^2$ được ước lượng không chệch trên `--qk-k-hsub-queries` query mỗi chuỗi.
- Tương tác giữa các flip không được mô hình hoá trước. Thay vào đó, bước chấp nhận tính lại $\varepsilon$ **chính xác** trên tập chấp nhận và quyết định nhận hay bỏ theo từng nhóm KV.
- $G$ được làm mới chính xác sau mỗi `--qk-k-refresh-every` vòng.

K-pass dự đoán $\Delta J$ chỉ bằng đường chéo (không có tương tác), nên batch lớn gồm nhiều flip cùng chiều sẽ overshoot. Vì vậy K-pass có bộ tham số batch riêng, mặc định nhỏ: `--qk-k-batch-init 16 --qk-k-batch-max 64 --qk-k-row-cap 1`. Có thể chạy sweep bằng `scripts/rtn/run_kpass_sweep.sh`. Trong bảng tổng hợp, `q_sp` / `k_sp` là median theo layer của Spearman(ΔJ dự đoán, ΔJ chính xác); `q_acc` / `k_acc` là tỉ lệ batch được nhận; `q_bs` / `k_bs` là số flip trung bình mỗi batch.

Thứ tự các pass do `--qk-schedule` quyết định, ví dụ `Q`, `K`, `QK`, `QKQ`. Pass sau tính lại thống kê theo trọng số mới nhất của pass trước. Một weight đã flip ở pass trước thì bị khoá, và vẫn bị tính vào budget của hàng đó. Q-pass và K-pass dùng cùng các vị trí query cho cùng seed, nên $J$ của chúng so sánh trực tiếp được.

Các chế độ:

| Flag | Ý nghĩa |
|---|---|
| `--qk-correct full` | Query thật (chế độ mặc định của phương pháp) |
| `--qk-correct meanquery` | Thay query lỗi bằng $\mu$ (ablation "mean-guided"; $M=\mu\mu^\top$) |
| `--qk-accept exact` / `none` | Có / không dùng tập chấp nhận độc lập |
| `--qk-strata 1` | K-FAC thường (không phân tầng) |
| `--qk-audit-every N` | Cứ N vòng, tính thêm $\Delta J$ chính xác trên tập tối ưu cho batch, để đo độ chính xác của dự đoán Kronecker |

Chia dữ liệu calibration (cùng pool C4, $n=128$ × 2048 token):

- chuỗi `[0, qk_opt_seqs)` dùng cho thống kê tối ưu;
- chuỗi `[qk_opt_seqs, qk_opt_seqs+qk_accept_seqs)` làm tập chấp nhận;
- AWQ/CLC dùng toàn bộ pool;
- diagnostics dùng các chunk WikiText-2 **test** (held-out).

## Cài đặt

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124   # chọn theo CUDA
```

Model private: đặt `HF_TOKEN=...` trong `.env`.

## Kiểm tra toán học trước (CPU, vài giây)

```bash
python -m pytest tests/ -v
```

Bộ test kiểm tra:

- Các đẳng thức RoPE so với ma trận quay tường minh.
- $J$ so với một cài đặt brute-force độc lập (cả full và mean-query).
- $\Delta J$ của một flip đúng bằng $\delta G+\tfrac12\delta^2H$.
- $\Delta J$ của batch qua `AcceptCache` là chính xác, và cập nhật $c_t$ cũng chính xác.
- `KronCurvature` là PSD, có đường chéo đúng bằng $H$, và matvec khớp dạng dense; trường hợp $C_t$ hằng thì Kronecker bằng đúng Hessian thật.
- Các bất biến của corrector: chỉ flip ±1 theo đúng hướng bù, không vượt budget, và $J_{\mathcal A}$ không tăng, khớp khi tính lại từ đầu.

## Thí nghiệm: RTN trước

```bash
MODEL_PATH=meta-llama/Meta-Llama-3-8B BITS=3 bash scripts/rtn/run_rtn_ablation.sh
MODEL_PATH=meta-llama/Meta-Llama-3-8B BITS=4 bash scripts/rtn/run_rtn_ablation.sh
# chỉ chạy một số arm:
ARMS=rtn,rtn_clcq,rtn_qk BITS=3 bash scripts/rtn/run_rtn_ablation.sh
```

| Arm | Mục đích (bước trong kế hoạch) |
|---|---|
| `fp` | Tham chiếu |
| `rtn`, `rtn_clcq`, `rtn_clc` | **Bước 1 (go/no-go)**: `diag_before` cho `kl_full`, `kl_q_only`, `kl_k_only`, `kl_mq`, `kl_qf`, `kl_interaction` |
| `rtn_qk` | **Bước 2**: phương pháp chính, so trực tiếp với `rtn_clcq` ở cùng `FLIP` |
| `rtn_clc_qk`, `rtn_clcnq_qk` | QK chạy sau CLC-all / CLC trên mọi layer trừ q |
| `rtn_qk_mq`, `rtn_qk_mq_acc` | Mean-query vs query thật (đo giá trị của việc tính đến QF) |
| `rtn_qk_noacc` | Có/không tập chấp nhận (đo winner's curse) |
| `rtn_qk_k1` vs `rtn_qk_k3audit` | Kronecker $K=1$ vs $K=3$ (cột `audit_pearson` trong bảng) |
| `rtn_k`, `rtn_qkq`, `rtn_kq` | K-pass riêng; Q→K→Q; K→Q |
| `rtn_clcnq_qkq` | CLC cho mọi layer trừ q, sau đó Q→K→Q |
| `rtn_qk_acc64`, `rtn_qk_acc1024` | Kích thước tập chấp nhận |
| `rtn_qk_f02`, `rtn_qk_f10` | Flip budget |

Llama-3-8B (cấu hình đã chốt từ thí nghiệm trên 1B), chạy theo giai đoạn:

```bash
PHASE=smoke bash scripts/llama3_8b/run_8b.sh                 # layer 0-1, kiểm tra API/OOM và k_sp > 0
PHASE=rtn BITS_LIST="3 4" bash scripts/llama3_8b/run_8b.sh
PHASE=awq BITS_LIST="3 4" bash scripts/llama3_8b/run_8b.sh
```

Sau đó chạy AWQ:

```bash
MODEL_PATH=meta-llama/Meta-Llama-3-8B BITS=3 bash scripts/awq/run_awq_ablation.sh
```

Chạy lẻ một lần:

```bash
python run_qk.py --model-path meta-llama/Meta-Llama-3-8B --base rtn --bits 3 \
  --qk-correct full --diag --run-name rtn_qk_test --qk-layers 0-3 --no-eval-ppl
```

## Output

`results/<...>/<run_name>/` gồm:

- `config.json`: toàn bộ tham số.
- `layers.json`: thông tin theo từng layer:
  - flips/alpha của mỗi Linear;
  - `diag_before` / `diag_after` (mean, max và giá trị từng head của mỗi loại KL);
  - `qk`: `J_opt_before/after`, `J_acc_before/after` theo head, `flips_per_head`, `rounds`, các strata, `history` (pred vs acc), thời gian, peak memory.
- `summary.json`: PPL (WikiText-2, C4 tuỳ chọn), lm-eval, cùng các số tổng hợp: KL trung bình qua các layer (trước/sau), tổng flip, thời gian và bộ nhớ của QK.

Gom kết quả: `python summarize.py --results-dir results/rtn/Meta-Llama-3-8B_w3g128 --csv rtn_w3.csv`

## Ghi chú và giới hạn

- **Phạm vi:** Llama, Mistral, Qwen2 (RoPE kiểu rotate_half, có GQA, bias q/k). Model có QK-norm hoặc partial RoPE sẽ báo `NotImplementedError`. Sliding window không được mô hình hoá (chỉ cảnh báo).
- **Teacher là cục bộ theo layer:** $p_t$ tính từ Q, K FP trên **cùng** input $x$ lấy từ prefix đã quantize.
- **AWQ là biến thể của repo:** scale tìm riêng cho từng Linear, weight lưu dạng fake-quant; scale **không** được fold vào norm. Lattice của QK nằm trong miền đã scale: $\delta=\pm\text{scale}/s_i$. Muốn deploy với kernel thật thì cần fold scale; khi đó Q, K, V phải dùng chung scale. Đây là giới hạn có sẵn từ repo gốc.
- **Toàn bộ lattice tính ở float32.** Repo cũ tạo state ở bf16, khiến `pre_round` quá thô.
- **Bộ nhớ cho Llama-3-8B** (mặc định 64 chuỗi tối ưu, 16 chuỗi chấp nhận):
  - cache key khoảng 1 GB;
  - acceptance cache (512 query × 32 head × 128²) khoảng 1 GB;
  - $M_k$: 3 × 64 MB.

  Nếu thiếu bộ nhớ: dùng `--inps-device cpu`, giảm `--qk-opt-seqs`, `--qk-accept-queries-per-seq` hoặc `--qk-strata`.
- **TF32 tắt mặc định.** Thêm `--allow-tf32` để tăng tốc phép GEMM trong lúc correction.
- **Code mới chỉ được kiểm tra bằng cách đọc lại**, chưa chạy lần nào. Hãy chạy `pytest` trước, rồi thử một lần với `--qk-layers 0-1 --no-eval-ppl` để bắt lỗi API của transformers trước khi chạy cả bộ.
