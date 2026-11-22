# Qwen2-VL + Layered Gated Cross-Attention

## Mục đích

Kết quả chính của khóa luận (LLaVA-Med) cho thấy cross-attention thắng nối-token
+0.865 BERTScore-F1 — nhưng LLaVA-Med (Mistral-7B) có rào cản tiếng Việt rất nặng,
nên phép so sánh đó gộp chung 2 hiệu ứng: "sửa một base model bị hỏng" và
"chất lượng kiến trúc hợp nhất thị giác". Qwen2-VL là phép thử sạch để tách riêng
hiệu ứng thứ 2: nó đã thành công với nối-token (zero-shot BERTScore 0.86, sau LoRA
0.91 trên cùng tập tiếng Việt — không có rào cản ngôn ngữ cần sửa). Nếu
cross-attention **cũng** cải thiện thêm trên một base model đã tốt sẵn, đó là bằng
chứng mạnh rằng bản thân kiến trúc hợp nhất tốt hơn, không chỉ là cơ chế "cứu"
base model yếu.

## Thiết kế

So với Qwen2-VL gốc (patch ảnh sau ViT+PatchMerger được nối thẳng vào chuỗi input
tại vị trí token `<|image_pad|>`), file `modeling_qwen2vl_xattn.py` rút token ảnh ra
khỏi chuỗi tự-chú-ý hoàn toàn, tái tiêm qua `GatedCrossAttention` ở nhiều tầng decoder —
copy nguyên thiết kế từ `llava_mistral_xattn.py`:

- 1/3 tầng dưới: chỉ self-attention (không cross-attn)
- 1/3 tầng giữa: cross-attn thưa (top-k), mỗi 2 tầng
- 1/3 tầng trên: cross-attn dày, mọi tầng
- gate cố định α=0.05 + zero-init `out_proj` (giữ nguyên lý do chống vanishing-gradient
  đã ghi trong `cross_attention.py`)

Khác biệt so với LLaVA: `vision_dim == text_dim` (PatchMerger của Qwen2-VL đã chiếu ảnh
vào đúng không gian hidden của decoder), nên cấu trúc cross-attn dựng được ngay trong
`__init__` từ config, không cần bước "build sau khi load vision tower" như LLaVA. Cũng
**không** dùng `VisualPositionEmbedding` 2D học được (như LLaVA) vì ViT của Qwen2-VL đã
tự áp rotary position embedding cho patch trước khi merge — thêm 1 positional embedding
cố định-lưới sẽ xung đột với độ phân giải ảnh thay đổi theo từng ảnh (không cố định
576 patch như CLIP).

## Dữ liệu — dùng lại nguyên, không cần tạo mới

Dùng thẳng đúng 3 file JSONL đã dùng cho LoRA v1/v2/v3:

```
bone_data/dataset_train_minimal.jsonl
bone_data/dataset_val_minimal.jsonl
bone_data/dataset_test_minimal.jsonl
```

`dataset.py` ở đây đọc đúng schema `{"image", "question", "answer"}` — schema này
giống chính xác với `train_xattn.py` (script train XAttn gốc cho LLaVA-Med) và
`train_lora_baseline*.py`, nên không cần convert/tiền xử lý gì thêm. Khác biệt duy
nhất là **cách dùng** dữ liệu: ảnh được xử lý riêng qua `processor.image_processor`
(không qua `apply_chat_template` với content-block ảnh), để chuỗi token đưa vào
decoder không chứa token ảnh — xem phần Thiết kế ở trên.

## ⚠️ Rủi ro / chưa kiểm thử — đọc trước khi chạy

Code được viết bằng cách đọc trực tiếp `modeling_qwen2_vl.py` của `transformers==5.7.0`
(cài local) + tham khảo `train_lora_qwenvl.py` (đã chạy thành công trên server, env
`qwenvl` riêng — **không phải** env `llava`). Tôi **không có quyền truy cập env `qwenvl`
trên server** nên 2 điểm sau chỉ được xử lý phòng hờ (defensive), chưa xác nhận đúng với
version thật trên server:

1. Đường dẫn `model.model.language_model.layers` (transformers mới) vs
   `model.model.layers` (transformers cũ, ~4.45-4.49 khi Qwen2-VL mới ra) — xử lý ở
   `_get_text_decoder()`.
2. `self.visual(pixel_values, grid_thw=...)` trả về tensor trực tiếp (cũ) vs trả về
   `ModelOutput` có `.pooler_output` (mới) — xử lý ở `extract_visual_tokens()`.
3. `Qwen2VLDecoderLayer` kế thừa `GradientCheckpointingLayer` (cơ chế checkpoint nội
   bộ riêng của transformers cho model multimodal) — gọi
   `model.gradient_checkpointing_enable()` như các script LoRA vẫn làm *có thể* không
   tương thích 100% với forward hook (cách cross-attn được tái tiêm). Nếu smoke test
   báo lỗi liên quan `checkpoint`/`recompute`, thử bỏ dòng
   `gradient_checkpointing_enable()` trong `train_qwenvl_xattn.py` trước (đổi lại VRAM
   sẽ tốn hơn, có thể cần giảm `--max_length` hoặc dùng `--max_pixels` nhỏ hơn).

**Luôn chạy `--smoke_test` trước khi train thật** — 1 sample, 1 forward+backward, không
qua Trainer, in ra loss + gradient norm của cross-attn rồi thoát. Nếu lỗi
AttributeError/TypeError, paste lại đây, tôi sửa theo đúng version thật trên server (giống
cách đã sửa `train_lora_baseline_v2/v3` qua vài vòng debug).

## Cách chạy

### 0. Đẩy cả folder lên server

```bash
scp -r qwenvl_xattn user@ec2-host:/home/ubuntu/workspaces/KLTN/KhoaLuanTotNghiep/
```

Không cần đẩy thêm gì khác — `qwenvl_xattn/` không phụ thuộc package `llava`, tự
chứa toàn bộ code cần thiết (`cross_attention.py`, `modeling_qwen2vl_xattn.py`,
`dataset.py`). Dữ liệu (`bone_data/...`) và `compute_metrics.py` đã có sẵn trên
server từ các lần chạy LoRA trước, không cần đẩy lại.

### Các bước

Chạy trong env `qwenvl` (không phải `llava`):

```bash
cd qwenvl_xattn

# 1. Smoke test — LUÔN chạy trước
python train_qwenvl_xattn.py --smoke_test \
    --base_model Qwen/Qwen2-VL-7B-Instruct \
    --data_path  ../bone_data/dataset_train_minimal.jsonl \
    --image_root ../bone_data

# 2. Train thật (Stage-1-only: đóng băng tất cả trừ cross-attn)
python train_qwenvl_xattn.py \
    --base_model Qwen/Qwen2-VL-7B-Instruct \
    --data_path      ../bone_data/dataset_train_minimal.jsonl \
    --val_data_path  ../bone_data/dataset_val_minimal.jsonl \
    --image_root     ../bone_data \
    --output_dir     ./outputs_qwenvl_xattn_minimal \
    --num_epochs 3

# 3. Inference
python run_inference_qwenvl_xattn.py \
    --model_path  ./outputs_qwenvl_xattn_minimal \
    --base_model  Qwen/Qwen2-VL-7B-Instruct \
    --data_path   ../bone_data/dataset_test_minimal.jsonl \
    --image_root  ../bone_data \
    --output_file ../results/outputs/qwenvl_xattn_minimal_outputs.jsonl

# 4. Metric (từ thư mục gốc repo)
cd ..
python compute_metrics.py \
    --files  results/outputs/qwenvl_xattn_minimal_outputs.jsonl \
             results/outputs/lora_qwenvl_minimal_outputs.jsonl \
    --names  "QwenVL_XAttn_Minimal" "QwenVL_LoRA_Minimal" \
    --output_file results/metrics/qwenvl_xattn_vs_lora_metrics.json \
    --bert_model  roberta-large
```

## Cách đọc kết quả

So sánh trực tiếp với `LoRA-QwenVL-Minimal` đã có (BERTScore-F1 0.9099,
`results/all_metrics.json`):

- **Nếu XAttn > LoRA trên Qwen2-VL** (dù chỉ nhích nhẹ): bằng chứng mạnh cho luận điểm
  "cross-attention tốt hơn nối-token một cách tổng quát, độc lập với base model" — đáng
  viết thành 1 mục mới trong chương 4 (so sánh thuần kiến trúc, kiểm soát base model).
- **Nếu XAttn ≈ LoRA hoặc kém hơn**: không phủ nhận đóng góp chính (XAttn vẫn thắng rõ
  trên LLaVA-Med), nhưng thu hẹp phạm vi tuyên bố lại thành "cross-attention đặc biệt có
  giá trị khi base model có rào cản ngôn ngữ/domain — chưa chắc còn lợi thế khi base
  model đã rất mạnh sẵn". Đây vẫn là một phát hiện hợp lệ, chỉ cần diễn đạt khác.

Cả 2 kết quả đều dùng được cho luận văn — không có kết quả "thất bại" ở thí nghiệm này.
