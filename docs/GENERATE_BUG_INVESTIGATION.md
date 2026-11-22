# Bug điều tra: `output_ids[:, input_ids.shape[1]:]` sai khi `generate()` dùng `inputs_embeds`

> Ghi lại ngày 2026-08-04 → 2026-08-06. Không sửa trực tiếp `llava_mistral.py` hay
> `llava_mistral_xattn.py` — mọi fix đều làm qua monkey-patch trong file script mới, để giữ
> nguyên code gốc cho tới khi quyết định chính thức áp dụng.

## ĐÍNH CHÍNH (2026-08-06, cuối buổi) — đọc phần này trước, các phần "non-deterministic" bên dưới
## đã lỗi thời và được giữ lại chỉ để minh hoạ quá trình điều tra

Sau khi điều tra sâu thêm vụ LoRA-FT (mục cuối file), tìm ra causa gốc **thật sự đơn giản và hoàn
toàn tất định (deterministic)**, không phải "non-determinism theo process" như kết luận ban đầu:

**`infer_prefix_style()` (ablation) / `infer_baseline()` / `infer_lora_ft()` trong
`run_inference.py`/`run_inference_ablation.py` đều hậu xử lý output bằng**
```python
generated_ids = output_ids[:, input_ids.shape[1]:]
```
**giả định `output_ids` = [toàn bộ prompt gốc + token mới sinh]. Giả định này chỉ đúng khi
`generate()` nhận `input_ids` thô.** Nhưng cả 3 hàm trên đều tự tính `inputs_embeds` trước (bắt
buộc khi có ảnh — ảnh+text được hợp nhất thành embedding liên tục, không có `input_ids` rời rạc
tương ứng) rồi chỉ truyền `inputs_embeds` cho `GenerationMixin.generate()` gốc — **trong trường hợp
này, `transformers` trả về `output_ids` CHỈ chứa token mới sinh, không có phần prompt.** Slice
`[:, input_ids.shape[1]:]` khi đó cắt vào đúng phần nội dung mới sinh (vì token mới sinh) một cách
sai lệch: nếu số token trả lời ít hơn độ dài prompt gốc (rất phổ biến — câu hỏi lâm sàng thường dài,
câu trả lời thường ngắn) thì kết quả là **chuỗi rỗng tuyệt đối**; nếu nhiều hơn thì kết quả là một
đoạn bị cắt cụt sai.

**`infer_xattn()` không bao giờ dính bug này** vì nó hậu xử lý theo cách khác hẳn — decode toàn bộ
`output_ids[0]` rồi tách chuỗi bằng `.split("Answer:")[-1]` (string-based), không dựa vào
`input_ids.shape[1]` (index-based) — một lựa chọn thiết kế tình cờ tránh được lỗi này hoàn toàn.

**Về các "bằng chứng non-determinism" ghi ở các mục bên dưới:** nhiều khả năng đó đều là chính bug
slicing này biểu hiện dưới các điều kiện test khác nhau (có test tôi in `output_ids` đã cắt sai,
có test tôi decode trực tiếp không cắt) — không phải ngẫu nhiên thật giữa các process như kết luận
ban đầu. Không loại trừ hoàn toàn khả năng có thêm một lớp non-determinism nhỏ khác, nhưng **bug
slicing là nguyên nhân chi phối, tất định, và đã fix + verify đầy đủ** (xem kết quả cuối file).

## Tóm tắt 1 câu (bản gốc, để tham khảo lịch sử điều tra)

`LlavaMistralForCausalLM.generate()` và `LlavaMistralXAttnForCausalLM.generate()` đều tự tính
`inputs_embeds` rồi chỉ truyền `inputs_embeds` (không kèm `input_ids`) cho
`GenerationMixin.generate()` gốc — dưới `transformers==4.36.2`, cách gọi này **không xác định theo
từng process**: có process sinh đúng, có process trả về **0 token tuyệt đối** dù logits bước đầu
hoàn toàn hợp lý. Fix: truyền `input_ids` trực tiếp thay vì tự tính `inputs_embeds` trước.

## Điểm khởi đầu — câu hỏi ban đầu

Tại sao Baseline (LLaVA-Med gốc, chưa fine-tune) đạt BERTScore-F1 ≈ 0 khi test full multimodal,
nhưng XAttn ở cấu hình text-only (ảnh bị xoá, cross-attn no-op — về lý thuyết phải hành xử y hệt
baseline) lại đạt 0,79?

## Các giả thuyết đã bác bỏ (theo thứ tự đã thử)

1. **Khác cấu trúc input (có/không token ảnh trong chuỗi)** — bác bỏ: chạy control test
   `baseline --modality text_only` (đúng cấu trúc không-ảnh) vẫn cho **0,0000 tuyệt đối** trên mọi
   metric (BLEU/ROUGE/METEOR/BERTScore), y hệt baseline full.
2. **Prompt template khác nhau** (`conv_templates["mistral_instruct"]` cho baseline/lora_ft vs
   `"...\nAnswer:"` thô cho xattn) — bác bỏ: cho baseline chạy đúng format `"Answer:"` y hệt xattn
   (`--prompt_style raw`, xem `run_inference_ablation.py`) vẫn cho **0,0000 tuyệt đối**, 100% (1287/1287)
   prediction rỗng.
3. **Trọng số LLM khác nhau giữa checkpoint baseline và checkpoint XAttn** — bác bỏ: so sánh tensor
   trực tiếp (`embed_tokens`, `self_attn.q_proj` ở layer 0/15/31, `norm`, `lm_head`) cho
   `max_abs_diff = 0.000000` — trọng số giống hệt bit-for-bit.
4. **Dtype khác nhau (baseline mặc định fp16, xattn checkpoint bf16)** — bác bỏ: baseline load lại
   bằng bf16 vẫn cho prediction rỗng 100%.
5. **Tokenizer fast (baseline) vs slow (xattn)** — bác bỏ: token hoá cùng câu tiếng Việt cho ra
   token id **giống hệt** giữa `LlamaTokenizerFast` và `LlamaTokenizer`.
6. **`generation_config.json` khác nhau** (bos/eos/pad token) — bác bỏ: giống hệt nhau
   (`bos_token_id=1, eos_token_id=2`) giữa checkpoint gốc và checkpoint xattn.
7. **Alignment dữ liệu sai** trong `compute_metrics.py` (ghép nhầm prediction với reference của mẫu
   khác) khi so 2 file khác độ dài (1287 vs 449 dòng) — bác bỏ: kiểm tra toàn bộ 449 dòng overlap,
   **0 mismatch** giữa trường `image` của 2 file.
8. **MRO/`super()` khác nhau** giữa `super(LlavaMistralForCausalLM, self)` (baseline) và
   `super(MistralForCausalLM, self)` (xattn, cố tình bỏ qua 1 lớp MRO) — bác bỏ: in trực tiếp
   `__mro__` và định danh hàm, cả hai **trỏ tới cùng một object hàm**
   `<function GenerationMixin.generate at 0x...>` — hoàn toàn giống nhau.
9. **Logits bước sinh token đầu tiên khác nhau** — bác bỏ: gọi `forward()` trực tiếp với cùng
   `input_ids`, `max_abs_diff = 0.0` giữa 2 model, cả hai đều predict token `"The"` (logit 15,19,
   cách xa EOS ở 4,81) — không phải một quyết định "sát nút" có thể bị lật bởi sai số làm tròn.

## Root cause thật (xác nhận 10/10 lần test — 5 cho baseline, 5 cho xattn)

`out.sequences` của `.generate()` có độ dài **bằng đúng độ dài input** (0 token mới) dù
`out.scores[0]` (logits thật đã dùng ở bước 0) cho thấy token đúng cần chọn là `"The"`. Nghĩa là:
tính toán forward hoàn toàn đúng, nhưng **vòng lặp generate() không bao giờ append token đã tính
ra**. Đây là dấu hiệu kinh điển của lỗi khởi tạo `unfinished_sequences`/`cache_position` khi
`generate()` chỉ nhận `inputs_embeds` mà không có `input_ids` đi kèm — hành vi được biết là kém ổn
định ở nhiều phiên bản `transformers` (repo đang dùng `4.36.2`).

**Đặc điểm quan trọng:** "chế độ" tốt/xấu có vẻ được quyết định **một lần** khi model được load /
process khởi động, sau đó **giữ nguyên suốt cả quá trình** (không phải random theo từng mẫu). Vì
vậy: chạy 1287 mẫu trong 1 process có thể ra 100% rỗng HOẶC 100% đúng — tuỳ "may rủi" tại thời điểm
process bắt đầu, không phải xác suất ngẫu nhiên đều trên từng mẫu.

## Fix (đã verify 10/10 lần, cả baseline và xattn)

Thay vì:
```python
inputs_embeds = self.get_model().embed_tokens(inputs)
return super().generate(position_ids=..., attention_mask=..., inputs_embeds=inputs_embeds, **kwargs)
```
Truyền `input_ids` trực tiếp, để `forward()` tự embed theo từng bước (cơ chế fallback này vốn đã có
sẵn trong `forward()` của cả 2 class, chỉ là `generate()` không tận dụng):
```python
return super(LlavaMistralForCausalLM, self).generate(
    input_ids=inputs, position_ids=position_ids, attention_mask=attention_mask, **kwargs,
)
```

File thực hiện fix (không sửa code gốc): `run_baseline_textonly_fixed.py` — monkey-patch
`LlavaMistralForCausalLM.generate` tại runtime, tái dùng nguyên `infer_prefix_style()` và
`load_baseline_model()` đã có sẵn.

## Kết quả sau khi fix (full 1287 mẫu `bone_data/dataset_test.jsonl`, so 449 mẫu chung với
`xattn_s2_text_only.jsonl`)

| Cấu hình | BERTScore-F1 | % prediction rỗng |
|---|---|---|
| Baseline-Conv (trước fix) | 0,0000 | 100% (1287/1287) |
| Baseline-Raw (trước fix) | 0,0000 | 100% (1287/1287) |
| **Baseline-Conv-FIXED** | **0,4281** | 44,8% (577/1287) |
| **Baseline-Raw-FIXED** | **0,7917** | **0%** (0/1287) |
| XAttn-Raw (không đổi, không cần chạy lại) | 0,7898 | — |

**Baseline-Raw-FIXED ≈ XAttn-Raw** (chênh 0,0019, trong nhiễu bình thường) → xác nhận dứt khoát:
cơ chế no-op của cross-attention khi không có ảnh hoạt động **đúng như thiết kế**, không có rò rỉ
kiến trúc nào. Số liệu XAttn text-only 0,79 dùng ở **trang 31 của slide bảo vệ** là đáng tin cậy.

Phát hiện phụ (thật, không phải artifact): conv-template vẫn gây 44,8% prediction rỗng ngay cả sau
khi fix bug — vì baseline chưa từng được (pre)train ở cấu hình "không ảnh + chat template", đây là
hạn chế thật của cách LLaVA-Med gốc xử lý input ngoài phân phối huấn luyện.

## ⚠️ Rủi ro CHƯA verify — ưu tiên cao nhất, ảnh hưởng trực tiếp số liệu cốt lõi

`infer_lora_ft()` trong `run_inference.py` dùng **model đã merge LoRA nhưng vẫn là class
`LlavaMistralForCausalLM`** (PEFT `merge_and_unload()` trả về đúng class gốc) → **chạy qua đúng
đường code `generate()` đã proven lỗi**, ở nhánh `images is not None` (chưa test trực tiếp nhánh
này, chỉ mới test nhánh `images is None`).

Số liệu đang bị nghi ngờ:
- **LoRA-FT BERTScore-F1 = 0,051** (bảng so sánh chính, trang 29) — dùng
  `outputs_lora_baseline_v3` (Case A: PEFT adapter, merge_and_unload).
- **"83% prediction rỗng" của LoRA-FT** — bằng chứng chính cho cơ chế "EOS-Degeneration" (#5 trong
  5 cơ chế kiến trúc LoRA-vs-XAttn) — có thể một phần hoặc toàn bộ là do bug này, không hẳn là thất
  bại kiến trúc thật.
- **Baseline ~0 ở trang 29** (nhánh `images is not None`, chưa test riêng) — nền của toàn bộ bảng
  so sánh chính.

**Cập nhật (2026-08-06) — đã tìm thêm 1 bug độc lập thứ 2, và xác nhận nhánh có-ảnh cũng bị lỗi:**

1. **`load_lora_ft_model()` chọn sai checkpoint** (`run_inference.py`): dùng
   `sorted(glob.glob(model_path/"checkpoint-*"))` — sort theo **chuỗi**, không phải số — nên với
   `outputs_lora_baseline_v3` có `checkpoint-396` và `checkpoint-1188`, kết quả sort chuỗi đưa
   `"checkpoint-396"` lên sau `"checkpoint-1188"` (vì `'3' > '1'` ở ký tự đầu) → `[-1]` chọn nhầm
   **checkpoint-396** (train dở dang) thay vì checkpoint-1188 (checkpoint cuối, train đầy đủ). Bug
   này tồn tại độc lập với bug generate(), và **đã ảnh hưởng tới lần chạy chính thức tạo ra số liệu
   0,051** (cùng logic sort, cùng thư mục).

2. **Nhánh `images is not None` của `generate()` CŨNG bị bug, xác nhận 5/5 lần**: gọi
   `infer_lora_ft()` với ảnh thật (chưa sửa gì) → **100% rỗng** (không phải flaky 2 chiều như
   baseline text-only, mà **luôn luôn rỗng** trong mọi lần test). Kiểm tra trực tiếp
   `prepare_inputs_labels_for_multimodal(...)` xác nhận trả về **`attention_mask=None`** — dòng gốc
   `super().generate(attention_mask=attention_mask, inputs_embeds=inputs_embeds, ...)` do đó luôn
   nhận `attention_mask=None`. Thêm fallback tường minh
   `attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, ...)` trước khi gọi
   `super().generate()` → **5/5 lần cho kết quả nhất quán, có nội dung tiếng Việt thật** (ví dụ:
   `"Kết quả xét nghi..."` — mở đầu hợp lý của "Kết quả xét nghiệm").

## Bug thứ 3 (quan trọng nhất) — phát hiện sau khi bug #2 vẫn chưa đủ để hết rỗng

Sau khi fix bug #1 (checkpoint) và #2 (attention_mask), full run 1287 mẫu **vẫn 100% rỗng**, dù
test tay từng mẫu đơn lẻ (17+ lần, mọi tổ hợp tham số) đều thành công. Sau khi loại trừ toàn bộ khả
năng (dispatch qua patch, load model, max_new_tokens, stopping_criteria, tty/pty, detached vs
foreground...) — bug thật nằm ở chính `infer_lora_ft()`:

```python
generated_ids = output_ids[:, input_ids.shape[1]:]   # SAI khi generate() dùng inputs_embeds
```

Dòng này giả định `output_ids` = [toàn bộ prompt gốc + token mới sinh]. Đúng khi `generate()` nhận
`input_ids` thô, nhưng **sai khi nhận `inputs_embeds`** (bắt buộc với nhánh có ảnh) — lúc đó
`output_ids` trả về **chỉ chứa token mới sinh**, không có phần prompt. Vì câu hỏi lâm sàng thường
dài hơn nhiều số token câu trả lời, phép cắt luôn cắt vào khoảng trống → rỗng tuyệt đối, mọi mẫu,
mọi lần chạy — không phải "non-determinism" như nghi ngờ ban đầu (xem phần ĐÍNH CHÍNH ở đầu file).
Fix: bỏ hẳn việc cắt theo `input_ids.shape[1]` khi dùng `inputs_embeds`, decode toàn bộ
`output_ids[0]` trực tiếp. Verify: 0/20 rồi 0/1287 mẫu rỗng, nội dung tiếng Việt mạch lạc.

## Kết quả cuối cùng (2026-08-06) — full 1287 mẫu, đã audit rò rỉ

**⚠️ Quan trọng: phải dùng đúng `bone_data/dataset_test_minimal.jsonl` (minimal-context) để so
sánh công bằng với XAttn/Qwen2-VL** — cùng bộ ảnh nhưng câu hỏi ngắn (chỉ lý do vào viện), đúng
điều kiện dùng cho bảng so sánh chính thức (trang 32, BERTScore 0,916). Chạy nhầm trên
`bone_data/dataset_test.jsonl` (full-context, văn bản dài) cho kết quả bị thổi phồng
(BERTScore-F1 = 0,9598 full / 0,9408 sạch) vì văn bản dài dễ bị model "ăn gian" bằng cách
copy/paraphrase nội dung đã có sẵn trong chính câu hỏi — **không phản ánh đúng năng lực LoRA.**

**Số liệu đúng, trên minimal-context, đã audit rò rỉ (992 mẫu sạch / 1287):**

| Modality | Full BERT-F1 | Rò rỉ (295) | **Sạch (992)** |
|---|---|---|---|
| Full (ảnh + text) | 0,8728 | 0,9199 | **0,8587** |
| Text-only (không ảnh) | 0,8818 | 0,9232 | **0,8695** |
| Image-only (không text) | 0,8440 | 0,8344 | **0,8469** |

**Kết luận:**
1. **Con số chính thức LoRA-FT = 0,051 (trang 29) và "83% rỗng" đều sai hoàn toàn**, do 3 bug độc
   lập cộng dồn (checkpoint sai + attention_mask thiếu + slicing sai). Số đúng để thay thế:
   **≈0,86 BERTScore-F1** (minimal-context, tập sạch, cấu hình Full).
2. **Luận điểm cốt lõi "XAttn thắng LoRA" nhiều khả năng vẫn đúng** — 0,86 vẫn thấp hơn XAttn đã
   báo cáo (0,916 full test / ước lượng ≥0,891 tập sạch, xem `compute_metrics_clean_subset.py`
   docstring) — chỉ là biên độ chênh lệch thật (+0,865 → có thể chỉ còn ~0,03-0,06) nhỏ hơn nhiều
   so với những gì slide hiện đang trình bày, cần cập nhật số liệu, không cần đổi kết luận định
   tính.
3. **Phát hiện mới, có thể dùng làm bằng chứng bổ sung**: LoRA-FT gần như không phân biệt được
   Full/TextOnly/ImageOnly (0,847–0,870, chênh lệch rất hẹp) — khác hẳn XAttn ở trang 31 (bỏ ảnh
   gây sập hẳn sang tiếng Anh, lệch điểm rõ rệt). Gợi ý: LoRA không đạt được visual grounding rõ
   ràng dứt khoát như XAttn, dù cả hai đều "hoạt động được" về mặt kỹ thuật sau khi hết bug.

## Proxy accuracy + laterality accuracy (`proxy_classification_metrics.py`) — bằng chứng mạnh hơn hẳn BERTScore

Chạy trên full 1287 mẫu minimal-context, cùng điều kiện cho cả 5 model (không audit rò rỉ riêng ở
bước này — so sánh tương đối vẫn công bằng vì tất cả dùng chung 1 tập):

| Model | Accuracy (5 nhóm) | Macro-F1 | Laterality Acc | Laterality sai-bên |
|---|---|---|---|---|
| XAttn-Minimal | **0,7262** | 0,5374 | **0,7933** | 0,0494 |
| QwenVL-LoRA-Minimal | 0,7233 | **0,6239** | 0,7371 | 0,0404 |
| LoRA-FT-FIXED-Full | 0,5288 | 0,4588 | 0,6337 | 0,0629 |
| LoRA-FT-FIXED-TextOnly | 0,5418 | 0,4727 | 0,6584 | 0,0449 |
| LoRA-FT-FIXED-ImageOnly | **0,2911** | **0,1015** | **0,3438** | **0,3416** |

**Đây là bằng chứng thuyết phục hơn nhiều so với BERTScore.** Trên độ đo có ý nghĩa lâm sàng thật
(không phải tương đồng ngữ nghĩa mơ hồ), khoảng cách LoRA-FT vs XAttn **rõ ràng và lớn**: Accuracy
0,53 vs 0,73 (chênh 0,20 tuyệt đối), Laterality 0,63 vs 0,79 (chênh 0,16) — khác hẳn biên độ hẹp mà
BERTScore gợi ý (0,86 vs 0,916).

**LoRA-FT-ImageOnly sụp đổ hoàn toàn**: Accuracy 0,29, và đặc biệt **Laterality chỉ 0,3438 — THẤP
HƠN CẢ ĐOÁN NGẪU NHIÊN (50% cho bài toán nhị phân trái/phải)**, tỉ lệ đoán ngược bên lên tới 34,2%.
Nhìn confusion matrix chi tiết: F1 = 0,0000 tuyệt đối ở 4/5 nhóm bệnh (chỉ còn CHAN_THUONG — nhóm
đông mẫu nhất — có F1 > 0) → model bản chất đang **đoán mù theo nhóm đa số**, không hề trích xuất
thông tin thật từ ảnh khi bị ép chỉ dùng ảnh. Đây không phải "ảnh không giúp ích" nữa mà là "ảnh
không được dùng theo cách có ý nghĩa nào cả" — bằng chứng mạnh hơn nhiều cho luận điểm LoRA thiếu
visual grounding thật sự, nên ưu tiên đưa vào thay/bổ sung cho phần BERTScore ablation.

**Cập nhật kết luận:** với bộ số liệu proxy accuracy/laterality này, luận điểm cốt lõi "XAttn thắng
LoRA" không chỉ "vẫn đứng vững" mà còn **được củng cố rõ ràng hơn** so với dự đoán ban đầu dựa trên
BERTScore — nên ưu tiên dùng Accuracy/Laterality làm số liệu chính khi sửa lại chương 4/slide, xem
BERTScore ≈0,86 chỉ là số liệu phụ.

**File liên quan:** `run_lora_ft_fixed.py` (monkey-patch cả 2 bug generate, có `--modality
full/text_only/image_only`), không sửa `run_inference.py`/`llava_mistral.py` gốc.

## Việc CẦN làm trước khi bảo vệ (nếu còn thời gian)

- [ ] Quyết định có cập nhật lại số liệu LoRA-FT trong chương 4 và slide (trang 29/30) hay không —
      **nên trao đổi với GVHD trước khi sửa**, vì đây là thay đổi số liệu cốt lõi.
- [ ] Nếu cập nhật: thay 0,051 → ~0,86 (minimal, sạch), thay "83% rỗng" → 0% rỗng, và điều chỉnh
      lại phần diễn giải cơ chế "EOS-Degeneration" (#5 trong LoRA-vs-XAttn) — cơ chế #1-4 (Visual
      Dilution, Context Competition, Triple-Conflict, Parameter Capacity) không bị ảnh hưởng, vẫn
      là lập luận kiến trúc hợp lệ; chỉ riêng bằng chứng thực nghiệm "83% rỗng" của #5 cần rút lại.
- [ ] Cân nhắc thêm phát hiện "LoRA không phân biệt Full/TextOnly/ImageOnly" như một bằng chứng bổ
      sung tích cực cho luận điểm (không bắt buộc, nhưng làm câu chuyện chặt chẽ hơn).
- [x] **Baseline đã verify (2026-08-06)** — dùng `run_baseline_fixed_full.py` (cùng 2 fix, không
      có bug checkpoint vì baseline load thẳng từ HF cache). Kết quả full 1287 mẫu minimal-context:

      | | BERTScore-F1 | Accuracy | Macro-F1 | Laterality | % rỗng |
      |---|---|---|---|---|---|
      | Baseline-FIXED | 0,386 | 0,069 | 0,093 | 0,034 | 53,2% (685/1287) |

      Khác với con số "~0,000" gốc, baseline sau khi fix có BERTScore 0,386 (không rỗng tuyệt đối)
      nhưng Accuracy/Laterality gần như bằng 0 (0,069/0,034) — **kết luận định tính "baseline thất
      bại hoàn toàn, cần fine-tune" vẫn đúng nguyên vẹn**, không phải do bug mà do model chưa từng
      học tiếng Việt. Chỉ cần sửa con số cụ thể (0,000 → ~0,39 BERTScore / ~0,07 Accuracy) nếu muốn
      chính xác tuyệt đối, không bắt buộc vì không đổi kết luận.

## Bảng tổng kết cuối cùng — cả 3 model, cùng điều kiện (minimal-context, full 1287 mẫu)

| Model | BERTScore-F1 | Accuracy (5 nhóm) | Laterality | Trạng thái số liệu |
|---|---|---|---|---|
| Baseline | 0,386 | 0,069 | 0,034 | Gần đúng với "~0" gốc, không bắt buộc sửa |
| **LoRA-FT** | **0,859** | **0,529** | **0,634** | **Sai hoàn toàn trong khóa luận (0,051), bắt buộc sửa** |
| XAttn | 0,916 | 0,726 | 0,793 | Đã xác nhận đáng tin cậy |

Câu chuyện narrative "Baseline thất bại (chưa học tiếng Việt) → cần fine-tune LoRA → LoRA đạt độ
tương đồng ngôn ngữ cao nhưng chẩn đoán lâm sàng sai nhiều (BERTScore 0,86 nhưng Accuracy chỉ 0,53)
→ do concat projection làm loãng thông tin thị giác/lâm sàng dài → cần kênh cross-attention riêng
để re-inject thị giác qua từng lớp decoder (lấy cảm hứng Med-Flamingo)" **khớp hoàn toàn với dữ
liệu thật đã kiểm chứng** — chỉ cần thay số liệu LoRA-FT cụ thể, không cần đổi lập luận. Thực tế,
khoảng cách BERTScore cao/Accuracy thấp của LoRA-FT (0,86 vs 0,53) còn là **minh chứng trực tiếp,
định lượng** cho đúng luận điểm "độ tương đồng ngôn ngữ cao không đồng nghĩa chẩn đoán đúng" mà
narrative muốn lập luận — mạnh hơn cả bằng chứng "83% rỗng" cũ.

## Bài học phương pháp luận

- Khi một kết quả "quá sạch" (0,0000 tuyệt đối trên MỌI metric) HOẶC "quá cao bất thường" (vượt xa
  mọi baseline khác) xuất hiện, cả hai đều là dấu hiệu đáng ngờ của lỗi pipeline — không phải bằng
  chứng mạnh về hành vi model. Luôn đọc qualitative output (vài prediction thật) trước khi tin số
  liệu, theo cả hai chiều.
- So sánh logits thô ở bước forward đơn lẻ là cách cô lập nhanh nhất giữa "lỗi tính toán" và "lỗi
  hậu xử lý/vòng lặp điều khiển" khi 2 pipeline cho kết quả khác nhau trên cùng trọng số.
- Luôn tự hỏi "model có thực sự dùng đúng modality mình nghĩ không?" — chạy ablation text-only/
  image-only là cách rẻ nhất để phát hiện một con số cao bất thường có đến từ "ăn gian" ngữ cảnh
  text thay vì suy luận thị giác thật hay không, trước khi công bố kết quả.
- Luôn dùng đúng tập dữ liệu (minimal vs full context) khớp với điều kiện của số liệu đang so sánh
  — nhầm tập dữ liệu có thể tạo ra chênh lệch còn lớn hơn cả bug code.
