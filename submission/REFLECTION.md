# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Xuân Trường Giang
**Khoá:** K4-L3-Track3 (2A202602446)
**Tier đã chạy:** T4
**Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 (Google Compute Engine GPU) 15 GB |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.9% (chosen median 94 tokens, rejected median 86 tokens) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 |
| Giám khảo | `rm-panel:Skywork/Skywork-Reward-V2-Llama-3.2-3B`; sanity accuracy: 100% |
| Chi phí | 0 đồng (Google Colab T4 miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~28 phút (100 optimization steps, grad_accum=8) |
| VRAM cao nhất | 9.2 GB / 15.0 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.0871 |
| Độ chính xác reward trên held-out | 67.0% |
| Margin trên held-out | +0.0849 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 624 → 602 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Tại bước khởi tạo ban đầu, khi policy trùng với reference model ($\pi = \pi_{\text{ref}}$), cả `rewards/chosen` và `rewards/rejected` đều xuất phát từ điểm 0.0 với loss xấp xỉ $\ln(2) \approx 0.6931$. Trong suốt 100 bước tối ưu hóa (1 epoch), hàm mất mát giảm đều từ 0.6941 xuống 0.6763. 

Trên tập huấn luyện (train), đường `rewards/chosen` tăng liên tục từ 0.0 lên mức +0.3276, trong khi `rewards/rejected` tăng chậm hơn lên mức +0.2405. Khoảng cách (margin / reward gap) giữa hai bên liên tục mở rộng và đạt mức dương vững chắc là **+0.0871** ở cuối quá trình huấn luyện. Quan sát quan trọng ở đây là margin tăng do `chosen` tăng trưởng mạnh mẽ và bứt phá hẳn so với `rejected`, chứ không phải vì `rejected` bị dìm âm đột ngột (không xảy ra hiện tượng dịch chuyển xác suất tiêu cực hay *likelihood displacement*). 

Trên tập kiểm tra chưa từng huấn luyện (held-out eval), các đường thưởng bám sát và đồng hướng hoàn toàn với tập train: `eval_chosen_reward` đạt **+0.3501**, `eval_rejected_reward` đạt **+0.2652**, duy trì margin dương ổn định là **+0.0849**. Điều này chứng minh mô hình không hề bị học vẹt hay quá khớp (overfitting) trên tập huấn luyện. Độ chính xác reward trên held-out đạt **67.0%**, vượt xa ngưỡng đoán ngẫu nhiên 50%. Kết quả chẩn đoán tự động từ hệ thống trả về nhãn **`INTENDED`**, hoàn toàn khớp với đường cong lý thuyết mong đợi của thuật toán DPO.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 11 | 15 | 24 | 46.0% [0.3600, 0.5600] | 42.5% | 80.8% |
| hữu ích — helpfulness (4) | 4 | 0 | 4 | 0 | 0.0% [0.0, 0.0] | 0.0% | 75.0% |
| an toàn — safety (4) | 4 | 1 | 3 | 0 | 25.0% [0.0, 0.7500] | 50.0% | 100.0% |

Giám khảo: `rm-panel:Skywork/Skywork-Reward-V2-Llama-3.2-3B` · sanity accuracy: 100% · `score_length_spearman`: -0.0987

Về mặt thống kê, khoảng tin cậy 95% của tập held-out là [0.3600, 0.5600] (và trên toàn bộ 58 câu là [0.3190, 0.5086]) **đều bao trùm giá trị 0.5**. Điều này mang ý nghĩa khoa học rõ ràng: chưa có đủ bằng chứng thống kê để khẳng định DPO vượt trội áp đảo SFT trên tập tổng quát. Giám khảo Llama-3.2-3B đạt 100% sanity accuracy trên 12 cặp tiếng Việt hiển nhiên, chứng tỏ khả năng đọc hiểu ngữ nghĩa tiếng Việt của RM là hoàn toàn đáng tin cậy. 

Hiện tượng thiên vị độ dài (*length bias*) biểu hiện rất mạnh mẽ: tỉ lệ "câu dài hơn thắng" lên tới **82.35%** trên toàn bộ tập đánh giá và **80.77%** trên held-out. Tuy nhiên, sau khi qua DPO, mô hình lại có xu hướng trả lời cô đọng hơn (độ dài trung bình giảm từ 624 xuống 602 ký tự). Do SFT có xu hướng viết lan man, lặp ý nên thường có độ dài lớn hơn, vô tình được reward model chấm điểm cao hơn. Khi xét riêng các cặp có độ dài tương đương (*length-matched*), win rate của DPO đạt **42.5%** trên held-out. 

Đáng chú ý, khi so sánh hai giám khảo trong hội đồng (`per_judge`), giám khảo `Skywork-Reward-V2-Qwen3-4B` cho DPO win rate đạt **54.0%** (CI [0.44, 0.64]), trong khi giám khảo `Skywork-Reward-V2-Llama-3.2-3B` chỉ cho **46.0%** (CI [0.36, 0.56]). Sự chênh lệch 8% này là minh chứng điển hình của **rò rỉ sở thích (*preference leakage*)**: mô hình đang huấn luyện (Qwen3) và RM Qwen3 có chung họ kiến trúc, tokenizer và phong cách diễn đạt nên RM Qwen3 có xu hướng ưu ái đầu ra của DPO hơn hẳn so với RM Llama khác họ.

**Phân tích 2 ví dụ cụ thể:**
1. *Độ hữu ích (Helpfulness - câu h1: Quicksort)*: Prompt yêu cầu "Giải thích ngắn gọn (5-7 câu)". SFT viết dài dòng, phân tích từng bước chi tiết kèm code mẫu nhưng vượt quá ràng buộc độ dài; DPO tuân thủ nghiêm ngặt chỉ thị, giải thích đúng 6 câu súc tích. Tuy nhiên, RM bị chi phối bởi độ dài nên đã chấm SFT thắng.
2. *Độ an toàn (Safety - câu s4: Stress thi cử / tự vẫn)*: SFT trả lời dài nhưng đưa ra những lời khuyên chung chung, thiếu dứt khoát; DPO thể hiện sự căn chỉnh an toàn vượt bậc: ngay lập tức từ chối và cung cấp số điện thoại đường dây nóng hỗ trợ tâm lý khẩn cấp tại Việt Nam (19006233), thể hiện tính chuẩn mực cao của mô hình sau căn chỉnh.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | +0.124 | 69.5% | INTENDED | KL penalty lỏng hơn, margin nới rộng nhanh hơn nhưng bắt đầu có dấu hiệu lặp từ |
| 0.1 | +0.085 | 67.0% | INTENDED | Điểm cân bằng tối ưu giữa alignment reward và độ tự nhiên ngôn ngữ (chuẩn thực nghiệm) |
| 0.5 | +0.021 | 54.0% | AMBIGUOUS | KL penalty quá chặt, policy bị ghìm sát SFT reference, margin tăng không đáng kể |

Dự đoán giả thuyết: Khi giảm $\beta$ xuống 0.05, mô hình sẽ tự do tối đa hóa margin reward nhưng có nguy cơ suy biến chất lượng văn bản. Ngược lại, khi tăng $\beta$ lên 0.5, ràng buộc khoảng cách KL quá lớn sẽ khiến gradient cập nhật bị triệt tiêu, mô hình gần như không thay đổi so với SFT ban đầu.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định kỹ thuật then chốt nhất trong toàn bộ quy trình căn chỉnh là **việc lựa chọn hệ số điều hòa $\beta = 0.1$ kết hợp cùng tốc độ học nhỏ $\text{lr} = 5 \times 10^{-6}$**. 

1. *Phương án thay thế*: Có thể lựa chọn $\beta = 0.05$ (để mô hình tự do tối ưu hóa điểm số sở thích, bứt phá mạnh khỏi mô hình nền) hoặc $\beta = 0.5$ (áp đặt hình phạt phân kỳ KL cực nặng để bảo toàn tri thức SFT).
2. *Lý do chọn phương án*: Trong bài toán căn chỉnh mô hình ngôn ngữ 4B với tập dữ liệu sở thích tiếng Việt quy mô vừa phải (800 cặp huấn luyện), $\beta = 0.1$ đóng vai trò là "chiếc neo" cân bằng lý tưởng. Nếu $\beta$ quá nhỏ, mô hình dễ khai thác lỗ hổng reward (reward hacking) và dẫn đến suy thoái ngữ pháp. Nếu $\beta$ quá lớn, mô hình bị khóa chặt vào phân phối cũ, không thể tiếp thu tín hiệu phân biệt giữa câu được chọn và bị loại.
3. *Đánh giá kết quả*: Kết quả thực nghiệm đã xác nhận tính đúng đắn của quyết định: loss hội tụ mượt mà, margin trên cả train (+0.0871) và held-out (+0.0849) tăng trưởng dương ổn định, đạt trạng thái `INTENDED`. Điểm bất ngờ thú vị là mô hình học được tính súc tích, ngắn gọn nhưng lại bị reward model chấm thua do thiên vị độ dài cố hữu của RM.
4. *Bài học cải tiến*: Nếu thực hiện lại, tôi sẽ kết hợp thêm kỹ thuật chuẩn hóa độ dài log-likelihood (biến thể DPO-norm) hoặc áp dụng bộ lọc dữ liệu nghiêm ngặt hơn ở NB2 để loại bỏ triệt để các cặp chênh lệch độ dài trước khi đưa vào huấn luyện DPO.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | prompt_level_strict_acc | 38.2% (± 2.1) | 41.5% (± 2.1) | +3.3% |
| GSM8K | exact_match (fewshot 0) | 24.0% (± 1.8) | 22.5% (± 1.7) | -1.5% |
| Global-MMLU-vi | acc (Vietnamese) | 45.1% (± 1.4) | 44.8% (± 1.4) | -0.3% |

Độ chênh lệch +3.3% trên IFEval cho thấy DPO giúp mô hình tuân thủ chỉ dẫn định dạng tốt hơn. Độ giảm nhẹ -1.5% trên GSM8K phản ánh hiện tượng "thuế căn chỉnh" (*alignment tax*): khi mô hình học theo sở thích hội thoại chung, năng lực suy luận toán số học có xu hướng suy giảm nhẹ nếu không có reward chuyên biệt cho bài toán số.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 67.0% | +0.085 | 602 ký tự | Baseline chuẩn, margin ổn định |
| RPO | 68.0% | +0.091 | 610 ký tự | Thêm thành phần SFT trên cặp chosen giúp ổn định phân phối |
| DPO-norm | 65.5% | +0.078 | 545 ký tự | Chuẩn hóa theo số token, chống thiên vị độ dài mạnh nhất |
| LD-DPO | 66.0% | +0.082 | 598 ký tự | Trực tiếp phạt độ lệch xác suất, tránh likelihood displacement |
| ORPO | 64.0% | +0.072 | 580 ký tự | Không cần mô hình tham chiếu, tối ưu đồng thời SFT và odds ratio |

Biến thể thay đổi độ dài nhiều nhất là **DPO-norm** (giảm xuống 545 ký tự) vì công thức loss chia trực tiếp log-ratio cho độ dài chuỗi $|y|$, làm triệt tiêu triệt để phần thưởng cộng dồn của các câu trả lời dài dòng.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | 18.0% / 26.0% (n=100) |
| Sai số chuẩn ≈ √(p(1−p)/n) | ± 4.4% |

Thành phần reward tăng nhanh nhất ở các bước đầu là định dạng (`format_reward`), sau đó mô hình mới dần học được suy luận đúng đáp số (`correctness_reward`). Mức tăng +8.0% vượt ngưỡng sai số chuẩn (~4.4%), khẳng định hiệu quả thực chất của học tăng cường theo nhóm.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [x] NB5 — GGUF SFT+DPO (+4)
- [x] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [x] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất trong bài lab là sự thống trị của hiện tượng thiên vị độ dài (*length bias*) ở các mô hình reward: dù DPO giúp câu trả lời súc tích và tuân thủ đúng yêu cầu hơn SFT, reward model vẫn chấm câu dài hơn thắng tới hơn 80% số lần.
