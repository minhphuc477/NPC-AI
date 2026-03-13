# OQ4 / RQ4 Completion Playbook

Mục tiêu của playbook này là hoàn tất RQ4 (đánh đổi chất lượng-độ trễ) đồng thời tăng xác suất đạt toàn bộ cổng chất lượng nghiêm ngặt.

## 1) Vấn đề đang chặn pass

Ở run mới nhất `20260312T213231Z`, cổng fail do 2 điều kiện:

1. `controlled_vs_raw:naturalness` âm (`mean_delta=-0.007848`).
2. `human_pref_soft_win` với `baseline_no_context_phi3_latest` dưới ngưỡng (`0.458333 < 0.5`).

Tham chiếu:
- `artifacts/proposal/20260312T213231Z/quality_gate_report_final.json`
- `artifacts/proposal/20260312T213231Z/human_eval_summary.json`

## 2) Những gì đã làm thêm trong repo

Đã cập nhật pipeline chính để hỗ trợ chạy arm đã tune xuyên suốt proposal -> human eval -> quality gate:
- File: `scripts/run_kaggle_full_results.py`
- Tham số mới:
  - `--proposal-control-alt-profile`
  - `--proposal-control-alt-arm-id`
  - `--proposal-control-alt-overrides-file`
  - `--proposal-target-arm`

Điều này cho phép dùng arm đã tune làm target chính thay vì bị khóa cứng vào `proposed_contextual_controlled`.

## 3) Tuning nhanh đã chạy (thực tế)

Đã chạy thử tuning có ràng buộc:
- `scripts/tune_control_architecture.py`
- Output: `artifacts/proposal_control_tuning/auto_tune/20260313T004330Z`
- Kết quả nhanh cho thấy candidate tốt nhất ở vòng nhỏ là trial baseline (`overrides={}`), tức cần chạy tuning sâu hơn để vượt nhiễu run.

## 4) Quy trình chạy để đạt tất cả mục tiêu

### Bước A: Tuning sâu (khuyến nghị)

```powershell
Set-Location "F:\NPC AI"
.\.venv\Scripts\python.exe scripts\tune_control_architecture.py \
  --host "http://127.0.0.1:11434" \
  --candidate-model "elara-npc:latest" \
  --baseline-model "phi3:mini" \
  --scenarios "data/proposal_eval_scenarios_112_diverse.jsonl" \
  --output-root "artifacts/proposal_control_tuning/auto_tune" \
  --seed 37 \
  --train-seeds "19,23" \
  --valid-seeds "29,31" \
  --trials 12 \
  --topk-validate 3 \
  --train-max-scenarios 36 \
  --valid-max-scenarios 36 \
  --max-tokens 72 \
  --temperature 0.15 \
  --timeout-s 120 \
  --min-overall-delta 0.0 \
  --min-context-delta 0.0 \
  --min-persona-delta 0.0 \
  --max-fallback-increase 0.02 \
  --max-retry-increase 0.03 \
  --max-first-pass-drop 0.02
```

Lấy file `recommended_overrides.json` từ run tuning mới nhất.

### Bước B: Chạy full pipeline với arm tuned làm target

```powershell
Set-Location "F:\NPC AI"
.\.venv\Scripts\python.exe scripts\run_kaggle_full_results.py \
  --host "http://127.0.0.1:11434" \
  --candidate-model "elara-npc:latest" \
  --baseline-model "phi3:mini" \
  --baseline-models "phi3:latest" \
  --scenario-file "data/proposal_eval_scenarios_large_v2.jsonl" \
  --proposal-control-alt-profile custom \
  --proposal-control-alt-arm-id proposed_contextual_controlled_tuned \
  --proposal-control-alt-overrides-file "<PATH_TO_RECOMMENDED_OVERRIDES.json>" \
  --proposal-target-arm proposed_contextual_controlled_tuned \
  --proposal-max-tokens 72 \
  --proposal-temperature 0.15 \
  --serving-max-tokens 56 \
  --serving-temperature 0.15 \
  --run-security-benchmark \
  --require-security-benchmark \
  --gate-min-external-significant-wins 10 \
  --gate-min-human-soft-win-rate 0.5 \
  --gate-min-human-kappa 0.2
```

### Bước C: Tiêu chí xác nhận đạt mục tiêu

Mở `quality_gate_report_final.json` của proposal run mới và xác nhận:
- `overall_pass = true`
- `controlled_vs_raw:naturalness` = pass
- `human_pref_soft_win:baseline_no_context_phi3_latest` = pass
- Security checks = pass (nếu bật strict security)

## 5) Lưu ý quan trọng để giảm dao động run

1. Giữ cố định model tags trong toàn bộ chiến dịch benchmark.
2. Giữ thống nhất `temperature`, `max_tokens`, seed, scenario file giữa các run so sánh.
3. Tăng số scenario của human-eval (>= 36 hiện tại là tối thiểu; nên tăng khi cần ổn định soft-win).
4. Nếu pass/fail sát ngưỡng, chạy lặp 2-3 run và chọn run có profile ổn định theo CI thay vì một lần chạy đơn.
