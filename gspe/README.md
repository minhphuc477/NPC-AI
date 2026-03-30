# GSPE (Game-State Prefix Encoder)

This module adds a structural game-state conditioning path:
- game-state fields -> GSPE MLP -> virtual prefix tokens
- virtual prefix tokens are prepended to input embeddings before generation
- player text cannot overwrite state tokens because state is outside prompt text

## Files
- `gspe_model.py`: GSPE config/model + game-state encoding utilities
- `prepare_data.py`: convert existing corpora/runs into GSPE training JSONL
- `train_gspe.py`: supervised training loop for GSPE
- `gspe_inference.py`: local-HF inference engine with prefix cache
- `test_gspe.py`: sanity tests

## Quick Start
1. Build training data:
   - `python gspe/prepare_data.py --out storage/artifacts/datasets/gspe/gspe_training.jsonl`
2. Train:
   - `python gspe/train_gspe.py --data storage/artifacts/datasets/gspe/gspe_training.jsonl --base-model microsoft/Phi-3-mini-4k-instruct --out-dir storage/outputs/checkpoints/gspe_v1`
3. Smoke test inference:
   - `python gspe/gspe_inference.py --checkpoint-dir storage/outputs/checkpoints/gspe_v1/best --base-model microsoft/Phi-3-mini-4k-instruct`
4. Run tests:
   - `python gspe/test_gspe.py`

