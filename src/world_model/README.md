# world_model

NVIDIA Cosmos-Predict2.5-2B LoRA 파인튜닝과 추론

- 학습 데이터 전처리 : 16fps · 528×960 · 93프레임 클립, `videos/*.mp4` + `metas/*.txt`
- LoRA 학습 : rank/alpha 8/8, lr 1e-4, BF16
- Image2World 추론
