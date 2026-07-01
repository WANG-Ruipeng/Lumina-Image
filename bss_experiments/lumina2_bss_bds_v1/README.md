# Lumina-Image 2.0 BSS/BDS Scheduler Probe

This folder contains a lightweight scaffold for a Lumina-Image 2.0
scheduler-naturalness experiment.

Primary target:

- Model: `Alpha-VLLM/Lumina-Image-2.0`
- Main comparison: same-NFE uniform sampling versus BSS boundary splitting
- Main solver: native Lumina Euler
- Fallback backend: Diffusers `Lumina2Pipeline` with `FlowMatchEulerDiscreteScheduler`

Rules:

- Do not train or modify model weights.
- Do not store Hugging Face tokens in code.
- Do not commit model weights, generated images, or large metrics artifacts.
- Treat `uniform50` as a reference image, not ground truth.
- Treat this as a quick scheduler probe, not a public T2I benchmark.

Expected Colab weight path:

`/content/drive/MyDrive/ModelWeights/Lumina-Image-2.0/`

Expected Colab run mirror:

`/content/drive/MyDrive/Colab_Projects/Lumina2-BSS-BDS/lumina2_bss_bds_v1/`

Start with:

```bash
python bss_experiments/lumina2_bss_bds_v1/scripts/audit_lumina2.py \
  --lumina_root /content/Lumina-Image-2.0 \
  --weights_root /content/drive/MyDrive/ModelWeights/Lumina-Image-2.0 \
  --experiment_root /content/Lumina2-BSS-Runs/lumina2_bss_bds_v1

python bss_experiments/lumina2_bss_bds_v1/scripts/make_manifest_lumina2_bds.py \
  --experiment_root /content/Lumina2-BSS-Runs/lumina2_bss_bds_v1
```

Then validate schedules before running model inference:

```bash
python bss_experiments/lumina2_bss_bds_v1/scripts/validate_schedule.py \
  --manifest /content/Lumina2-BSS-Runs/lumina2_bss_bds_v1/manifests/lumina2_smoke_manifest.csv
```

