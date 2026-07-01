# Native Patch Notes

The native Lumina backend should be preferred for the Euler and Midpoint
scheduler probe, but BSS requires custom schedule injection. The official
repository supports solver selection through `sample.py` and `transport/`,
while the initial public sampler builds its time grid internally.

Use `scripts/patch_lumina_native_custom_grid.py` inside Colab after cloning the
official Lumina repository. The patch is intentionally small and adds a custom
time-grid argument to the ODE path only. DPM is left as audit-only until its
solver coordinate can be safely injected.

