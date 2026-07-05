# Inference-only subset of basicsr (vendored from codeformer-pip).
# Heavy training-time imports (losses, data, train) and their deps (lpips,
# tb-nightly, etc.) are intentionally skipped here. Import arch/util modules
# directly, e.g. `from codeformer.basicsr.archs.codeformer_arch import CodeFormer`.
