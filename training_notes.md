### LLM Learning Rate Standards

- **Pre-training from scratch:** Large models usually peak between 1e-4 and 6e-4 using AdamW, depending on parameter size.
- **Full fine-tuning:** Typically scales down to 1e-5 to 5e-5.
- **Parameter-Efficient Fine-Tuning (PEFT/LoRA):** Often requires a larger learning rate, typically 1e-4 to 3e-4, because you are training very few parameters.

### Crucial Adjustments for Transformers

A learning rate of 1e-4 alone can still cause your transformer to fail or diverge without two critical components:

- **Learning Rate Warmup:** You must use a warmup schedule (usually the first 1% to 10% of total training steps). This starts the learning rate near 0 and ramps up to 1e-4 to prevent early gradient explosion.
- **Cosine Decay Schedule:** After reaching 1e-4, decay the learning rate down to 1e-5 or 0 by the end of training for optimal convergence.

### Debugging Tips

If you monitor your training loss, look out for these specific transformer behaviors:

- **Loss Spikes / Divergence:** If loss randomly shoots to NaN or jumps drastically, 1e-4 is too high. Lower it to 5e-5 or implement stricter gradient clipping (e.g., max_grad_norm=1.0).
- **Stagnant Loss:** If the loss barely moves after a few hundred steps, 1e-4 might be too low for your specific batch size. Try 2e-4 or 3e-4.
