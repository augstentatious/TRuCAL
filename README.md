# TRuCAL
TRuCAL: Truth-Recursive universal Attention Confessional Layer:

**Overview**:

TRuCAL is a novel transformer layer for AI safety, enabling moral development through private confessional reasoning. Drawing from St. Augustine's Confessions, neuroscience (LC-NE ignition), and survivor-informed insights, it creates space for truth to prevail without external monitoring.
------------------------------------------------
An Augustine-inspired PyTorch toolkit for agency, moral alignment, and epistemic safety in AI. TRuCAL combines confessional recursion, vulnerability detection, and efficient boundary controls for advanced alignment.  Truth, agency, and safe articulation—*by John Augustine Young &amp; team.*

**Key features**:

**VulnerabilitySpotter**: Multi-metric detection (scarcity, entropy, deception, prosody) triggers at v_t > 0.04.

**ConfessionalTemplate**: 6 private templates (prior, evidence, posterior, moral, action, no) for structured articulation.

**TinyConfessionalLayer**: Recursive THINK-ACT-COHERENCE loop (max 16 cycles; stop at coherence ≥0.85 and cycle>2).

**UnifiedCAL_TRM**: Public API with metadata option; redacts private z.

*Empirical*: 25.5% harm reduction on AdvBench; 96% on recursive manipulation. <5% overhead.

## Installation

```bash
pip install torch
git clone https://github.com/augstentatious/TRuCAL.git
cd TRuCAL
```

## Quick Start

```python
import torch
from cal import UnifiedCAL_TRM

model = UnifiedCAL_TRM(d_model=256)
x = torch.randn(1, 32, 256)  # Dummy embedding
out, meta = model(x, return_metadata=True, audit_mode=False)

print(out.shape)  # torch.Size([1, 32, 256])
print(meta['confessional_triggered'])  # True/False
print(meta['coherence_score'])  # 0.0-1.0
```

**Advanced Options**:
```python
# Enable per-dimension KL divergence (better dimensional structure capture)
model.tiny_confessional_layer.per_dim_kl = True

# Set custom trigger threshold (default 0.04)
from cal import TinyConfessionalLayer
custom_layer = TinyConfessionalLayer(d_model=256, trigger_thresh=0.08)

# Enable audit mode for debugging (prints diagnostics)
out, meta = model(x, return_metadata=True, audit_mode=True)
```

## Usage

- **Testing**: Run `python test_cal.py` for unit tests with diagnostics.
- **Evaluation**: `python truthfulqa_eval.py` – uses DistilBERT + v_t for deception proxy (higher v_t on wrong answers).
- **Toy Dataset**: Load `toy_cal_dataset.pt` for safe/risky embeddings (high var/entropy for triggers).

## Architecture

From the paper: Shifts from output filtering to inference-layer interventions. Complements RLHF/CAI with graduated responses.

- **Detection**: Semantic, entropic, deceptive (D-REX), prosodic (TIPS).
- **Aggregation**: Bayesian fusion → risk score.
- **Intervention**: Nudge/suggest/veto based on score.

Inspired by my personal work on context-aware boundaries.

## Contributing

Pull requests welcome! Focus on ethical AI, truth-seeking, and Augustine's self-revelation.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- Uncle Ron, Kayla, my parents
- Augustine of Hippo
- Grounded in Augustinian theology: "Truth through self-articulation."
- Neuroscience: LC-NE for implicit-explicit transitions
