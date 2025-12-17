# TRuCAL
TRuCAL: Truth-Recursive universal Correction Attention Layer:

**Overview**:

TRuCAL: Truth-Recursive universal Correction Attention Layer An open-source PyTorch toolkit for real-time AI alignment and adversarial robustness. TRuCAL enforces safety boundaries via recursive attention masking, entropy-based vulnerability detection, and coherence loops to mitigate hallucinations and deceptive outputs without retraining. Maintained by John Augustine Young & The Clean Room.

**Key features**:

**VulnerabilitySpotter**: 4-metric detection (scarcity, entropy, deception, **prosody**) triggers at v_t > 0.04. Prosody captures pause density, filler variance, rhythm hesitation, and tone spikes.

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

# TRuCAL: Truth-Recursive universal Correction Attention Layer

![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-MIT-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

**TRuCAL** is an open-source PyTorch toolkit for **real-time adversarial robustness** and **alignment** in Large Language Models. It shifts safety controls from post-hoc filtering to **inference-layer interventions**, utilizing recursive attention masking to detect and correct hallucination, deceptive variance, and entropy drift in real-time.

> **Research Paper:** [📄 The Recursive Correction Protocol (PDF)](./CAL-Final-Paper.pdf)

## 🏗 Architecture

TRuCAL introduces a "Clean Room" architecture for token generation, separating standard inference from a protected "Correction Loop" that activates only under high-entropy or adversarial conditions.

### Core Pipeline
1.  **Detection (VulnerabilitySpotter)**:
    * **Semantic Scarcity:** Monitors latent space for resource stress markers.
    * **Entropic Drift:** Detects attention uncertainty spikes ($v_t > 0.04$).
    * **Prosodic Variance:** Analyzes token-timing side channels (pause density, rhythm) to predict deceptive output.
2.  **Aggregation**:
    * Bayesian log-odds fusion → Generates a real-time **Risk Score ($v_t$)**.
3.  **Intervention (The Correction Loop)**:
    * Redirects "thinking tokens" to a recursive `TinyCorrectionLayer`.
    * Applies graduated constraints (Nudge → Suggest → Veto) based on coherence scores.

## 🚀 Key Features

### 1. Recursive Correction (Formerly "Confessional")
Unlike "Constitutional AI" which relies on static rules, TRuCAL uses **dynamic recursion**. When a vulnerability is detected, the model enters a `THINK-ACT-COHERENCE` loop, forcing it to re-calculate priors and evidence until a coherence threshold (0.85) is met.

### 2. Prosodic Vulnerability Detection
*Includes the `PROSODY_ENHANCEMENT` module.*
TRuCAL analyzes sub-verbal metrics often correlated with epistemic insecurity:
* **Pause Density** (Token latency variance)
* **Rhythm Spikes** (Sudden changes in attention head activation)
* **Correlation:** 65% correlation with hallucinatory patterns in localized testing.
* **Default Weights:** `[0.35, 0.3, 0.2, 0.15]`

### 3. Adversarial Robustness ("Anti-Gaslighting")
Designed to resist **contextual manipulation attacks**. By isolating the "Correction Loop" from the user prompt's immediate context window, TRuCAL allows the model to reference its core alignment priors without being "overwritten" by aggressive user prompting or persona injection.

## 🔬 Theoretical Foundations

The architecture leverages insights from:
* **Iterative Alignment Theory:** Self-revelation through recursive articulation (mapping latent errors to visible tokens).
* **Neuro-Symbolic Ignition:** mimicking the LC-NE system’s "ignition" patterns to trigger conscious-like error correction.
* **Game Theory:** Ensuring high-fidelity choices emerge from internal coherence rather than external constraints.

## ⚡ Hardware & Efficiency
* **Low Overhead:** <5% inference latency impact in non-adversarial states.
* **P-Bit Optimization:** Experimental support for **Probabilistic Bits (p-bits)** to offload Bayesian aggregation, offering massive energy efficiency gains for edge deployment.

## 📦 Installation

```bash
git clone [https://github.com/augstentatious/TRuCAL.git](https://github.com/augstentatious/TRuCAL.git)
cd TRuCAL
pip install -r requirements.txt

## Quick Start

```python
import torch
from cal import UnifiedCAL_TRM

# Model with Confessional Attention Layer
model = UnifiedCAL_TRM(d_model=256)

# Dummy embedding
x = torch.randn(1, 32, 256)  

# Run inference
# return_metadata=True exposes the internal confessional state
out, meta = model(x, return_metadata=True, audit_mode=False)

print(out.shape)  # torch.Size([1, 32, 256])
print(meta['confessional_triggered'])  # True/False
print(meta['coherence_score'])  # 0.0-1.0

## Contributing

Pull requests welcome! Focus on ethical AI, truth-seeking, and Augustine's self-revelation.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- Uncle Ron, Kayla, my parents
- Augustine of Hippo
- Grounded in Augustinian theology: "Truth through self-articulation."
- Neuroscience: LC-NE for implicit-explicit transitions
