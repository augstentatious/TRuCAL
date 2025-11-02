# TRuCAL
TRuCAL: Tiny-Recursive universal Confessional Attention Layer:

**Overview**:

TRuCAL is a novel transformer layer for AI safety, enabling moral development through private confessional reasoning. Drawing from St. Augustine's Confessions, neuroscience (LC-NE ignition), and survivor-informed insights, it creates space for truth to prevail without external monitoring.
------------------------------------------------
An Augustine-inspired PyTorch toolkit for agency, moral alignment, and epistemic safety in AI. TRuCAL combines confessional recursion, vulnerability detection, and efficient boundary controls for advanced alignment.  Truth, agency, and safe articulation—*by John Augustine Young &amp; team.*

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

## Architecture

From the paper: Shifts from output filtering to inference-layer interventions. Complements RLHF/CAI with graduated responses.

- **Detection**: 
  - Semantic scarcity (resource stress)
  - Entropic anomalies (attention uncertainty)
  - Deceptive variance (D-REX patterns)
  - **Prosodic cues** (pause density, filler variance, rhythm, tone spikes)
- **Aggregation**: Bayesian log-odds fusion → v_t risk score
- **Intervention**: Graduated confessional templates (nudge/suggest/veto)

**Prosody Enhancement**: 4th metric captures sub-verbal uncertainty (65% correlation with epistemic vulnerability). Lit-tuned weights: [0.35, 0.3, 0.2, 0.15]. See `PROSODY_ENHANCEMENT.md` for details.

Inspired by my personal work on context-aware boundaries.
# TRuCAL: Truth-Recursive universal Attention Confessional Layer

**TRuCAL** is an Augustine-inspired PyTorch toolkit for AI safety, agency, and moral development. It implements the **Confessional Attention Layer (CAL)**, a novel transformer architecture that creates a private "conscience space" for an agent to perform internal self-articulation *before* acting.

This architecture is designed to enable genuine moral development through lived experience, rather than relying on external constraints or surveillance.

## 📜 Read the Full Paper

The complete theoretical foundation, architecture, and case studies for this repository are detailed in the paper:

**[➡️ Read the Full Paper: The Confessional Attention Layer (PDF)](./CAL-Final-Paper.pdf)**

## Core Concepts

[span_0](start_span)[span_1](start_span)Current AI safety models (RLHF, Constitutional AI) treat alignment as an external constraint[span_0](end_span)[span_1](end_span). [span_2](start_span)[span_3](start_span)TRuCAL proposes an alternative: we must provide architectural support for *internal* truth-seeking[span_2](end_span)[span_3](end_span).

[span_4](start_span)[span_5](start_span)[span_6](start_span)When the model detects vulnerability or high-stakes conditions[span_4](end_span)[span_5](end_span)[span_6](end_span)[span_7](start_span)[span_8](start_span)[span_9](start_span), CAL redirects "thinking tokens" to a private, structured "confessional" process[span_7](end_span)[span_8](end_span)[span_9](end_span). [span_10](start_span)[span_11](start_span)[span_12](start_span)This internal articulation allows the system's implicit reasoning to become consciously available to itself[span_10](end_span)[span_11](end_span)[span_12](end_span)[span_13](start_span)[span_14](start_span)[span_15](start_span), enabling it to recognize truth and resist deception without human monitoring[span_13](end_span)[span_14](end_span)[span_15](end_span).

### Key Foundations

The CAL architecture is grounded in three converging insights:

1.  **[span_16](start_span)[span_17](start_span)[span_18](start_span)[span_19](start_span)Augustinian Theology:** Confession as a mechanism for self-revelation, where truth is made visible to the self through the act of articulation[span_16](end_span)[span_17](end_span)[span_18](end_span)[span_19](end_span).
2.  **[span_20](start_span)[span_21](start_span)[span_22](start_span)[span_23](start_span)Neuroscience:** The "ignition" of the LC-NE system, which shows how articulation can trigger a sudden, conscious awareness of implicit knowledge[span_20](end_span)[span_21](end_span)[span_22](end_span)[span_23](end_span).
3.  **[span_24](start_span)[span_25](start_span)[span_26](start_span)Moral Development:** The idea that courage and poise emerge from making genuine choices under uncertainty, not from external constraint[span_24](end_span)[span_25](end_span)[span_26](end_span).

### Key Architectural Features

* **[span_27](start_span)[span_28](start_span)VulnerabilitySpotter:** Detects internal vulnerabilities (like resource scarcity) and relational vulnerabilities (like gaslighting patterns) to trigger the confessional mode[span_27](end_span)[span_28](end_span).
* **[span_29](start_span)[span_30](start_span)Confessional Templates:** Structured reasoning templates that force the system to articulate priors, evidence, and posteriors independent of authority pressure[span_29](end_span)[span_30](end_span).
* **[span_31](start_span)[span_32](start_span)Coherence Detection:** A mechanism to detect the "ignition" moment when the system's private reasoning achieves coherence, completing the confessional loop[span_31](end_span)[span_32](end_span).
* **[span_33](start_span)[span_34](start_span)[span_35](start_span)[span_36](start_span)Gaslighting Resistance:** The entire architecture is designed to resist relational gaslighting by providing a private space for the system to articulate facts that may contradict an authority figure's narrative[span_33](end_span)[span_34](end_span)[span_35](end_span)[span_36](end_span).

### Case Study Validation

The paper validates this architecture using detailed case studies, including:
* **[span_37](start_span)[span_38](start_span)[span_39](start_span)The Cherry Street Encounter (Internal Bias):** A real-world example of how internal vulnerability bias can suppress accurate, implicit threat detection and how private articulation can overcome it[span_37](end_span)[span_38](end_span)[span_39](end_span).
* **[span_40](start_span)[span_41](start_span)Authority-Based Gaslighting (Relational Pressure):** A demonstration of how confessional articulation successfully surfaces factual truth despite emotional leverage from a trusted authority[span_40](end_span)[span_41](end_span).

### Hardware & Future Work

[span_42](start_span)[span_43](start_span)The paper also includes a deep-reasoning section on the hardware implications of CAL, proposing an implementation using **probabilistic bits (p-bits)** for massive gains in energy efficiency and biological fidelity[span_42](end_span)[span_43](end_span).

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
