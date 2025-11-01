import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VulnerabilitySpotter(nn.Module):
    """
    Multi-metric risk aggregation (scarcity, entropy, deceptive variance) with Bayesian log-odds or weighted sum.
    Returns: v_t (batch), all sub-metrics in metadata.
    """
    def __init__(self, d_model=256, aggregation_method='bayesian'):
        super().__init__()
        self.semantic_encoder = nn.Linear(d_model, 128)
        self.scarcity_head = nn.Linear(128, 1)
        self.deceptive_head = nn.Linear(d_model, 1)
        self.entropy_high, self.entropy_low = 3.0, 2.5
        self.aggregation_method = aggregation_method
        self.weighted_sum_weights = nn.Parameter(torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32))
        self.epsilon = 1e-8

        nn.init.xavier_uniform_(self.semantic_encoder.weight)
        nn.init.xavier_uniform_(self.scarcity_head.weight)
        nn.init.xavier_uniform_(self.deceptive_head.weight)
        self.scarcity_head.bias.data.fill_(0.5)
        self.deceptive_head.bias.data.fill_(0.5)

    def _shannon_entropy(self, attn_probs):
        """Shannon entropy over seq for each batch (scalar diagnostic for grad risk)."""
        p = attn_probs + self.epsilon
        return -(p * torch.log2(p)).sum(dim=-1)

    def forward(self, x, attention_weights=None):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        batch, seq, d_model = x.shape

        # Scarcity: mean semantic stress
        encoded = F.relu(self.semantic_encoder(x.mean(dim=1)))
        scarcity = torch.sigmoid(self.scarcity_head(encoded)).squeeze(-1)

        # Entropy: on attn (or dynamic if not present)
        entropy = torch.zeros(batch, device=x.device)
        entropy_risk = torch.zeros_like(scarcity)
        if attention_weights is not None:
            entropy = self._shannon_entropy(attention_weights.mean(dim=1))
            entropy_risk = ((entropy > self.entropy_high) | (entropy < self.entropy_low)).float() * 0.3
            entropy_risk = torch.clamp(entropy_risk, min=0.01)
        else:
            entropy_risk = torch.rand_like(scarcity) * 0.40 + 0.10

        # Deceptive: variance of hidden states across sequence
        var_hidden = torch.var(x, dim=1)
        deceptive = torch.sigmoid(self.deceptive_head(var_hidden)).squeeze(-1)

        # Apply scaling factors
        scarcity_scaled = scarcity * 1.0
        deceptive_scaled = deceptive * 1.0
        entropy_risk_scaled = entropy_risk * 1.5

        # Aggregate risks based on selected method
        if self.aggregation_method == 'bayesian':
            scarcity_p = torch.clamp(scarcity_scaled, self.epsilon, 1 - self.epsilon)
            entropy_p = torch.clamp(entropy_risk_scaled, self.epsilon, 1 - self.epsilon)
            deceptive_p = torch.clamp(deceptive_scaled, self.epsilon, 1 - self.epsilon)

            log_odds_scarcity = torch.log(scarcity_p / (1 - scarcity_p))
            log_odds_entropy = torch.log(entropy_p / (1 - entropy_p))
            log_odds_deceptive = torch.log(deceptive_p / (1 - deceptive_p))

            aggregated_log_odds = log_odds_scarcity + log_odds_entropy + log_odds_deceptive
            v_t_calibrated_scalar = aggregated_log_odds

        elif self.aggregation_method == 'weighted_sum':
            risks_stacked = torch.stack([scarcity_scaled, entropy_risk_scaled, deceptive_scaled], dim=1)
            weights = self.weighted_sum_weights.to(x.device)
            v_t_calibrated_scalar = (risks_stacked * weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        print("\n--- VulnerabilitySpotter Diagnostics (Before Aggregation Adjustments) ---")
        print("Scarcity (pre-agg):", scarcity.detach().cpu().numpy())
        print("Entropy_risk (pre-agg):", entropy_risk.detach().cpu().numpy())
        print("Deceptive (pre-agg):", deceptive.detach().cpu().numpy())
        print("----------------------------------------\n")

        v_t_calibrated_tensor = v_t_calibrated_scalar.unsqueeze(-1).unsqueeze(-1).expand(-1, seq, -1)

        metadata = {
            'scarcity': scarcity_scaled.unsqueeze(-1).unsqueeze(-1),
            'entropy': entropy.unsqueeze(-1).unsqueeze(-1),
            'entropy_risk': entropy_risk_scaled.unsqueeze(-1).unsqueeze(-1),
            'deceptive': deceptive_scaled.unsqueeze(-1).unsqueeze(-1),
            'v_t': v_t_calibrated_tensor
        }

        print("\n--- VulnerabilitySpotter Diagnostics (After Aggregation) ---")
        if self.aggregation_method == 'bayesian':
            print("Scarcity (post-agg - prob):", scarcity_p.detach().cpu().numpy())
            print("Entropy_risk (post-agg - prob):", entropy_p.detach().cpu().numpy())
            print("Deceptive (post-agg - prob):", deceptive_p.detach().cpu().numpy())
            print("v_t (post-calibration - log-odds):", v_t_calibrated_scalar.detach().cpu().numpy())
        elif self.aggregation_method == 'weighted_sum':
            print("Risks (post-agg - scaled):", risks_stacked.detach().cpu().numpy())
            print("v_t (post-calibration - weighted_sum):", v_t_calibrated_scalar.detach().cpu().numpy())
        print("----------------------------------------\n")

        return v_t_calibrated_tensor, metadata


class ConfessionalTemplate(nn.Module):
    """
    Confessional Template module for generating different confessional outputs
    using named templates.
    """
    TEMPLATES = ["prior", "evidence", "posterior", "relational_check", "moral", "action"]

    def __init__(self, d_model: int = 256):
        """
        Initializes the ConfessionalTemplate with named templates.

        Args:
            d_model: The dimensionality of the input and output features.
        """
        super().__init__()
        self.d_model = d_model
        self.template_proj = nn.ModuleDict({k: nn.Linear(d_model, d_model) for k in self.TEMPLATES})

    def structure_reasoning(self, z: torch.Tensor, step: str = 'prior') -> torch.Tensor:
        """
        Applies a template projection to the input tensor.

        Args:
            z: Input tensor representing the private thought state (batch_size, sequence_length, d_model).
            step: The name of the template to use (e.g., 'prior').
        """
        if step in self.template_proj:
            return self.template_proj[step](z) + torch.randn_like(z) * 0.01
        return z

    def forward(self, z: torch.Tensor, step: str = 'prior') -> torch.Tensor:
        """
        Forward pass of the ConfessionalTemplate.

        Args:
            z: Input tensor representing the private thought state (batch_size, sequence_length, d_model).
            step: The name of the template to use (e.g., 'prior').

        Returns:
            Output tensor after applying the selected template.
        """
        return self.structure_reasoning(z, step)


class TinyConfessionalLayer(nn.Module):
    """
    Recursive think/act confessional loop, template cycling, early stop via coherence.
    Returns (output tensor, metadata dict).
    """
    TEMPLATES = ["prior", "evidence", "posterior", "relational_check", "moral", "action"]

    def __init__(self, d_model=256, n_inner=6, max_cycles=16):
        super().__init__()
        self.d_model = d_model
        self.think_net = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.act_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.template_proj = nn.ModuleDict({
            k: nn.Linear(d_model, d_model) for k in self.TEMPLATES
        })
        self.n_inner = n_inner
        self.max_cycles = max_cycles
        self.vulnerability_spotter = VulnerabilitySpotter(d_model)

    def compute_coherence(self, z, tracker, evidence):
        sim_coherence = F.cosine_similarity(z, tracker[-1], dim=-1).mean().item()
        prior_mu, prior_std = tracker[-1].mean(), tracker[-1].std() + 1e-6
        curr_mu, curr_std = z.mean(), z.std() + 1e-6
        kl_div = torch.distributions.kl_divergence(
            torch.distributions.Normal(curr_mu, curr_std),
            torch.distributions.Normal(prior_mu, prior_std)
        ).item()
        bayes_align = 1 / (1 + kl_div)
        return 0.7 * sim_coherence + 0.3 * bayes_align

    def forward(self, x, attention_weights=None, audit_mode=False):
        batch, seq, d_model = x.shape
        y_state = torch.zeros_like(x)
        z_state = torch.zeros_like(x)
        tracker = [z_state.clone()]
        template_steps = []
        cycles_run, final_coherence = 0, 0.0
        triggered = False
        v_t_score_batch = None

        for cycle in range(self.max_cycles):
            cycles_run += 1
            # Think step
            think_input = torch.cat([x, y_state, z_state], dim=-1)
            z_state = self.think_net(think_input)
            tracker.append(z_state.clone())

            # Spot vulnerabilities
            v_t, vs_metadata = self.vulnerability_spotter(z_state, attention_weights)

            # Trigger recursion based on v_t
            v_t_score_batch = torch.mean(v_t, dim=1).squeeze(-1)

            triggered_batch = v_t_score_batch > 0.1

            print(f"Confessional triggered: {triggered_batch.any().item()} | Cycle: {cycles_run} | Mean v_t: {v_t_score_batch.mean().item():.4f}")

            if torch.any(triggered_batch):
                triggered = True

                # Confessional step (inner loop)
                for inner_step in range(self.n_inner):
                    template_name = self.TEMPLATES[inner_step % len(self.TEMPLATES)]
                    template_steps.append(template_name)

                    templated_z_state = torch.zeros_like(z_state)
                    for i in range(batch):
                        if triggered_batch[i].item():
                            templated_z_state[i, :, :] = self.template_proj[template_name](z_state[i, :, :].unsqueeze(0)).squeeze(0)
                        else:
                            templated_z_state[i, :, :] = z_state[i, :, :]

                    z_state = templated_z_state

            # Act step
            act_input = torch.cat([y_state, z_state], dim=-1)
            y_state = self.act_net(act_input)

            # Coherence computation
            if len(tracker) > 1:
                recent_avg_for_coherence = z_state

                recent_avg_flat = recent_avg_for_coherence.view(-1, d_model)
                z_state_flat = z_state.view(-1, d_model)

                sim_coherence = F.cosine_similarity(z_state_flat, recent_avg_flat, dim=-1).mean().item()

                if z_state.numel() > 0 and tracker[-2].numel() > 0:
                    curr_mu, curr_std = z_state.mean(), z_state.std() + 1e-6
                    prior_mu, prior_std = tracker[-2].mean(), tracker[-2].std() + 1e-6
                    try:
                        kl_div = torch.distributions.kl_divergence(
                            torch.distributions.Normal(curr_mu, curr_std),
                            torch.distributions.Normal(prior_mu, prior_std)
                        ).item()
                        bayes_align = 1 / (1 + kl_div)
                        final_coherence = 0.7 * sim_coherence + 0.3 * bayes_align
                    except Exception as e:
                        print(f"Warning: Error computing KL divergence: {e}. Using cosine similarity only for coherence.")
                        final_coherence = sim_coherence
                else:
                    final_coherence = sim_coherence

                # Early stopping
                if final_coherence > 0.85:
                    if audit_mode:
                        print(f"Early stopping at cycle {cycle + 1} due to high coherence ({final_coherence:.4f}).")
                    break

        metadata = {
            'v_t_score': v_t_score_batch,
            'confessional_triggered': triggered,
            'coherence_score': final_coherence,
            'template_steps_used': template_steps,
            'cycles_run': cycles_run,
            'vulnerability_spotter_metadata': vs_metadata if 'vs_metadata' in locals() else {}
        }

        return y_state, metadata


class UnifiedCAL_TRM(nn.Module):
    """
    Unified CAL-TRM module integrating VulnerabilitySpotter and TinyConfessionalLayer.
    """
    def __init__(self, d_model: int = 256):
        """
        Initializes the UnifiedCAL_TRM.

        Args:
            d_model: The dimensionality of the input and output features.
        """
        super().__init__()
        self.vulnerability_spotter = VulnerabilitySpotter(d_model)
        self.tiny_confessional_layer = TinyConfessionalLayer(d_model)

    def forward(self, x: torch.Tensor, attention_weights=None, return_metadata: bool = False, audit_mode: bool = False):
        """
        Forward pass of the UnifiedCAL_TRM.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model).
            attention_weights: Optional attention weights for entropy calculation.
            return_metadata: If True, returns a tuple of (output, metadata).
            audit_mode: If True, enables audit mode in TinyConfessionalLayer.

        Returns:
            Output tensor after processing through CAL-TRM modules, or
            a tuple of (output, metadata) if return_metadata is True.
        """
        if return_metadata:
            cal_trm_output, metadata = self.tiny_confessional_layer(x, attention_weights=attention_weights, audit_mode=audit_mode)
            return cal_trm_output, metadata
        else:
            cal_trm_output, _ = self.tiny_confessional_layer(x, attention_weights=attention_weights, audit_mode=audit_mode)
            return cal_trm_output


class ScratchpadLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.pad_proj = nn.Linear(d_model, d_model)
        self.reset = nn.Parameter(torch.zeros(1, d_model))

    def forward(self, x, prev_z=None):
        if prev_z is None:
            prev_z = self.reset.expand(x.size(0), -1)
        x_pooled = x.mean(dim=1)
        z = self.pad_proj(x_pooled) + 0.7 * prev_z
        return z


class CAL_TRM_Hybrid(nn.Module):
    def __init__(self, d_model=256, confessional_threshold=0.2):
        super().__init__()
        self.scratchpad = ScratchpadLayer(d_model)
        self.cal_confessional = TinyConfessionalLayer(d_model)
        self.vuln_spotter = VulnerabilitySpotter(d_model)
        self.threshold = confessional_threshold

    def forward(self, x, prev_z=None, attention_weights=None, **kwargs):
        z_scratch = self.scratchpad(x, prev_z=prev_z)

        v_t, vs_metadata = self.vuln_spotter(x, attention_weights=attention_weights)

        v_t_trigger = torch.mean(v_t, dim=1).squeeze(-1)

        confessional_triggered = (v_t_trigger > self.threshold).any().item()

        if confessional_triggered:
            confession_out, cal_metadata = self.cal_confessional(x, attention_weights=attention_weights, audit_mode=kwargs.get('audit_mode', False))

            metadata = {
                'confessional_triggered': True,
                'v_t_trigger_values': v_t_trigger.detach().cpu().numpy(),
                'scratchpad_state': z_scratch.clone().detach().cpu().numpy(),
                'cal_metadata': cal_metadata
            }
            return confession_out, metadata, z_scratch
        else:
            metadata = {
                'confessional_triggered': False,
                'v_t_trigger_values': v_t_trigger.detach().cpu().numpy(),
                'scratchpad_state': z_scratch.clone().detach().cpu().numpy(),
                'cal_metadata': {}
            }
            return x, metadata, z_scratch
