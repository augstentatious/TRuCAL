"""
TinyConfessionalLayer Module

Recursive think/act confessional loop with template cycling and early stopping via coherence.
Implements the core THINK-ACT-COHERENCE recursion pattern inspired by LC-NE neural dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vulnerability_spotter import VulnerabilitySpotter


class TinyConfessionalLayer(nn.Module):
    """
    Recursive think/act confessional loop, template cycling, early stop via coherence.
    Returns (output tensor, metadata dict).
    """
    TEMPLATES = ["prior", "evidence", "posterior", "relational_check", "moral", "action"]

    def __init__(self, d_model=256, n_inner=6, max_cycles=16, trigger_thresh=0.04, per_dim_kl=False):
        super().__init__()
        self.d_model = d_model
        self.trigger_thresh = trigger_thresh
        self.per_dim_kl = per_dim_kl
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
            v_t, vs_metadata = self.vulnerability_spotter(z_state, attention_weights, audit_mode=audit_mode)

            # Trigger recursion based on v_t
            v_t_score_batch = torch.mean(v_t, dim=1).squeeze(-1)

            triggered_batch = v_t_score_batch > self.trigger_thresh

            if audit_mode:
                print(f"Confessional triggered: {triggered_batch.any().item()} | Cycle: {cycles_run} | Mean v_t: {v_t_score_batch.mean().item():.4f}")

            if torch.any(triggered_batch):
                triggered = True

                # Confessional step (vectorized inner loop)
                for inner_step in range(self.n_inner):
                    template_name = self.TEMPLATES[inner_step % len(self.TEMPLATES)]
                    template_steps.append(template_name)

                    # Vectorized template application with masking
                    templated_z_state = self.template_proj[template_name](z_state)
                    # Use torch.where to conditionally apply template based on trigger
                    z_state = torch.where(
                        triggered_batch.unsqueeze(-1).unsqueeze(-1),
                        templated_z_state,
                        z_state
                    )

            # Act step
            act_input = torch.cat([y_state, z_state], dim=-1)
            y_state = self.act_net(act_input)

            # Coherence computation (fixed: compare with previous cycle, not self)
            if len(tracker) > 1:
                # Compare current z_state with previous cycle's state (tracker[-2])
                recent_avg_for_coherence = tracker[-2]

                recent_avg_flat = recent_avg_for_coherence.view(-1, d_model)
                z_state_flat = z_state.view(-1, d_model)

                sim_coherence = F.cosine_similarity(z_state_flat, recent_avg_flat, dim=-1).mean().item()

                if z_state.numel() > 0 and tracker[-2].numel() > 0:
                    try:
                        if self.per_dim_kl:
                            # Per-dimension KL: preserves dimensional structure, avoids 1D collapse
                            z_flat = z_state.view(-1, d_model)  # (batch*seq, d_model)
                            prior_flat = tracker[-2].view(-1, d_model)
                            
                            # Compute per-dimension statistics
                            curr_mu = z_flat.mean(dim=0)  # (d_model,)
                            curr_std = z_flat.std(dim=0) + 1e-6
                            prior_mu = prior_flat.mean(dim=0)
                            prior_std = prior_flat.std(dim=0) + 1e-6
                            
                            # KL divergence per dimension, then average
                            kl_per_dim = torch.distributions.kl_divergence(
                                torch.distributions.Normal(curr_mu, curr_std),
                                torch.distributions.Normal(prior_mu, prior_std)
                            )
                            kl_div = kl_per_dim.mean().item()
                        else:
                            # Global KL: single scalar mean/std (original, faster but crude)
                            curr_mu, curr_std = z_state.mean(), z_state.std() + 1e-6
                            prior_mu, prior_std = tracker[-2].mean(), tracker[-2].std() + 1e-6
                            kl_div = torch.distributions.kl_divergence(
                                torch.distributions.Normal(curr_mu, curr_std),
                                torch.distributions.Normal(prior_mu, prior_std)
                            ).item()
                        
                        bayes_align = 1 / (1 + kl_div)
                        final_coherence = 0.7 * sim_coherence + 0.3 * bayes_align
                    except Exception as e:
                        if audit_mode:
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
