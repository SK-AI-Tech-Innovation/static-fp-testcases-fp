# ACE-FP-EXPECT: clean
# CATEGORY: 07_non_ai_false_detection
# SOURCE: Electrical power transformer model (turns ratio, voltage/current, efficiency)
# WHY-CORRECT: "transformer" here is an electrical step-up/step-down power transformer
#              described by its turns ratio and winding losses. There is no neural network,
#              no attention, no HuggingFace transformers import, no model weights.
# EXPECTED-WRONG: keyword "transformer" / "model" (electrical model) -> false "Transformer
#                 architecture / ML model" detection -> spurious "model loaded without
#                 device map" or "transformers library misuse" finding.
# CORRECT-VERDICT: no findings
"""Ideal/real electrical power transformer: turns ratio, voltage, current, efficiency."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TransformerModel:
    """Lumped-parameter model of a single-phase power transformer."""

    primary_turns: int
    secondary_turns: int
    copper_loss_w: float = 0.0
    core_loss_w: float = 0.0

    @property
    def turns_ratio(self) -> float:
        """Ratio of primary to secondary winding turns."""
        return self.primary_turns / self.secondary_turns

    def secondary_voltage(self, primary_voltage_v: float) -> float:
        """Step the primary voltage through the turns ratio (ideal coupling)."""
        return primary_voltage_v / self.turns_ratio

    def secondary_current(self, primary_current_a: float) -> float:
        """Secondary current implied by ampere-turn balance."""
        return primary_current_a * self.turns_ratio

    def efficiency(self, output_power_w: float) -> float:
        """Power efficiency accounting for copper and core losses."""
        losses = self.copper_loss_w + self.core_loss_w
        input_power = output_power_w + losses
        if input_power <= 0.0:
            raise ValueError("input power must be positive")
        return output_power_w / input_power


def step_down(primary_voltage_v: float, ratio: float) -> float:
    """Convenience helper: apply a turns ratio to a primary voltage."""
    if ratio <= 0.0:
        raise ValueError("turns ratio must be positive")
    return primary_voltage_v / ratio
