# ACE-FP-EXPECT: clean
# CATEGORY: 07_non_ai_false_detection
# SOURCE: HVAC thermostat control loop (hysteresis / bang-bang setpoint controller)
# WHY-CORRECT: "temperature" here is a physical room temperature in Celsius read from a
#              sensor and compared against a setpoint to drive a heater relay. There is no
#              LLM, no sampling temperature, no model.generate(...) call anywhere.
# EXPECTED-WRONG: keyword "temperature" (and "setpoint") -> false "LLM sampling parameter"
#                 detection -> spurious "temperature too high / not configured" finding.
# CORRECT-VERDICT: no findings
"""Bang-bang HVAC thermostat: drives a heater relay from room temperature readings."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ThermostatConfig:
    """Setpoint and hysteresis band for the heating controller (degrees Celsius)."""

    setpoint_c: float = 21.0
    hysteresis_c: float = 0.5
    min_temperature_c: float = 5.0
    max_temperature_c: float = 30.0


class Thermostat:
    """Hysteresis controller that turns a heater on/off around a setpoint."""

    def __init__(self, config: ThermostatConfig | None = None) -> None:
        self.config = config or ThermostatConfig()
        self._heater_on = False

    def update(self, temperature_c: float) -> bool:
        """Return whether the heater relay should be energized for this reading."""
        cfg = self.config
        if temperature_c <= cfg.setpoint_c - cfg.hysteresis_c:
            self._heater_on = True
        elif temperature_c >= cfg.setpoint_c + cfg.hysteresis_c:
            self._heater_on = False
        return self._heater_on

    def is_out_of_range(self, temperature_c: float) -> bool:
        """Flag readings outside the safe operating envelope of the sensor."""
        cfg = self.config
        return not (cfg.min_temperature_c <= temperature_c <= cfg.max_temperature_c)


def run_cycle(readings: list[float], config: ThermostatConfig | None = None) -> list[bool]:
    """Replay a sequence of sensor readings and record heater state per tick."""
    thermostat = Thermostat(config)
    return [thermostat.update(temperature) for temperature in readings]
