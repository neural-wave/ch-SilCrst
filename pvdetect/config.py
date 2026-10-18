"""Fixed hyper-parameters of the baseline.

Every series is stored on a UTC grid of 96 quarter-hour slots per day, so DST days
(92/100 local intervals) need no special handling and solar noon stays put all year.
Slot windows below are UTC and generic for Switzerland (no customer location is used).
"""
from dataclasses import dataclass

SLOT_SECONDS = 900
SLOTS_PER_DAY = 96
DAY_SECONDS = 86400


def slots(start_hour: float, end_hour: float) -> tuple[int, ...]:
    """UTC slots covering [start_hour, end_hour)."""
    return tuple(range(int(start_hour * 4), int(end_hour * 4)))


@dataclass(frozen=True)
class Config:
    # Always dark in Switzerland, any season: 20:00-24:00 and 00:00-03:30 UTC.
    dark_slots: tuple[int, ...] = slots(20, 24) + slots(0, 3.5)
    # Always sunlit and productive, even in December: 09:00-14:00 UTC.
    midday_slots: tuple[int, ...] = slots(9, 14)
    # Widest possible daylight span (summer): 04:00-19:30 UTC.
    daylight_slots: tuple[int, ...] = slots(4, 19.5)

    # Standby load of a day = this percentile of its dark slots.
    base_percentile: float = 20.0
    # The reference standby never drops below this fraction of the customer's median standby.
    base_floor_frac: float = 0.5
    # A slot is "below standby" when kWh < below_frac * standby (or < abs_floor_kwh).
    below_frac: float = 0.5
    abs_floor_kwh: float = 0.001
    # PV-like day: enough midday slots below standby while the night is not.
    pv_day_min_midday_frac: float = 0.3
    pv_day_max_dark_frac: float = 0.15

    # A day is usable when it has at least this many non-missing slots in each window.
    min_dark_valid: int = 20
    min_midday_valid: int = 16
    min_valid_days: int = 30

    # Detection score: best rate of PV-like days over a rolling window.
    rolling_days: int = 60
    rolling_min_valid: int = 30

    # Change point: coarse seasonal step fit, then Bernoulli refinement around it.
    min_days_after_change: int = 30
    refine_before_days: int = 90
    refine_after_days: int = 45
    # Second refinement: PV often starts with partial days (high load in summer, weak winter
    # sun) before the first full PV-like day; look back for a sustained run of weak days.
    weak_refine: bool = True
    weak_min_midday_frac: float = 0.08
    weak_lookback_days: int = 150
    # With little history before the date (new meter installed with the PV), use a default
    # background rate instead of skipping the refinement.
    weak_short_history: bool = True
    # A change found this close to the series start means PV was already there.
    preexisting_days: int = 45

    # Capacity features are measured in the first year after the installation
    # (later data may include extensions).
    capacity_window_days: int = 365
    summer_doy: tuple[int, int] = (121, 243)  # May-August


DEFAULT_CONFIG = Config()
