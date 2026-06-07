from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ACLine:
    line_id: str | int
    from_bus: str | int
    to_bus: str | int
    y_mag: float
    y_angle_rad: float
    rate_pu: float | None = None


def active_flow_pu(v_i: float, theta_i: float, v_j: float, theta_j: float, y_mag: float, y_angle: float) -> float:
    """AC active flow from i to j using admittance magnitude/angle form.

    Matches the common expression:
    P_ij = V_i^2 |Y_ij| cos(phi_ij) - V_i V_j |Y_ij| cos(theta_i - theta_j + phi_ij)
    where phi_ij is the admittance angle.
    """
    return v_i * v_i * y_mag * math.cos(y_angle) - v_i * v_j * y_mag * math.cos(theta_i - theta_j + y_angle)


def reactive_flow_pu(v_i: float, theta_i: float, v_j: float, theta_j: float, y_mag: float, y_angle: float) -> float:
    """AC reactive flow from i to j using admittance magnitude/angle form."""
    return -v_i * v_i * y_mag * math.sin(y_angle) - v_i * v_j * y_mag * math.sin(theta_i - theta_j + y_angle)


def apparent_flow_sq_pu(p_ij: float, q_ij: float) -> float:
    return p_ij * p_ij + q_ij * q_ij


def dc_flow_pu(theta_i: float, theta_j: float, x_pu: float) -> float:
    if x_pu == 0:
        raise ValueError("Line reactance x_pu must be nonzero for DC flow.")
    return (theta_i - theta_j) / x_pu


def voltage_bounds_for_case(case_id: str) -> tuple[float, float] | None:
    if case_id == "case_a_dc_uc":
        return None
    if case_id == "case_b_ac_uc_benders":
        return 0.9, 1.1
    if case_id == "case_c_ac_uc_relaxed_voltage":
        return 0.5, 1.5
    raise ValueError(f"Unknown case id: {case_id}")

