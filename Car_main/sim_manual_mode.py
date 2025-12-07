"""Manual mode handler for the simulated Pi client."""

from __future__ import annotations

import time
from typing import Dict


# Manual mode state
_current_speed = 0.0
_current_angle = 0.0
_emergency_active = False
_route_active = False
_grid_route_active = False
_last_command_ts = 0.0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _update_last_command_timestamp() -> None:
    global _last_command_ts
    _last_command_ts = time.time()


def apply_manual_control(speed: float, angle: float) -> bool:
    """Simulate applying manual control values from joystick."""
    global _current_speed, _current_angle
    if _emergency_active:
        print("[BLOCKED] Manual control blocked - emergency stop active")
        return False
    if _route_active or _grid_route_active:
        print("[BLOCKED] Manual control blocked - autonomous mode active")
        return False
    s = _clamp(float(speed), -1.0, 1.0)
    a = _clamp(float(angle), -1.0, 1.0)
    _current_speed = s
    _current_angle = a
    _update_last_command_timestamp()
    motor_pwm = int(abs(s) * 255)
    motor_direction = "FORWARD" if s >= 0 else "REVERSE"
    servo_center = 1500
    servo_range = 500
    servo_position = servo_center + int(a * servo_range)
    turn_factor = a * 0.5
    left_motor = _clamp(s + turn_factor, -1.0, 1.0)
    right_motor = _clamp(s - turn_factor, -1.0, 1.0)
    left_pwm = int(abs(left_motor) * 255)
    right_pwm = int(abs(right_motor) * 255)
    speed_pct = int(s * 100)
    angle_deg = int(a * 45)
    print(f"[CONTROL] Manual control -> speed={s:.2f} ({speed_pct}%), angle={a:.2f} ({angle_deg}°)")
    print(f"[MOTORS] PWM={motor_pwm} {motor_direction}, Servo={servo_position}μs, Differential L={left_pwm} R={right_pwm}")
    return True


def set_emergency_active(active: bool) -> None:
    global _emergency_active
    previous = _emergency_active
    _emergency_active = bool(active)
    if previous and not _emergency_active:
        print("✅ Emergency mode cleared - normal operation restored")
    elif not previous and _emergency_active:
        print("[EMERGENCY] Manual mode locked")


def clear_emergency() -> None:
    set_emergency_active(False)


def mark_autonomous_active(route_active: bool, grid_route_active: bool = False) -> None:
    global _route_active, _grid_route_active
    _route_active = bool(route_active)
    _grid_route_active = bool(grid_route_active)


def set_manual_flags(emergency_active: bool, route_active: bool, grid_route_active: bool) -> None:
    set_emergency_active(emergency_active)
    mark_autonomous_active(route_active, grid_route_active)


def reset_state() -> None:
    global _current_speed, _current_angle, _emergency_active, _route_active, _grid_route_active, _last_command_ts
    _current_speed = 0.0
    _current_angle = 0.0
    _emergency_active = False
    _route_active = False
    _grid_route_active = False
    _last_command_ts = 0.0


def get_manual_state() -> Dict[str, float | bool]:
    return {
        "current_speed": _current_speed,
        "current_angle": _current_angle,
        "emergency_active": _emergency_active,
        "route_active": _route_active,
        "grid_route_active": _grid_route_active,
        "last_command_ts": _last_command_ts,
    }
