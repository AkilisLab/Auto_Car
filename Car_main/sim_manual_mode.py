"""Manual mode handler for the simulated Pi client."""

from __future__ import annotations

import time
from typing import Dict

try:
    from devices.raspbot import Raspbot  # type: ignore
except Exception as exc:  # hardware layer is optional in some environments
    Raspbot = None  # type: ignore
    _raspbot_import_error = exc
else:
    _raspbot_import_error = None

_raspbot_instance = None
_raspbot_init_failed = False


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


def _get_raspbot():
    global _raspbot_instance, _raspbot_init_failed, _raspbot_import_error
    if _raspbot_instance is not None:
        return _raspbot_instance
    if _raspbot_init_failed:
        return None
    if Raspbot is None:  # type: ignore
        if _raspbot_import_error is not None:
            print(f"[WARN] Raspbot interface unavailable: {_raspbot_import_error}")
            _raspbot_import_error = None
        _raspbot_init_failed = True
        return None
    try:
        _raspbot_instance = Raspbot()  # type: ignore[call-arg]
    except Exception as exc:  # I2C bus might be absent during development
        print(f"[WARN] Unable to initialize Raspbot interface: {exc}")
        _raspbot_init_failed = True
        _raspbot_instance = None
    return _raspbot_instance


def _apply_drive_to_robot(left_value: float, right_value: float) -> None:
    bot = _get_raspbot()
    if not bot:
        return

    def drive_motor(motor_id: int, value: float) -> None:
        abs_value = abs(value)
        # Treat very small magnitudes as zero to avoid humming the motors.
        if abs_value < 0.02:
            pwm = 0
            direction = 0
        else:
            pwm = int(min(255, round(abs_value * 255)))
            direction = 0 if value >= 0 else 1
        try:
            bot.Ctrl_Car(motor_id, direction, pwm)
        except Exception as exc:
            print(f"[WARN] Motor command failed (motor {motor_id}): {exc}")

    # Motors 0/1 are left side, 2/3 are right side in raspbot.py mapping.
    drive_motor(0, left_value)
    drive_motor(1, left_value)
    drive_motor(2, right_value)
    drive_motor(3, right_value)


def _stop_robot_motors() -> None:
    bot = _get_raspbot()
    if not bot:
        return
    for motor_id in range(4):
        try:
            bot.Ctrl_Car(motor_id, 0, 0)
        except Exception as exc:
            print(f"[WARN] Failed to stop motor {motor_id}: {exc}")


def apply_manual_control(speed: float, angle: float) -> bool:
    """Simulate applying manual control values from joystick."""
    global _current_speed, _current_angle
    if _emergency_active:
        print("[BLOCKED] Manual control blocked - emergency stop active")
        _stop_robot_motors()
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
    _apply_drive_to_robot(left_motor, right_motor)
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
        print("[EMERGENCY] control locked")
    if _emergency_active:
        _stop_robot_motors()


def clear_emergency() -> None:
    set_emergency_active(False)


def mark_autonomous_active(route_active: bool, grid_route_active: bool = False) -> None:
    global _route_active, _grid_route_active
    _route_active = bool(route_active)
    _grid_route_active = bool(grid_route_active)
    if _route_active or _grid_route_active:
        _stop_robot_motors()


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
    _stop_robot_motors()


def get_manual_state() -> Dict[str, float | bool]:
    return {
        "current_speed": _current_speed,
        "current_angle": _current_angle,
        "emergency_active": _emergency_active,
        "route_active": _route_active,
        "grid_route_active": _grid_route_active,
        "last_command_ts": _last_command_ts,
    }
