"""Auto mode handler for the simulated Pi client."""

from __future__ import annotations

import math
import random
import time
from typing import Dict, Iterable, List, Tuple

# Autonomous navigation state (GPS-style mock)
_route_active = False
_route_destination = ""
_route_settings: Dict[str, object] = {}
_current_lat = 40.7128  # Starting at NYC coordinates
_current_lng = -74.0060
_target_lat = 0.0
_target_lng = 0.0
_route_progress = 0.0
_route_start_time = 0.0
_simulated_speed = 0.0  # autonomous speed
_waypoints: List[Tuple[float, float]] = []
_current_waypoint = 0
_route_rng: random.Random | None = None
_route_seed_repr = ""

# Grid-based navigation state (from server A* waypoints)
_grid_route_active = False
_grid_waypoints: List[Dict[str, object]] = []
_grid_current_index = 0
_grid_start_time = 0.0
_grid_speed_cells_per_sec = 0.5  # simulation speed across cells


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sync_manual_autonomy_flags() -> None:
    try:
        import sim_manual_mode

        sim_manual_mode.mark_autonomous_active(_route_active, _grid_route_active)
    except Exception:
        # Manual module can be unavailable during isolated tests.
        pass


def _build_route_rng(destination: str, settings: Dict[str, object]) -> random.Random:
    seed = settings.get("seed")
    if seed is None:
        key = f"{destination.lower().strip()}|{settings.get('route_type', 'fastest')}|{settings.get('max_speed', 35)}"
        seed = key
    return random.Random(str(seed))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


# ---------------------------------------------------------------------------
# Route management
# ---------------------------------------------------------------------------

def start_autonomous_route(destination: str, settings: Dict[str, object]) -> None:
    global _route_active, _route_destination, _route_settings, _route_start_time, _current_waypoint
    global _route_progress, _route_rng, _route_seed_repr
    _route_active = True
    _route_destination = destination
    _route_settings = dict(settings)
    _route_start_time = time.time()
    _current_waypoint = 0
    _route_progress = 0.0
    _route_rng = _build_route_rng(destination, _route_settings)
    _route_seed_repr = str(_route_settings.get("seed", destination))
    generate_mock_route(destination, rng=_route_rng)
    print(f"[ROUTE] Started autonomous navigation to: {destination}")
    print(f"[ROUTE] Settings: {_route_settings}")
    _sync_manual_autonomy_flags()


def stop_autonomous_route(reason: str = "user_request") -> None:
    global _route_active, _simulated_speed
    if _route_active:
        print(f"[ROUTE] Stopped autonomous navigation - reason: {reason}")
    _route_active = False
    _simulated_speed = 0.0
    _sync_manual_autonomy_flags()


def generate_mock_route(destination: str, *, rng: random.Random | None = None) -> Iterable[Tuple[float, float]]:
    """Generate mock GPS waypoints for a destination."""
    global _waypoints, _target_lat, _target_lng, _current_lat, _current_lng
    rng = rng or _route_rng or random
    destinations = {
        "home": (40.7500, -73.9850),
        "work": (40.7589, -73.9851),
        "mall": (40.7614, -73.9776),
        "airport": (40.6413, -73.7781),
        "downtown": (40.7831, -73.9712),
    }
    dest_lower = destination.lower()
    for key, coords in destinations.items():
        if key in dest_lower:
            _target_lat, _target_lng = coords
            break
    else:
        _target_lat = _current_lat + rng.uniform(-0.05, 0.05)
        _target_lng = _current_lng + rng.uniform(-0.05, 0.05)
    num_waypoints = rng.randint(3, 8)
    _waypoints = []
    for i in range(num_waypoints + 1):
        progress = i / num_waypoints
        lat = _current_lat + (_target_lat - _current_lat) * progress
        lng = _current_lng + (_target_lng - _current_lng) * progress
        if 0 < i < num_waypoints:
            lat += rng.uniform(-0.002, 0.002)
            lng += rng.uniform(-0.002, 0.002)
        _waypoints.append((lat, lng))
    print(f"[ROUTE] Generated {len(_waypoints)} waypoints to {destination}")
    return _waypoints


def calculate_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    lat_diff = lat2 - lat1
    lng_diff = lng2 - lng1
    return math.sqrt(lat_diff ** 2 + lng_diff ** 2) * 69


def get_navigation_instruction(current_waypoint_idx: int) -> str:
    if current_waypoint_idx >= len(_waypoints) - 1:
        return "Arrive at destination"
    instructions = [
        "Continue straight",
        "Turn right on Oak Street",
        "Turn left on Main Street",
        "Take the ramp to Highway 95",
        "Exit at Downtown",
        "Turn right on Broadway",
        "Turn left on 5th Avenue",
    ]
    rng = _route_rng or random
    return rng.choice(instructions)


# ---------------------------------------------------------------------------
# Simulation loops
# ---------------------------------------------------------------------------

async def simulate_autonomous_navigation() -> Dict[str, object] | None:
    global _current_lat, _current_lng, _route_progress, _simulated_speed, _current_waypoint
    if not _route_active or not _waypoints:
        return None
    if _current_waypoint < len(_waypoints):
        target_lat, target_lng = _waypoints[_current_waypoint]
        lat_step = (target_lat - _current_lat) * 0.0001
        lng_step = (target_lng - _current_lng) * 0.0001
        _current_lat += lat_step
        _current_lng += lng_step
        distance_to_waypoint = calculate_distance(_current_lat, _current_lng, target_lat, target_lng)
        if distance_to_waypoint < 0.1:
            _current_waypoint += 1
            print(f"[NAV] Reached waypoint {_current_waypoint}/{len(_waypoints)}")
        _route_progress = _current_waypoint / max(1, len(_waypoints) - 1)
        rng = _route_rng or random
        route_type = _route_settings.get("route_type")
        if route_type == "eco":
            _simulated_speed = rng.uniform(20, 30)
        elif route_type == "safe":
            _simulated_speed = rng.uniform(15, 25)
        else:
            _simulated_speed = rng.uniform(25, 35)
        _simulated_speed += rng.uniform(-2, 2)
        _simulated_speed = _clamp(_simulated_speed, 0, _route_settings.get("max_speed", 35))
        remaining_distance = calculate_distance(_current_lat, _current_lng, _target_lat, _target_lng)
        eta_minutes = 0
        if _simulated_speed > 0:
            eta_minutes = max(1, int(remaining_distance / (_simulated_speed / 60)))
        status = "navigating" if _current_waypoint < len(_waypoints) else "arrived"
        if status == "arrived":
            stop_autonomous_route("destination_reached")
        return {
            "status": status,
            "destination": _route_destination,
            "current_lat": round(_current_lat, 6),
            "current_lng": round(_current_lng, 6),
            "distance_remaining": remaining_distance,
            "eta_minutes": eta_minutes,
            "next_instruction": get_navigation_instruction(_current_waypoint),
            "route_progress": min(1.0, _route_progress),
            "current_speed": int(_simulated_speed),
            "speed_limit": _route_settings.get("max_speed", 35),
            "seed": _route_seed_repr,
        }
    return None


async def simulate_grid_navigation() -> Dict[str, object] | None:
    global _grid_current_index, _route_progress, _simulated_speed
    if not _grid_route_active or not _grid_waypoints:
        return None
    total = len(_grid_waypoints)
    elapsed = time.time() - _grid_start_time
    target_index = int(elapsed * _grid_speed_cells_per_sec)
    if target_index >= total:
        target_index = total - 1
    if target_index != _grid_current_index:
        print(f"[NAV] Reached waypoint {target_index + 1}/{total}")
        _grid_current_index = target_index
    progress = _grid_current_index / max(1, total - 1)
    _route_progress = progress
    _simulated_speed = 20 + 10 * progress
    current_wp = _grid_waypoints[_grid_current_index]
    status = "navigating" if _grid_current_index < total - 1 else "arrived"
    if status == "arrived":
        stop_grid_route()
    return {
        "status": status,
        "destination": f"grid:{current_wp.get('row')},{current_wp.get('col')}",
        "current_lat": round(_current_lat, 6),
        "current_lng": round(_current_lng, 6),
        "distance_remaining": max(0, (total - 1 - _grid_current_index)),
        "eta_minutes": max(0, int((total - 1 - _grid_current_index) / max(0.1, _grid_speed_cells_per_sec * 60))),
        "next_instruction": f"Head {current_wp.get('heading', 'E')} to cell ({current_wp.get('row')},{current_wp.get('col')})",
        "route_progress": progress,
        "current_speed": int(_simulated_speed),
        "speed_limit": _route_settings.get("max_speed", 35) if _route_settings else 35,
    }


# ---------------------------------------------------------------------------
# Grid route management and state accessors
# ---------------------------------------------------------------------------

def start_grid_route(waypoints: Iterable[Dict[str, object]], start_time: float | None = None) -> None:
    global _grid_route_active, _grid_waypoints, _grid_current_index, _grid_start_time
    _grid_route_active = True
    _grid_waypoints = list(waypoints)
    _grid_current_index = 0
    _grid_start_time = start_time if start_time is not None else time.time()
    print(f"[ROUTE] Grid route received: {len(_grid_waypoints)} waypoints")
    if _grid_waypoints:
        print(f"[ROUTE] First WP: {_grid_waypoints[0]}, Last WP: {_grid_waypoints[-1]}")
    _sync_manual_autonomy_flags()


def stop_grid_route() -> None:
    global _grid_route_active
    if _grid_route_active:
        print("[ROUTE] Grid route stopped")
    _grid_route_active = False
    _sync_manual_autonomy_flags()


def get_auto_state() -> Dict[str, object]:
    return {
        "route_active": _route_active,
        "grid_route_active": _grid_route_active,
        "simulated_speed": _simulated_speed,
        "current_lat": _current_lat,
        "current_lng": _current_lng,
        "route_progress": _route_progress,
        "seed": _route_seed_repr,
    }
