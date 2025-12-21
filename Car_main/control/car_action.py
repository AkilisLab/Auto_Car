#!/usr/bin/env python3
"""Shared car action helpers for manual_rc.

Provides CarActions which exposes movement loops that accept a threading.Event
to signal stop. These loops are intended to be run in background threads and
will stop motors in their finally block for safety.
"""
import time
from typing import Optional


class CarActions:
    def __init__(self, bot, base_speed: int = 100):
        self.bot = bot
        self.base_speed = base_speed

    def stop_motors(self) -> None:
        for mid in range(4):
            try:
                self.bot.Ctrl_Car(mid, 0, 0)
            except Exception:
                pass

    def set_all_motors(self, direction: int, speed: int) -> None:
        for mid in range(4):
            try:
                self.bot.Ctrl_Car(mid, direction, speed)
            except Exception:
                pass

    def blink_all(self, color: int, on_time: float = 0.12, off_time: float = 0.12) -> None:
        try:
            self.bot.Ctrl_WQ2812_ALL(1, color)
            time.sleep(on_time)
            self.bot.Ctrl_WQ2812_ALL(0, 0)
            time.sleep(off_time)
        except Exception:
            time.sleep(on_time + off_time)

    # movement loops
    def forward_loop(self, stop_event) -> None:
        try:
            while not stop_event.is_set():
                self.set_all_motors(0, self.base_speed)
                self.blink_all(2)
        finally:
            self.stop_motors()

    def backward_loop(self, stop_event) -> None:
        try:
            while not stop_event.is_set():
                self.set_all_motors(1, self.base_speed)
                self.blink_all(0)
        finally:
            self.stop_motors()

    def left_loop(self, stop_event) -> None:
        try:
            while not stop_event.is_set():
                try:
                    self.bot.Ctrl_Car(0, 1, int(self.base_speed * 0.6))
                    self.bot.Ctrl_Car(1, 1, int(self.base_speed * 0.6))
                    self.bot.Ctrl_Car(2, 0, int(self.base_speed * 0.6))
                    self.bot.Ctrl_Car(3, 0, int(self.base_speed * 0.6))
                except Exception:
                    pass
                # blink LEDs while turning
                self.blink_all(1)
        finally:
            self.stop_motors()

    def right_loop(self, stop_event) -> None:
        try:
            while not stop_event.is_set():
                try:
                    self.bot.Ctrl_Car(0, 0, int(self.base_speed * 0.6))
                    self.bot.Ctrl_Car(1, 0, int(self.base_speed * 0.6))
                    self.bot.Ctrl_Car(2, 1, int(self.base_speed * 0.6))
                    self.bot.Ctrl_Car(3, 1, int(self.base_speed * 0.6))
                except Exception:
                    pass
                # blink LEDs while turning
                self.blink_all(2)
        finally:
            self.stop_motors()

    def side_left_loop(self, stop_event) -> None:
        """Strafe left (mecanum-like): motors set so vehicle moves left without yaw.

        This assumes motor layout where setting opposing corners to
        opposite directions produces lateral motion. Adjust if hardware
        mapping differs.
        """
        try:
            while not stop_event.is_set():
                try:
                    # Example mapping for left strafe:
                    # front-left: backward, front-right: forward,
                    # rear-left: forward, rear-right: backward
                    self.bot.Ctrl_Car(0, 1, int(self.base_speed * 0.7))
                    self.bot.Ctrl_Car(1, 0, int(self.base_speed * 0.7))
                    self.bot.Ctrl_Car(2, 0, int(self.base_speed * 0.7))
                    self.bot.Ctrl_Car(3, 1, int(self.base_speed * 0.7))
                except Exception:
                    pass
                # blink LEDs while strafing
                self.blink_all(1)
        finally:
            self.stop_motors()

    def side_right_loop(self, stop_event) -> None:
        """Strafe right (mecanum-like)."""
        try:
            while not stop_event.is_set():
                try:
                    # inverse of side_left_loop
                    self.bot.Ctrl_Car(0, 0, int(self.base_speed * 0.7))
                    self.bot.Ctrl_Car(1, 1, int(self.base_speed * 0.7))
                    self.bot.Ctrl_Car(2, 1, int(self.base_speed * 0.7))
                    self.bot.Ctrl_Car(3, 0, int(self.base_speed * 0.7))
                except Exception:
                    pass
                self.blink_all(2)
        finally:
            self.stop_motors()

    def stop_loop(self, stop_event) -> None:
        try:
            while not stop_event.is_set():
                self.stop_motors()
                time.sleep(0.05)
        finally:
            self.stop_motors()


__all__ = ["CarActions"]
