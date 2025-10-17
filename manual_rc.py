#!/usr/bin/env python3
"""Manual remote control for Raspbot using keyboard keys.

Controls (press-and-hold semantics):
  w: forward
  s: backward
  a: left
  d: right
  q: LED effect on
  e: LED effect off

Servo controls:
  h: servo left (decrease angle)
  j: servo mid (decrease angle)
  k: servo right (decrease angle)
  u: servo left (increase angle)
  m: servo right (increase angle)

Press Ctrl-C to exit. This script uses the I2C Raspbot interface in
`devices/raspbot.py` (Raspbot class).
"""

import sys
import termios
import tty
import threading
import time

from devices.raspbot import Raspbot


def manual_control():
    bot = Raspbot()

    # servo angle state
    servo_angles = {1: 90, 2: 90}  # servo id -> angle (1 and 2 assumed)
    servo_step = 5

    # motor speed configuration
    base_speed = 100

    # LED effect thread control
    led_thread = None
    led_stop = threading.Event()
    # Continuous action thread for movement keys
    action_thread = None
    action_stop = threading.Event()
    current_action = None

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    print("Manual RC started. Use w/a/s/d to drive, q/e for LEDs, h/j/k/u/m for servos. p to exit.")
    try:
        tty.setcbreak(fd)
        # --- helper functions to reduce duplication ---
        def stop_motors():
            bot.Ctrl_Car(0, 0, 0)
            bot.Ctrl_Car(1, 0, 0)
            bot.Ctrl_Car(2, 0, 0)
            bot.Ctrl_Car(3, 0, 0)

        def set_all_motors(direction, speed):
            # direction: 0 forward, 1 backward
            for mid in range(4):
                bot.Ctrl_Car(mid, direction, speed)

        def blink_all(color, on_time=0.12, off_time=0.12):
            bot.Ctrl_WQ2812_ALL(1, color)
            time.sleep(on_time)
            bot.Ctrl_WQ2812_ALL(0, 0)
            time.sleep(off_time)

        def launch_action(name, fn):
            nonlocal action_thread, action_stop, current_action
            # stop previous action
            if action_thread is not None and action_thread.is_alive():
                action_stop.set()
                action_thread.join(timeout=0.5)
                action_stop.clear()
            current_action = name
            action_thread = threading.Thread(target=fn, daemon=True)
            action_thread.start()

        def launch_led_effect():
            nonlocal led_thread, led_stop
            if led_thread is None or not led_thread.is_alive():
                led_stop.clear()

                def led_effect():
                    try:
                        while not led_stop.is_set():
                            for i in range(256):
                                if led_stop.is_set():
                                    break
                                bot.Ctrl_WQ2812_brightness_ALL(0, 0, i)
                                time.sleep(0.01)
                            for i in range(255, -1, -1):
                                if led_stop.is_set():
                                    break
                                bot.Ctrl_WQ2812_brightness_ALL(0, 0, i)
                                time.sleep(0.01)
                    finally:
                        pass

                led_thread = threading.Thread(target=led_effect, daemon=True)
                led_thread.start()
        while True:
            ch = sys.stdin.read(1)
            if not ch:
                break
            # stop LED effect unless we're toggling it
            if ch != 'q' and led_thread is not None and led_thread.is_alive():
                led_stop.set()
                led_thread.join(timeout=0.5)
                led_thread = None
                led_stop.clear()

            # movement mapping (simple differential control)
            if ch == 'w':
                def action_forward():
                    try:
                        while not action_stop.is_set():
                            set_all_motors(0, base_speed)
                            blink_all(2)
                    finally:
                        stop_motors()

                launch_action('w', action_forward)
            elif ch == 'x':
                def action_backward():
                    try:
                        while not action_stop.is_set():
                            set_all_motors(1, base_speed)
                            blink_all(0)
                    finally:
                        stop_motors()

                launch_action('x', action_backward)
            elif ch == 'a':
                def action_left():
                    try:
                        while not action_stop.is_set():
                            bot.Ctrl_Car(0, 1, int(base_speed * 0.3))
                            bot.Ctrl_Car(1, 1, int(base_speed * 0.3))
                            bot.Ctrl_Car(2, 0, int(base_speed * 0.3))
                            bot.Ctrl_Car(3, 0, int(base_speed * 0.3))
                            # blink LEDs while turning
                            blink_all(1)
                    finally:
                        stop_motors()

                launch_action('a', action_left)
            elif ch == 'd':
                def action_right():
                    try:
                        while not action_stop.is_set():
                            bot.Ctrl_Car(0, 0, int(base_speed * 0.3))
                            bot.Ctrl_Car(1, 0, int(base_speed * 0.3))
                            bot.Ctrl_Car(2, 1, int(base_speed * 0.3))
                            bot.Ctrl_Car(3, 1, int(base_speed * 0.3))
                            # blink LEDs while turning
                            blink_all(2)
                    finally:
                        stop_motors()

                launch_action('d', action_right)
            elif ch == 's' or ch == ' ':
                # Launch a persistent stop action that holds motors off until
                # another action is launched.
                def action_stop_motors():
                    try:
                        # actively keep motors off until stop signaled
                        while not action_stop.is_set():
                            stop_motors()
                            time.sleep(0.05)
                    finally:
                        stop_motors()

                launch_action('stop', action_stop_motors)
            # LEDs
            elif ch == 'q':
                launch_led_effect()
            elif ch == 'e':
                bot.Ctrl_WQ2812_ALL(0, 0)  # turn off
            # Servo controls (left servo id=1, right servo id=2)
            elif ch == 'h':
                servo_angles[1] = min(180, servo_angles[1] + servo_step)
                bot.Ctrl_Servo(1, servo_angles[1])
            elif ch == 'u':
                servo_angles[2] = min(180, servo_angles[2] + servo_step)
                bot.Ctrl_Servo(2, servo_angles[2])
            elif ch == 'j':
                # center both servos
                servo_angles[1] = 90
                servo_angles[2] = 0
                bot.Ctrl_Servo(1, servo_angles[1])
                bot.Ctrl_Servo(2, servo_angles[2])
            elif ch == 'm':
                servo_angles[2] = max(0, servo_angles[2] - servo_step)
                bot.Ctrl_Servo(2, servo_angles[2])
            elif ch == 'k':
                servo_angles[1] = max(0, servo_angles[1] - servo_step)
                bot.Ctrl_Servo(1, servo_angles[1])
            elif ch == 'p':  # p to stop
                break
            else:
                print(f'Unmapped key: {repr(ch)}')

    except KeyboardInterrupt:
        pass
    finally:
        # stop motors and turn off LEDs
        try:
            bot.Ctrl_Car(0, 0, 0)
            bot.Ctrl_Car(1, 0, 0)
            bot.Ctrl_Car(2, 0, 0)
            bot.Ctrl_Car(3, 0, 0)
            # stop LED thread if running
            try:
                if led_thread is not None and led_thread.is_alive():
                    led_stop.set()
                    led_thread.join(timeout=0.5)
            except Exception:
                pass
            # stop action thread if running
            try:
                if action_thread is not None and action_thread.is_alive():
                    action_stop.set()
                    action_thread.join(timeout=0.5)
            except Exception:
                pass
            bot.Ctrl_WQ2812_ALL(0, 0)
        except Exception:
            pass
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print('\nManual RC exited.')


if __name__ == '__main__':
    manual_control()
