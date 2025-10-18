#!/usr/bin/env python3
"""Manual remote control for Raspbot using keyboard keys.

Controls (press-and-hold semantics):
  w: forward
  s: stop
  x: backward
  a: left
  d: right
  q: LED effect on
  e: LED effect off

Servo controls:
  h: servo left (decrease angle)
  j: servo mid (decrease angle)
  k: servo right (decrease angle)
  u: servo up (increase angle)
  m: servo down (increase angle)

Press Ctrl-C to exit. This script uses the I2C Raspbot interface in
`devices/raspbot.py` (Raspbot class).
"""

import sys
import termios
import tty
import threading
import time

from devices.raspbot import Raspbot
from control.car_action import CarActions
from devices.camera_node import Camera
import cv2


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

    # car action helpers
    actions = CarActions(bot, base_speed=base_speed)

    # camera preview thread control
    # create a persistent Camera instance and a background thread that
    # keeps the capture open. We only toggle whether frames are shown
    # in the window to avoid repeatedly opening/closing the VideoCapture
    # (which can trigger Qt plugin/windowing errors on some systems).
    camera = Camera(index=0, width=640, height=480, fps=15)
    camera_thread = None
    camera_stop = threading.Event()
    # when True the background thread will call cv2.imshow; toggling
    # this avoids reopening the capture. Note: the thread may start on
    # first request and will be stopped and capture released on program
    # exit.
    camera_show = False

    def start_camera():
        """Start showing camera frames. The background thread is started
        on first use and keeps the capture open. Subsequent calls only
        toggle showing frames in the window."""
        nonlocal camera_thread, camera_stop, camera_show
        # start the background thread if needed
        if camera_thread is None or not camera_thread.is_alive():
            camera_stop.clear()

            def cam_loop():
                nonlocal camera_show
                try:
                    # prefer background reader which keeps the cap open
                    if not camera.start():
                        camera.open()
                    while not camera_stop.is_set():
                        ret, frame = camera.read(wait=True)
                        if camera_show and ret and frame is not None:
                            try:
                                cv2.imshow('Camera', frame)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    # allow hiding the window with 'q'
                                    camera_show = False
                                    try:
                                        cv2.destroyWindow('Camera')
                                    except Exception:
                                        pass
                            except Exception:
                                # display may fail on headless systems; ignore
                                pass
                        else:
                            # not showing frames; sleep briefly
                            time.sleep(0.05)
                    try:
                        cv2.destroyWindow('Camera')
                    except Exception:
                        pass
                finally:
                    try:
                        camera.close()
                    except Exception:
                        pass

            camera_thread = threading.Thread(target=cam_loop, daemon=True)
            camera_thread.start()
            # small delay to let the thread open camera if needed
            time.sleep(0.05)

        # request that the window show frames
        camera_show = True

    def stop_camera():
        nonlocal camera_show
        # hide frames, but keep the background thread and capture alive
        # so we can quickly re-show without reopening the VideoCapture.
        camera_show = False

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    print("Manual RC started. Use w/a/s/d to drive, q/e for LEDs, h/j/k/u/m for servos. p to exit.")
    try:
        tty.setcbreak(fd)
        # --- helper functions to reduce duplication ---
        # Use CarActions helpers
        # wrappers removed - use actions.* methods directly

        def launch_action(name, fn):
            nonlocal action_thread, action_stop
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
                launch_action('w', lambda: actions.forward_loop(action_stop))
            elif ch == 'x':
                launch_action('x', lambda: actions.backward_loop(action_stop))
            elif ch == 'a':
                launch_action('a', lambda: actions.left_loop(action_stop))
            elif ch == 'd':
                launch_action('d', lambda: actions.right_loop(action_stop))
            elif ch == 's' or ch == ' ':
                # Launch a persistent stop action that holds motors off until
                # another action is launched.
                launch_action('stop', lambda: actions.stop_loop(action_stop))
            # LEDs
            elif ch == 'q':
                launch_led_effect()
            elif ch == 'v':
                # toggle camera preview. We keep the VideoCapture open in a
                # background thread and only toggle whether frames are shown
                # to avoid repeated open/close which can trigger Qt plugin
                # errors.
                if camera_show:
                    stop_camera()
                    print('Camera stopped')
                else:
                    start_camera()
                    print('Camera started (press v again to stop)')
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
            # stop camera background thread if running
            try:
                if camera_thread is not None and camera_thread.is_alive():
                    # request thread to stop and wait briefly
                    camera_stop.set()
                    camera_thread.join(timeout=1.0)
            except Exception:
                pass
        except Exception:
            pass
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print('\nManual RC exited.')


if __name__ == '__main__':
    manual_control()
