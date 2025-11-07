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
import os
from datetime import datetime
import select

from devices.raspbot import Raspbot
from control.car_action import CarActions
from devices.camera_node import Camera
from devices.oled import OLEDDisplay
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

    # recording state for video capture (toggled with 'r' while camera is on)
    recording = False
    recording_writer = None
    recording_filename = None
    recording_lock = threading.Lock()

    # OLED display (optional)
    oled = None
    try:
        oled = OLEDDisplay()
    except Exception:
        oled = None

    # IR sensor reading thread control
    ir_thread = None
    ir_stop = threading.Event()
    ir_active = False
    # Ultrasonic (sonic) sensor thread control
    sonic_thread = None
    sonic_stop = threading.Event()
    sonic_active = False

    def start_camera():
        """Start showing camera frames. The background thread is started
        on first use and keeps the capture open. Subsequent calls only
        toggle showing frames in the window."""
        nonlocal camera_thread, camera_stop, camera_show
        # start the background thread if needed
        if camera_thread is None or not camera_thread.is_alive():
            camera_stop.clear()

            # ensure the camera background reader is running before the
            # display thread starts so read() will return quickly.
            if not camera.start():
                # if start failed, try to open once and then start again
                try:
                    camera.open()
                    camera.start()
                except Exception:
                    pass

            def cam_loop():
                # camera_show and recording_* variables are owned by the
                # enclosing manual_control scope; we assign to
                # recording_writer here so declare nonlocal for that name.
                nonlocal camera_show, recording_writer, recording_filename, recording
                try:
                    while not camera_stop.is_set():
                        ret, frame = camera.read(wait=True)
                        if ret and frame is not None:
                            # If recording was requested but no writer exists yet,
                            # lazily create the VideoWriter using the incoming
                            # frame size so we don't need to query the camera
                            # object for width/height.
                            if recording:
                                if recording_writer is None:
                                    try:
                                        os.makedirs('recordings', exist_ok=True)
                                        if not recording_filename:
                                            recording_filename = time.strftime('recordings/rec_%Y%m%d_%H%M%S.avi')
                                        h, w = frame.shape[:2]
                                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                                        fps = getattr(camera, 'fps', 20) or 20
                                        recording_writer = cv2.VideoWriter(recording_filename, fourcc, fps, (w, h))
                                    except Exception:
                                        recording_writer = None
                                else:
                                    # write frame to disk; ignore write failures
                                    try:
                                        recording_writer.write(frame)
                                    except Exception:
                                        pass
                            else:
                                # if recording stopped and a writer exists, release it
                                if recording_writer is not None:
                                    try:
                                        recording_writer.release()
                                    except Exception:
                                        pass
                                    recording_writer = None
                                    recording_filename = None

                            # do not call GUI display functions from the background
                            # thread; the main thread will perform cv2.imshow and
                            # handle window events. Sleep briefly here to avoid
                            # a busy loop when not recording.
                            time.sleep(0.01)
                        # background thread exiting; main thread owns windows
                        pass
                finally:
                    # Do not close the camera here; the outer cleanup will
                    # release the capture. Closing here caused the device to
                    # be released while other threads might still expect it
                    # and made re-opening unreliable on some platforms.
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
    print("Manual RC started. Use w/a/s/d to drive, z/c to strafe, q/e for LEDs, h/j/k/u/m for servos, v to toggle camera, f to snapshot, r to record, i to toggle IR, o to toggle ultrasonic, p to exit.")
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
            # poll stdin with a short timeout so we can update the
            # camera preview from the main thread (cv2 GUI operations
            # must typically run in the main thread on many platforms).
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                ch = sys.stdin.read(1)
                # EOF -> exit
                if not ch:
                    break
            else:
                ch = None

            # when no keypress, give the GUI a chance to update
            if ch is None:
                if camera_show:
                    try:
                        ret, frame = camera.get_frame()
                    except Exception:
                        ret, frame = False, None
                    if ret and frame is not None:
                        try:
                            cv2.imshow('Camera', frame)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                camera_show = False
                                try:
                                    cv2.destroyWindow('Camera')
                                except Exception:
                                    pass
                        except Exception:
                            # headless or other GUI error; ignore and keep loop
                            pass
                # no key to process; continue polling
                continue
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
            elif ch == 'z':
                # strafe left
                launch_action('z', lambda: actions.side_left_loop(action_stop))
            elif ch == 'c':
                # strafe right
                launch_action('c', lambda: actions.side_right_loop(action_stop))
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
            elif ch == 'i':
                # toggle IR (line) sensor live readout
                if not ir_active:
                    # start IR read thread
                    ir_stop.clear()

                    def ir_loop():
                        try:
                            while not ir_stop.is_set():
                                try:
                                    data = bot.read_data_array(0x0a, 1)
                                except Exception:
                                    data = None
                                if data:
                                    try:
                                        val = int(data[0])
                                    except Exception:
                                        val = 0
                                else:
                                    val = 0
                                # bit mapping: x1 = bit3, x2=bit2, x3=bit1, x4=bit0
                                x1 = (val >> 3) & 0x01
                                x2 = (val >> 2) & 0x01
                                x3 = (val >> 1) & 0x01
                                x4 = val & 0x01
                                # Map 1->'W'(white) 0->'D'(dark)
                                s1 = 'W' if x1 else 'D'
                                s2 = 'W' if x2 else 'D'
                                s3 = 'W' if x3 else 'D'
                                s4 = 'W' if x4 else 'D'
                                lines = [f'IR1:{s1} IR2:{s2}', f'IR3:{s3} IR4:{s4}']
                                if oled is not None:
                                    try:
                                        # update only the top two lines so other info can remain
                                        oled.set_lines({0: lines[0], 1: lines[1]})
                                    except Exception:
                                        pass
                                else:
                                    print('IR:', lines)
                                # poll rate
                                time.sleep(0.15)
                        finally:
                            # clear display when stopping
                            if oled is not None:
                                try:
                                    oled.clear()
                                except Exception:
                                    pass

                    ir_thread = threading.Thread(target=ir_loop, daemon=True)
                    ir_thread.start()
                    ir_active = True
                    print('IR sensor readout started (press i to stop)')
                else:
                    # stop thread
                    ir_stop.set()
                    if ir_thread is not None and ir_thread.is_alive():
                        ir_thread.join(timeout=0.5)
                    ir_active = False
                    print('IR sensor readout stopped')
            elif ch == 'o':
                # toggle ultrasonic sensor live readout
                if not sonic_active:
                    sonic_stop.clear()

                    def sonic_loop():
                        try:
                            # enable ultrasonic sensor once
                            try:
                                bot.Ctrl_Ulatist_Switch(1)
                            except Exception:
                                pass
                            while not sonic_stop.is_set():
                                try:
                                    h = bot.read_data_array(0x1b, 1)
                                    l = bot.read_data_array(0x1a, 1)
                                except Exception:
                                    h, l = None, None
                                if h and l:
                                    try:
                                        dist = (int(h[0]) << 8) | int(l[0])
                                    except Exception:
                                        dist = None
                                else:
                                    dist = None
                                # display on OLED or console
                                text = f"US: {dist if dist is not None else 'N/A'} mm"
                                if oled is not None:
                                    try:
                                        # show ultrasonic on the last available line so it
                                        # doesn't overwrite IR lines
                                        ln = max(0, oled.num_lines - 1)
                                        oled.set_line(ln, text)
                                    except Exception:
                                        pass
                                else:
                                    print(text)
                                time.sleep(0.25)
                        finally:
                            try:
                                bot.Ctrl_Ulatist_Switch(0)
                            except Exception:
                                pass
                            if oled is not None:
                                try:
                                    oled.clear()
                                except Exception:
                                    pass

                    sonic_thread = threading.Thread(target=sonic_loop, daemon=True)
                    sonic_thread.start()
                    sonic_active = True
                    print('Ultrasonic readout started (press o to stop)')
                else:
                    sonic_stop.set()
                    if sonic_thread is not None and sonic_thread.is_alive():
                        sonic_thread.join(timeout=0.5)
                    sonic_active = False
                    print('Ultrasonic readout stopped')
            elif ch == 'f':
                # snapshot current frame to snapshots/ with timestamp
                if not camera_show:
                    print('Enable camera preview (press v) to snapshot frames.')
                else:
                    try:
                        # prefer get_frame() which returns the latest background frame
                        ret, frame = camera.get_frame()
                    except Exception:
                        # fallback to a direct read
                        ret, frame = camera.read(wait=True)
                    if ret and frame is not None:
                        try:
                            os.makedirs('snapshots', exist_ok=True)
                            name = datetime.now().strftime('snap_%Y%m%d_%H%M%S.jpg')
                            path = os.path.join('snapshots', name)
                            cv2.imwrite(path, frame)
                            print(f'Snapshot saved: {path}')
                        except Exception as e:
                            print('Failed to save snapshot:', e)
                    else:
                        print('No frame available for snapshot')
            elif ch == 'r':
                # start/stop recording to a file. Only allowed when camera
                # preview is active to ensure the underlying capture is open.
                if not camera_show:
                    print('Enable camera preview (press v) before recording (r).')
                else:
                    if not recording:
                        # request recording; the background camera thread
                        # will lazily create the VideoWriter when it receives
                        # the first frame so we choose a filename here.
                        recording = True
                        recording_filename = time.strftime('recordings/rec_%Y%m%d_%H%M%S.avi')
                        print(f'Starting recording -> {recording_filename}')
                    else:
                        recording = False
                        print('Stop recording (finalizing file)')
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
            # stop IR thread if running
            try:
                if ir_thread is not None and ir_thread.is_alive():
                    ir_stop.set()
                    ir_thread.join(timeout=0.5)
            except Exception:
                pass
            # stop ultrasonic thread if running
            try:
                if sonic_thread is not None and sonic_thread.is_alive():
                    sonic_stop.set()
                    sonic_thread.join(timeout=0.5)
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
            # ensure a recording writer is released if it exists
            try:
                if recording_writer is not None:
                    try:
                        recording_writer.release()
                    except Exception:
                        pass
                    recording_writer = None
            except Exception:
                pass
            # stop camera background thread if running
            try:
                if camera_thread is not None and camera_thread.is_alive():
                    # request thread to stop and wait briefly
                    camera_stop.set()
                    camera_thread.join(timeout=1.0)
            except Exception:
                pass
            # ensure the capture is released now that the thread has stopped
            try:
                camera.close()
            except Exception:
                pass
        except Exception:
            pass
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print('\nManual RC exited.')


if __name__ == '__main__':
    manual_control()
