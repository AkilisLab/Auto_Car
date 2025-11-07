#!/usr/bin/env python3
"""OLED display wrapper using Adafruit_SSD1306 and PIL (compatible with YB_oled imports).

This version avoids `board`/`busio` and uses the Adafruit_SSD1306 classes directly
so it only depends on the same libraries as `YB_oled.py`.
"""

import time
from PIL import Image, ImageDraw, ImageFont
import Adafruit_SSD1306 as SSD


class OLEDDisplay:
    def __init__(self, width=128, height=32, i2c_bus=1):
        # Instantiate the SSD1306 driver (128x32 or 128x64)
        if height == 32:
            self.display = SSD.SSD1306_128_32(rst=None, i2c_bus=i2c_bus, gpio=1)
        else:
            # fall back to 128x64 if requested
            self.display = SSD.SSD1306_128_64(rst=None, i2c_bus=i2c_bus, gpio=1)

        self.display.begin()
        self.width = width
        self.height = height

        # Create blank image for drawing
        self.image = Image.new("1", (self.width, self.height))
        self.draw = ImageDraw.Draw(self.image)
        self.font = ImageFont.load_default()

        # Number of text lines we can show (approx, based on 10px per line)
        self.num_lines = max(1, self.height // 10)
        # in-memory buffer for lines so callers can update individual lines
        self._lines = [""] * self.num_lines

        self.clear()
        self.set_line(0, "OLED Ready", refresh=True)

    # ------------------------------
    # General helpers
    # ------------------------------
    def clear(self):
        """Clear the display."""
        # clear both the device and our buffer
        self._lines = [""] * self.num_lines
        try:
            self.display.clear()
        except Exception:
            pass
        try:
            # ensure display shows cleared buffer
            self.display.image(Image.new("1", (self.width, self.height)))
            self.display.display()
        except Exception:
            pass

    def refresh_display(self):
        """Render current buffer to the device."""
        # draw buffer into image
        self.draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
        for i, text in enumerate(self._lines):
            y = i * 10
            try:
                self.draw.text((0, y), text, font=self.font, fill=255)
            except Exception:
                pass
        try:
            self.display.image(self.image)
            self.display.display()
        except Exception:
            # some drivers may have different method names; ignore
            pass

    def set_line(self, line: int, text: str, refresh: bool = True):
        """Set a specific text line in the buffer and optionally refresh.

        Lines are 0-indexed. If line >= num_lines it's clamped.
        """
        if line < 0:
            return
        idx = min(line, self.num_lines - 1)
        self._lines[idx] = str(text)
        if refresh:
            self.refresh_display()

    def set_lines(self, mapping: dict, refresh: bool = True):
        """Set multiple lines at once. mapping: {line_index: text}.
        Useful to update several fields together without flicker.
        """
        for k, v in mapping.items():
            if isinstance(k, int) and k >= 0:
                idx = min(k, self.num_lines - 1)
                self._lines[idx] = str(v)
        if refresh:
            self.refresh_display()

    def clear_line(self, line: int, refresh: bool = True):
        if line < 0:
            return
        idx = min(line, self.num_lines - 1)
        self._lines[idx] = ""
        if refresh:
            self.refresh_display()

    def show_text(self, text, line=0):
        """Backward-compatible: set a single line (overwrites that line only)."""
        self.set_line(line, text, refresh=True)

    def show_multiline(self, lines):
        """Display multiple lines starting at line 0. This updates the
        internal buffer so other independent updates can coexist."""
        for i, text in enumerate(lines):
            if i >= self.num_lines:
                break
            self._lines[i] = str(text)
        self.refresh_display()

    # ------------------------------
    # System / Status Info
    # ------------------------------
    def show_system_status(self, mode, uart, wifi, speed):
        lines = [
            f"MODE:{mode}",
            f"UART:{uart} WIFI:{wifi}",
            f"SPEED:{speed:.2f} m/s",
        ]
        self.show_multiline(lines)

    # ------------------------------
    # Motion / Control Data
    # ------------------------------
    def show_motion_data(self, speed, steer, pwm, pid):
        lines = [
            f"SPD:{speed:.2f}  STR:{steer:+.1f}",
            f"PWM:{pwm}",
            f"PID:P={pid[0]:.1f} I={pid[1]:.1f} D={pid[2]:.1f}",
        ]
        self.show_multiline(lines)

    # ------------------------------
    # Perception Feedback
    # ------------------------------
    def show_perception(self, lane, obj, dist):
        lines = [
            f"LANE:{lane}",
            f"OBJ:{obj}",
            f"DIST:{dist:.2f} m",
        ]
        self.show_multiline(lines)

    # ------------------------------
    # Voice / Command Status
    # ------------------------------
    def show_voice_status(self, phrase, command):
        lines = [
            f"VOICE:'{phrase}'",
            f"CMD:{command}",
        ]
        self.show_multiline(lines)

    # ------------------------------
    # Network Info
    # ------------------------------
    def show_network_info(self, ctrl, ip, wifi):
        lines = [
            f"CTRL:{ctrl}",
            f"IP:{ip}",
            f"WIFI:{wifi}",
        ]
        self.show_multiline(lines)

    # ------------------------------
    # Debug / Diagnostic
    # ------------------------------
    def show_debug_info(self, fw, errors, temp):
        lines = [
            f"FW:{fw}",
            f"I2C ERR:{errors}",
            f"TEMP:{temp}C",
        ]
        self.show_multiline(lines)

    # ------------------------------
    # Alerts / Custom Messages
    # ------------------------------
    def show_alert(self, msg):
        self.show_text(f"ALERT: {msg}")


if __name__ == "__main__":
    oled = OLEDDisplay()
    oled.show_system_status("AUTO", "OK", "Connected", 0.52)
    time.sleep(2)
    oled.show_motion_data(0.61, 15.0, 180, (1.2, 0.3, 0.7))
    time.sleep(2)
    oled.show_perception("CENTER", "HUMAN", 0.48)
    time.sleep(2)
    oled.show_voice_status("Go Forward", "DRIVE")
    time.sleep(2)
    oled.show_network_info("WEB", "192.168.0.25", "OK")
    time.sleep(2)
    oled.show_debug_info("v1.2.3", 0, 36)
    time.sleep(2)
    oled.show_alert("Obstacle Detected!")
    time.sleep(3)
    oled.clear()
