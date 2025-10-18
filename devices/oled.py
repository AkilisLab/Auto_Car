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

        self.clear()
        self.show_text("OLED Ready")

    # ------------------------------
    # General helpers
    # ------------------------------
    def clear(self):
        """Clear the display."""
        self.display.clear()
        # Some Adafruit drivers use .display() to show; YB uses image()/display()
        try:
            self.display.display()
        except Exception:
            pass

    def show_text(self, text, line=0):
        """Display text on a specific line."""
        self.draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
        y = line * 10
        self.draw.text((0, y), text, font=self.font, fill=255)
        try:
            self.display.image(self.image)
            self.display.display()
        except Exception:
            # some drivers have different methods
            try:
                self.display.image(self.image)
            except Exception:
                pass

    def show_multiline(self, lines):
        """Display multiple lines (list of strings)."""
        self.draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
        for i, text in enumerate(lines):
            y = i * 10
            self.draw.text((0, y), text, font=self.font, fill=255)
        try:
            self.display.image(self.image)
            self.display.display()
        except Exception:
            pass

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
