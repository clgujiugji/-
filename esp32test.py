import serial
import time

# 替换成你的串行端口名称，比如"COM3" 或 "/dev/ttyUSB0"
port = '/dev/cu.usbmodem101'
baudrate = 115200

ser = serial.Serial(port, baudrate, timeout=1)

try:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            print(line)
            if line == "OK":
                print("按钮被按下了")
except KeyboardInterrupt:
    print("程序被中断")

ser.close()
