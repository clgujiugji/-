import serial
import subprocess

# 替换成你的串行端口名称
port = '/dev/cu.usbmodem101'  # Windows示例, 在Linux/Mac上可能是像/dev/ttyUSB0
baudrate = 115200

ser = serial.Serial(port, baudrate, timeout=1)

try:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            print(line)
            if line == "OK":
                print("收到OK，运行脚本...")
                # 替换成你想运行的脚本的路径
                subprocess.run(["python", "/Users/qizhihan/Downloads/転倒/messagingtest2.py"])
except KeyboardInterrupt:
    print("程序被中断")

ser.close()
