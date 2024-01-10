import serial
import subprocess
import threading
from flask import Flask, request, render_template_string

app = Flask(__name__)

# 替换成你的串行端口名称
port = '/dev/cu.usbmodem101'  # Windows示例, 在Linux/Mac上可能是像/dev/ttyUSB0
baudrate = 115200

ser = serial.Serial(port, baudrate, timeout=1)

def connect_to_wifi(ssid, password):
    config = f"""<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
    <name>{ssid}</name>
    <SSIDConfig>
        <SSID>
            <name>{ssid}</name>
        </SSID>
    </SSIDConfig>
    <connectionType>ESS</connectionType>
    <connectionMode>auto</connectionMode>
    <MSM>
        <security>
            <authEncryption>
                <authentication>WPA2PSK</authentication>
                <encryption>AES</encryption>
                <useOneX>false</useOneX>
            </authEncryption>
            <sharedKey>
                <keyType>passPhrase</keyType>
                <protected>false</protected>
                <keyMaterial>{password}</keyMaterial>
            </sharedKey>
        </security>
    </MSM>
</WLANProfile>"""
    profile_path = f"{ssid}.xml"
    with open(profile_path, "w") as file:
        file.write(config)
    subprocess.run(["netsh", "wlan", "add", "profile", f"filename={profile_path}"])

    # 连接到无线网络
    subprocess.run(["netsh", "wlan", "connect", ssid])

def listen_for_esp32():
    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').rstrip()
                print(line)
                if line == "OK":
                    print("收到OK，运行脚本...")
                    subprocess.run(["python", "/path/to/your/script.py"])
    except KeyboardInterrupt:
        print("串行监听被中断")
    finally:
        ser.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ssid = request.form['ssid']
        password = request.form['password']
        connect_to_wifi(ssid, password)
        return 'Attempting to connect to WiFi...'
    return '''
        <form method="post">
            SSID: <input type="text" name="ssid"><br>
            Password: <input type="password" name="password"><br>
            <input type="submit" value="Submit">
        </form>
    '''

if __name__ == '__main__':
    threading.Thread(target=listen_for_esp32, daemon=True).start()
    app.run(host='0.0.0.0', port=80)
