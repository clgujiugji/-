import subprocess
from flask import Flask, request, render_template_string

app = Flask(__name__)

def connect_to_wifi(ssid, password):
    # 创建无线网络配置文件
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

    # 添加无线网络配置
    subprocess.run(["netsh", "wlan", "add", "profile", f"filename={profile_path}"])

    # 连接到无线网络
    subprocess.run(["netsh", "wlan", "connect", ssid])

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
    app.run(host='0.0.0.0', port=80)

