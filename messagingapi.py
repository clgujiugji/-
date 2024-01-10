import os
import requests
from linebot import LineBotApi
from linebot.models import ImageSendMessage

# 替换为你的 LINE Channel Access Token 和接收消息的用户或群组的 ID
LINE_CHANNEL_ACCESS_TOKEN = 'LYVWpWrl+P3WJmw3T1NbHcIxpQA57Wv9HXtfCOHufWJwIlH7EJxuq/SwrDRH6SoHOOtu4ReH5TpMA1epERU9znwv3WQz9KC3/TnAAyhsFOVMEI1g7KQB/M/SxQijOVWyBSYu10PVJoUn8ow7t6QKPwdB04t89/1O/w1cDnyilFU='
RECIPIENT_USER_ID = 'U73e7d9edb2ae24b83a82ca9b1069598a'#'U73e7d9edb2ae24b83a82ca9b1069598a'

def upload_to_transfer_sh(local_file_path):
    with open(local_file_path, 'rb') as f:
        response = requests.post('https://transfer.sh/', files={'file': f})
    if response.status_code == 200:
        return response.text.strip()
    return None

def send_image_message( image_url):
    line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
    image_message = ImageSendMessage(
        original_content_url=image_url,
        preview_image_url=image_url
    )
    #line_bot_api.broadcast(image_message)
    line_bot_api.push_message(image_message)

# 假设 fall_data 文件夹在当前目录下
folder_path = 'fall_data'
file_name = 'test.png'  # 替换为你的图片文件名

# 完整的图片路径
image_path = os.path.join(folder_path, file_name)

# 上传图片并获取 URL
image_url = upload_to_transfer_sh(image_path)

# 如果成功获取到 URL，则发送图片到 LINE
if image_url:
    send_image_message(image_url)
else:
    print("图片上传失败，请检查图片路径和网络连接。")
