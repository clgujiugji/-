import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Google Drive API 的范围
SCOPES = ['https://www.googleapis.com/auth/drive']

def google_drive_service():
    creds = None
    # token.pickle 存储了用户的访问和刷新令牌
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # 如果没有有效的凭据，则让用户登录
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # 保存凭据以备后续使用
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('drive', 'v3', credentials=creds)



def upload_to_google_drive(service, file_path):
    file_metadata = {'name': os.path.basename(file_path)}
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    
    # 生成共享链接
    service.permissions().create(fileId=file.get('id'), body={'type': 'anyone', 'role': 'reader'}).execute()
    return f"https://drive.google.com/uc?id={file.get('id')}"

# 创建 Google Drive 服务

service = google_drive_service()

# 假设 fall_data 文件夹在当前目录下
folder_path = 'fall_data'
file_name = 'test.png'  # 替换为你的图片文件名

# 完整的图片路径
image_path = os.path.join(folder_path, file_name)

# 上传图片并获取 URL
image_url = upload_to_google_drive(service, image_path)

# 如果成功获取到 URL，则发送图片到 LINE
#if image_url:
    #send_image_message(image_url)
#else:
    #print("图片上传失败，请检查图片路径和网络连接。")
print(image_url)
