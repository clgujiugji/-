

#line_bot_api =Configuration('LYVWpWrl+P3WJmw3T1NbHcIxpQA57Wv9HXtfCOHufWJwIlH7EJxuq/SwrDRH6SoHOOtu4ReH5TpMA1epERU9znwv3WQz9KC3/TnAAyhsFOVMEI1g7KQB/M/SxQijOVWyBSYu10PVJoUn8ow7t6QKPwdB04t89/1O/w1cDnyilFU=')
from linebot import LineBotApi
from linebot.models import TextSendMessage

line_bot_api = LineBotApi('LYVWpWrl+P3WJmw3T1NbHcIxpQA57Wv9HXtfCOHufWJwIlH7EJxuq/SwrDRH6SoHOOtu4ReH5TpMA1epERU9znwv3WQz9KC3/TnAAyhsFOVMEI1g7KQB/M/SxQijOVWyBSYu10PVJoUn8ow7t6QKPwdB04t89/1O/w1cDnyilFU=')

def send_line_notification(message):
    line_bot_api.broadcast(TextSendMessage(text=message))

# 在检测到摔倒时调用
send_line_notification("检测到摔倒，请立即检查！")


