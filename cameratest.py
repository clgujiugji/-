import cv2

def test_camera():
    # 创建 VideoCapture 对象，0 表示系统默认摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    while True:
        # 从摄像头读取一帧
        ret, frame = cap.read()

        # 如果正确读取帧，ret 为 True
        if not ret:
            print("无法从摄像头读取帧。")
            break

        # 显示图像
        cv2.imshow('Camera Test', frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()
