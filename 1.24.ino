const int buttonPin = 1; // 按钮连接到ESP32的GPIO2

void setup() {
    Serial.begin(115200); // 初始化串行通信
    pinMode(buttonPin, INPUT_PULLUP); // 设置按钮引脚为输入并启用内部上拉电阻
}

void loop() {
    int buttonState = digitalRead(buttonPin); // 读取按钮状态

    if (buttonState == LOW) { // 按钮按下时，GPIO2为低电平
        Serial.println("OK"); // 输出"OK"到串行监视器
        delay(200); // 简单的消抖动延迟
    }
}
