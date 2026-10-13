/*
  ESP32 / NodeMCU IoT Motor Sensor Telemetry Transmitter
  DC Series Motor Digital Twin Intelligent Monitoring

  Collects physical sensors:
  - ACS712 / Hall Effect Current Sensor -> Analog Pin A0 (GPIO 34)
  - Voltage Divider -> Analog Pin (GPIO 35)
  - MPU6050 Accelerometer (X, Y, Z Vibration) -> I2C (SDA 21, SCL 22)
  - DS18B20 / LM35 Temperature Sensors -> OneWire (GPIO 4 & 5)
  - Optical Tachometer / Hall Speed Sensor -> Interrupt (GPIO 18)

  Publishes directly over Wi-Fi to Firebase Realtime Database:
  https://predictive-maintenance-23249-default-rtdb.asia-southeast1.firebasedatabase.app/motor_telemetry/latest.json
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// ── Wi-Fi Configuration ──────────────────────────────────────
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// ── Firebase Realtime Database Endpoint ─────────────────────
const char* FIREBASE_URL  = "https://predictive-maintenance-23249-default-rtdb.asia-southeast1.firebasedatabase.app/motor_telemetry/latest.json";

// ── Sensor Calibration Offsets ──────────────────────────────
const int SAMPLE_INTERVAL_MS = 1000; // 1 reading per second
unsigned long lastSendTime = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n==============================================");
  Serial.println("  ESP32 MOTOR DIGITAL TWIN SENSOR NODE");
  Serial.println("==============================================");

  // Connect to Wi-Fi
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\n[Wi-Fi Connected!]");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  unsigned long now = millis();
  if (now - lastSendTime >= SAMPLE_INTERVAL_MS) {
    lastSendTime = now;

    if (WiFi.status() == WL_CONNECTED) {
      // 1. Read Physical Sensor Pins (Replace with your actual pin reads)
      float voltage = 224.5 + (random(-15, 15) / 10.0);
      float current = 12.3 + (random(-5, 5) / 10.0);
      float rpm     = 1510.0 + (random(-25, 25));
      float temp1   = 36.8 + (random(-5, 5) / 10.0);
      float temp2   = 32.4 + (random(-4, 4) / 10.0);
      float x       = 0.22 + (random(-2, 2) / 100.0);
      float y       = -1.08 + (random(-3, 3) / 100.0);
      float z       = 9.12 + (random(-4, 4) / 100.0);

      // 2. Build JSON Telemetry Payload
      StaticJsonDocument<256> doc;
      doc["Voltage"] = voltage;
      doc["Current"] = current;
      doc["RPM"]     = rpm;
      doc["Temp1"]   = temp1;
      doc["Temp2"]   = temp2;
      doc["X"]       = x;
      doc["Y"]       = y;
      doc["Z"]       = z;
      doc["timestamp"] = String(millis());

      String jsonPayload;
      serializeJson(doc, jsonPayload);

      // 3. Send HTTP PUT to Firebase Realtime Database
      HTTPClient http;
      http.begin(FIREBASE_URL);
      http.addHeader("Content-Type", "application/json");

      int httpResponseCode = http.PUT(jsonPayload);

      if (httpResponseCode == 200) {
        Serial.print("[Firebase SUCCESS] Telemetry pushed: ");
        Serial.println(jsonPayload);
      } else {
        Serial.print("[Firebase ERROR] HTTP Code: ");
        Serial.println(httpResponseCode);
      }

      http.end();
    } else {
      Serial.println("[Wi-Fi Disconnected] Reconnecting...");
      WiFi.reconnect();
    }
  }
}
