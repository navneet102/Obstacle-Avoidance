#include <ESP8266WiFi.h>

const char* ssid = "iQOO Z7s 5G";  // Replace with your hotspot SSID
const char* password = "a1b2c3d4e5f6g7";  // Replace with your hotspot password

int motorLeftPin1 = 12;
int motorLeftPin2 = 14;
int enableLeftPin = 13;
int motorRightPin1 = 2;
int motorRightPin2 = 4;
int enableRightPin = 0;

int trigPin = 16;
int echoPin = 5;

#define SOUND_SPEED 0.034
#define HALF_SPEED 115  // PWM value for approximately half speed (0-255 range)

float thresholdDistance = 31.0;

long duration;
float distanceCm;

WiFiServer server(80);  // Create a server that listens on port 80

void setup() {
  pinMode(motorLeftPin1, OUTPUT);
  pinMode(motorLeftPin2, OUTPUT);
  pinMode(enableLeftPin, OUTPUT);
  pinMode(motorRightPin1, OUTPUT);
  pinMode(motorRightPin2, OUTPUT);
  pinMode(enableRightPin, OUTPUT);

  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  Serial.begin(115200);

  WiFi.begin(ssid, password);  // Start connecting to the Wi-Fi network

  Serial.print("Connecting to ");
  Serial.println(ssid);

  // Wait until the connection is established
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected!");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());  // Print the IP address

  server.begin();  // Start the server
  Serial.println("Server started");
}

void loop() {
  distanceCm = getDistance();
  moveForward();
  if (distanceCm < thresholdDistance) {
    stopMotors();
    delay(1000);

    WiFiClient client = server.available();  // Check if a client has connected
    if (client) {
      Serial.println("New Client Connected");
      String receivedData = "";
      while (client.connected()) {
        while (client.available()) {
          char c = client.read();
          Serial.write(c);  // Print the received character to the Serial Monitor
          receivedData += c;
        }
        if (receivedData.length() > 0) {
          // Process receivedData to get the region
          char region = receivedData.charAt(0);
          rotateToRegion(region);
          delay(500);
          stopMotors();
          delay(1000);
          distanceCm = getDistance();
          if (distanceCm >= thresholdDistance) {
            moveForward();
          } else {
            rotateOpp(region);
            stopMotors();
            delay(1000);
            distanceCm = getDistance();
            if (distanceCm >= thresholdDistance) {
              moveForward();
            } else {
              rotateRight();
              delay(500);
              stopMotors();
              delay(1000);
              moveForward();
            }
          }

          // Send a response back to the client
          client.print("OK\n");
          client.stop();
          Serial.println("Response sent to client");
        }
      }
      Serial.println("Client Disconnected");
    }
  } else {
    moveForward();
  }

  delay(100);
}

void moveForward() {
  analogWrite(enableLeftPin, HALF_SPEED);
  analogWrite(enableRightPin, HALF_SPEED);
  digitalWrite(motorLeftPin1, HIGH);
  digitalWrite(motorLeftPin2, LOW);
  digitalWrite(motorRightPin1, HIGH);
  digitalWrite(motorRightPin2, LOW);
}

void stopMotors() {
  analogWrite(enableLeftPin, 0);
  analogWrite(enableRightPin, 0);
  digitalWrite(motorLeftPin1, HIGH);
  digitalWrite(motorLeftPin2, HIGH);
  digitalWrite(motorRightPin1, HIGH);
  digitalWrite(motorRightPin2, HIGH);
}

void rotateLeft() {
  analogWrite(enableLeftPin, HALF_SPEED);
  analogWrite(enableRightPin, HALF_SPEED);
  digitalWrite(motorLeftPin1, HIGH);
  digitalWrite(motorLeftPin2, LOW);
  digitalWrite(motorRightPin1, LOW);
  digitalWrite(motorRightPin2, HIGH);
}

void rotateRight() {
  analogWrite(enableLeftPin, HALF_SPEED);
  analogWrite(enableRightPin, HALF_SPEED);
  digitalWrite(motorLeftPin1, LOW);
  digitalWrite(motorLeftPin2, HIGH);
  digitalWrite(motorRightPin1, HIGH);
  digitalWrite(motorRightPin2, LOW);
}

void rotateToRegion(char r) {
  if (r == '1') {
    rotateLeft();
    delay(800);
  } else if (r == '2') {
    rotateLeft();
    delay(400);
  } else if (r == '3') {
    rotateRight();
    delay(400);
  } else if (r == '4') {
    rotateRight();
    delay(800);
  }
}

void rotateOpp(char r) {
  if (r == '1' || r == '2') {
    rotateRight();
    delay(1000);
  } else {
    rotateLeft();
    delay(1000);
  }
}

float getDistance() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  duration = pulseIn(echoPin, HIGH);

  return duration * SOUND_SPEED / 2;
}
