#include <ESP8266WiFi.h>

const char* ssid = "YourHotspotSSID";  // Replace with your hotspot SSID
const char* password = "YourHotspotPassword";  // Replace with your hotspot password

WiFiServer server(80);  // Create a server that listens on port 80


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

  WiFi.begin(ssid, password);
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
  Serial.print("Distance (cm): ");
  Serial.println(distanceCm);
  if (distanceCm < thresholdDistance) {
    stopMotors();
    delay(1000);
    WiFiClient client = server.available();  // Check if a client has connected
    char turnDir = '\0';  
    if (client) {
      Serial.println("New Client Connected");
      while (client.connected()) {
        if (client.available()) {
          char c = client.read();
          Serial.write(c);  // Print the received character to the Serial Monitor
          turnDir = c;
        }
      }
      client.stop();  // Close the connection
      Serial.println("Client Disconnected");
    }

    if(turnDir == '1'){
      rotateLeft();
      delay(800);
    }else if(turnDir == '2'){
      rotateLeft();
      delay(500);
    }else if(turnDir == '3'){
      rotateLeft();
      delay(500);
    }else if(turnDir == '4'){
      rotateLeft();
      delay(800);
    }
    stopMotors();
    delay(1000);
    distanceCm = getDistance();
    if (distanceCm >= thresholdDistance) {
      moveForward();
    } else {
      if(turnDir == '1' || turnDir == '2'){
        rotateRight();
        delay(1000);
      }else{
        rotateLeft();
        delay(1000);
      }   
      stopMotors();
      delay(1000);
      distanceCm = getDistance();
      if (distanceCm >= thresholdDistance) {
        moveForward();
      }
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
  Serial.println("Moving Forward");
}


void stopMotors() {
  analogWrite(enableLeftPin, 0);
  analogWrite(enableRightPin, 0);
  digitalWrite(motorLeftPin2, LOW);
  digitalWrite(motorLeftPin1, LOW);
  digitalWrite(motorRightPin1, LOW);
  digitalWrite(motorRightPin2, LOW);
  Serial.println("Motors Stopped");
}


void rotateLeft() {
  analogWrite(enableLeftPin, HALF_SPEED);
  analogWrite(enableRightPin, HALF_SPEED);
  digitalWrite(motorLeftPin1, HIGH);
  digitalWrite(motorLeftPin2, LOW);
  digitalWrite(motorRightPin1, LOW);
  digitalWrite(motorRightPin2, HIGH);
  Serial.println("Rotating Left");
}


void rotateRight() {
  analogWrite(enableLeftPin, HALF_SPEED);
  analogWrite(enableRightPin, HALF_SPEED);
  digitalWrite(motorLeftPin1, LOW);
  digitalWrite(motorLeftPin2, HIGH);
  digitalWrite(motorRightPin1, HIGH);
  digitalWrite(motorRightPin2, LOW);
  Serial.println("Rotating Right");
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
