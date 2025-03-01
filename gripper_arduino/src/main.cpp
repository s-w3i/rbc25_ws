#include <Arduino.h>
#include <Servo.h>

Servo gripperServo;  // Create a Servo object for the gripper

// Define servo angles for open and closed positions
const int OPEN_ANGLE = 0;    // Adjust as needed for your gripper
const int CLOSE_ANGLE = 180;  // Adjust as needed for your gripper

void setup() {
  Serial.begin(9600);              // Start serial communication at 9600 baud
  gripperServo.attach(9);          // Attach the servo to digital pin 9
  gripperServo.write(OPEN_ANGLE);  // Initialize with gripper open
  Serial.println("Gripper Controller Ready. Send '0' for open, '1' for close.");
}

void loop() {
  if (Serial.available() > 0) {
    // Read the incoming serial data until newline
    String command = Serial.readStringUntil('\n');
    command.trim();  // Remove any extra whitespace or newline characters
    
    // Process the command if it's valid
    if (command == "0") {
      gripperServo.write(OPEN_ANGLE);
      Serial.println("Gripper opened");
    } else if (command == "1") {
      gripperServo.write(CLOSE_ANGLE);
      Serial.println("Gripper closed");
    } else {
      Serial.println("Invalid command. Please send '0' for open or '1' for close.");
    }
  }
  delay(20);  // Small delay for stability
}
