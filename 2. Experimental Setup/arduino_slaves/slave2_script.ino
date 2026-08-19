#include <Arduino_CAN.h>
#include <AccelStepper.h>
#include <math.h>

#define SLAVE_INDEX 2   // 1 for slave 1, 2 for slave 2

#define DIR_PIN 2
#define STEP_PIN 3
#define ENABLE_PIN 4
#define CAN_STBY_PIN 7

AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

static uint32_t const ID_CMD_BROADCAST = 0x100;

const long STEPS_PER_REV = 1600;   // 1/8 microstep
const int DIR_SIGN = -1;

float MAX_SPEED_STEPS = 8000.0f;
float ACCEL_STEPS     = 15000.0f;

long degToSteps(float deg) {
  return lround((deg / 360.0f) * STEPS_PER_REV) * DIR_SIGN;
}

float stepsToDegUnwrapped(long steps) {
  return (360.0f * steps / STEPS_PER_REV) * DIR_SIGN;
}

float wrap360(float deg) {
  while (deg >= 360.0f) deg -= 360.0f;
  while (deg < 0.0f) deg += 360.0f;
  return deg;
}

float wrap180(float deg) {
  while (deg >= 180.0f) deg -= 360.0f;
  while (deg < -180.0f) deg += 360.0f;
  return deg;
}

float currentClockDeg() {
  return wrap360(stepsToDegUnwrapped(stepper.currentPosition()));
}

float uint16x10ToDeg(uint16_t v) {
  return ((float)v) / 10.0f;
}

void commandClockAngle(float cmdClockDeg) {
  float currClockDeg = currentClockDeg();
  float deltaDeg = wrap180(cmdClockDeg - currClockDeg);
  long deltaSteps = degToSteps(deltaDeg);
  long targetSteps = stepper.currentPosition() + deltaSteps;

  stepper.moveTo(targetSteps);

  Serial.print("EXEC_CMD,cmd=");
  Serial.print(cmdClockDeg, 4);
  Serial.print(",curr=");
  Serial.print(currClockDeg, 4);
  Serial.print(",delta=");
  Serial.print(deltaDeg, 4);
  Serial.print(",target_clock=");
  Serial.print(wrap360(currClockDeg + deltaDeg), 4);
  Serial.print(",target_steps=");
  Serial.println(targetSteps);
}

void handleLocalSerial(String line) {
  line.trim();
  if (line.length() == 0) return;

  if (line == "PING") {
    Serial.println("PONG");
    return;
  }

  if (line == "ZERO") {
    stepper.setCurrentPosition(0);
    Serial.println("ZERO_OK");
    return;
  }

  if (line == "POS?") {
    Serial.print("POS_CLOCK_DEG,");
    Serial.println(currentClockDeg(), 4);
    return;
  }

  if (line.startsWith("SPEED,")) {
    float v = line.substring(6).toFloat();
    if (v > 0.0f) {
      MAX_SPEED_STEPS = v;
      stepper.setMaxSpeed(MAX_SPEED_STEPS);
      Serial.print("SPEED_OK,");
      Serial.println(MAX_SPEED_STEPS, 2);
    } else {
      Serial.println("ERR_BAD_SPEED");
    }
    return;
  }

  if (line.startsWith("ACCEL,")) {
    float a = line.substring(6).toFloat();
    if (a > 0.0f) {
      ACCEL_STEPS = a;
      stepper.setAcceleration(ACCEL_STEPS);
      Serial.print("ACCEL_OK,");
      Serial.println(ACCEL_STEPS, 2);
    } else {
      Serial.println("ERR_BAD_ACCEL");
    }
    return;
  }

  Serial.print("ERR_UNKNOWN,");
  Serial.println(line);
}

void setup() {
  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, LOW);

  Serial.begin(115200);
  //while (!Serial) {}

  pinMode(CAN_STBY_PIN, OUTPUT);
  digitalWrite(CAN_STBY_PIN, LOW);

  if (!CAN.begin(CanBitRate::BR_500k)) {
    //Serial.println("CAN.begin failed");
    while (1) {}
  }

  stepper.setMaxSpeed(MAX_SPEED_STEPS);
  stepper.setAcceleration(ACCEL_STEPS);
  stepper.setCurrentPosition(0);

  Serial.print("SLAVE_READY,index=");
  Serial.println(SLAVE_INDEX);
  Serial.println("ZERO_IS_PLUS_Z");
  Serial.print("STEPS_PER_REV,");
  Serial.println(STEPS_PER_REV);
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    handleLocalSerial(line);
  }

  if (CAN.available()) {
    CanMsg const msg = CAN.read();

    if (msg.id == CanStandardId(ID_CMD_BROADCAST) && msg.data_length >= 6) {
      uint16_t k  = (uint16_t(msg.data[0]) << 8) | msg.data[1];

      if (k == 0xFFFF) {
        Serial.println("RETURN_TO_ZERO");
        commandClockAngle(0.0f);
      } else {
        uint16_t a1 = (uint16_t(msg.data[2]) << 8) | msg.data[3];
        uint16_t a2 = (uint16_t(msg.data[4]) << 8) | msg.data[5];

        float myDeg = (SLAVE_INDEX == 1) ? uint16x10ToDeg(a1)
                                         : uint16x10ToDeg(a2);

        Serial.print("RX_CMD,k=");
        Serial.print(k);
        Serial.print(",my_deg=");
        Serial.println(myDeg, 4);

        commandClockAngle(myDeg);
      }
    }
  }

  stepper.run();
}