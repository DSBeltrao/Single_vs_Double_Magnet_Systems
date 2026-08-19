#include <Arduino_CAN.h>
#include <math.h>

#define CAN_STBY_PIN 7

static uint32_t const ID_CMD_BROADCAST = 0x100;

struct CmdRow {
  uint32_t k;
  float t_s;
  float theta1_deg;
  float theta2_deg;
};

const int MAX_ROWS = 2500;
CmdRow rows[MAX_ROWS];
int rowCount = 0;

enum State {
  WAITING_FOR_START,
  RECEIVING_ROWS,
  READY_TO_RUN,
  RUNNING
};

State state = WAITING_FOR_START;

unsigned long runStartMicros = 0;
int runIndex = 0;

float wrap360(float deg) {
  while (deg >= 360.0f) deg -= 360.0f;
  while (deg < 0.0f) deg += 360.0f;
  return deg;
}

bool parseCSVLine(const String& line, CmdRow& out) {
  int p1 = line.indexOf(',');
  int p2 = line.indexOf(',', p1 + 1);
  int p3 = line.indexOf(',', p2 + 1);

  if (p1 < 0 || p2 < 0 || p3 < 0) return false;

  String s0 = line.substring(0, p1);
  String s1 = line.substring(p1 + 1, p2);
  String s2 = line.substring(p2 + 1, p3);
  String s3 = line.substring(p3 + 1);

  out.k          = (uint32_t)s0.toInt();
  out.t_s        = s1.toFloat();
  out.theta1_deg = wrap360(s2.toFloat());
  out.theta2_deg = wrap360(s3.toFloat());
  return true;
}

uint16_t degToUint16x10(float deg) {
  deg = wrap360(deg);
  return (uint16_t)lround(deg * 10.0f);
}

void sendBroadcastAngles(uint16_t k, float theta1_deg, float theta2_deg) {
  uint16_t a1 = degToUint16x10(theta1_deg);
  uint16_t a2 = degToUint16x10(theta2_deg);

  uint8_t data[6];
  data[0] = (k >> 8) & 0xFF;
  data[1] = k & 0xFF;
  data[2] = (a1 >> 8) & 0xFF;
  data[3] = a1 & 0xFF;
  data[4] = (a2 >> 8) & 0xFF;
  data[5] = a2 & 0xFF;

  CanMsg msg(CanStandardId(ID_CMD_BROADCAST), 6, data);
  int rc = CAN.write(msg);

  Serial.print("BCAST,k=");
  Serial.print(k);
  Serial.print(",a1=");
  Serial.print(theta1_deg, 4);
  Serial.print(",a2=");
  Serial.print(theta2_deg, 4);
  Serial.print(",rc=");
  Serial.println(rc);
}

void sendReturnToZero() {
  uint8_t data[6];
  data[0] = 0xFF;
  data[1] = 0xFF;
  data[2] = 0;
  data[3] = 0;
  data[4] = 0;
  data[5] = 0;

  CanMsg msg(CanStandardId(ID_CMD_BROADCAST), 6, data);
  int rc = CAN.write(msg);

  Serial.print("RETURN_TO_ZERO_SENT,rc=");
  Serial.println(rc);
}

void resetRows() {
  rowCount = 0;
  runIndex = 0;
  state = WAITING_FOR_START;
}

void startRun() {
  runStartMicros = micros();
  runIndex = 0;
  state = RUNNING;
  Serial.println("RUN_STARTED");
}

void handleRun() {
  if (runIndex >= rowCount) {
    Serial.println("RUN_DONE");
    sendReturnToZero();
    state = READY_TO_RUN;
    return;
  }

  unsigned long elapsed = micros() - runStartMicros;
  unsigned long target = (unsigned long)(rows[runIndex].t_s * 1000000.0f);

  if (elapsed >= target) {
    float elapsed_s = elapsed / 1000000.0f;
    float late_ms = (elapsed_s - rows[runIndex].t_s) * 1000.0f;

    Serial.print("ROW_EXEC,k=");
    Serial.print(rows[runIndex].k);
    Serial.print(",t_cmd=");
    Serial.print(rows[runIndex].t_s, 6);
    Serial.print(",t_act=");
    Serial.print(elapsed_s, 6);
    Serial.print(",late_ms=");
    Serial.print(late_ms, 3);
    Serial.print(",a1=");
    Serial.print(rows[runIndex].theta1_deg, 4);
    Serial.print(",a2=");
    Serial.println(rows[runIndex].theta2_deg, 4);

    sendBroadcastAngles((uint16_t)rows[runIndex].k,
                        rows[runIndex].theta1_deg,
                        rows[runIndex].theta2_deg);

    runIndex++;
  }
}

void handleLine(String line) {
  line.trim();
  if (line.length() == 0) return;

  if (line == "PING") {
    Serial.println("PONG");
    return;
  }

  if (line == "START") {
    resetRows();
    state = RECEIVING_ROWS;
    Serial.println("READY_FOR_ROWS");
    return;
  }

  if (line == "END") {
    if (state == RECEIVING_ROWS) {
      state = READY_TO_RUN;
      Serial.print("UPLOAD_DONE,rows=");
      Serial.println(rowCount);
    } else {
      Serial.println("ERR_BAD_STATE_END");
    }
    return;
  }

  if (line == "RUN") {
    if (state == READY_TO_RUN && rowCount > 0) {
      startRun();
    } else {
      Serial.println("ERR_NOT_READY_TO_RUN");
    }
    return;
  }

  if (state == RECEIVING_ROWS) {
    if (rowCount >= MAX_ROWS) {
      Serial.println("ERR_FULL");
      return;
    }

    CmdRow row;
    if (!parseCSVLine(line, row)) {
      Serial.print("ERR_PARSE,");
      Serial.println(line);
      return;
    }

    rows[rowCount++] = row;

    Serial.print("ROW_OK,");
    Serial.print(row.k);
    Serial.print(",");
    Serial.print(row.t_s, 6);
    Serial.print(",");
    Serial.print(row.theta1_deg, 4);
    Serial.print(",");
    Serial.println(row.theta2_deg, 4);
    return;
  }

  Serial.print("ERR_UNKNOWN,");
  Serial.println(line);
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  pinMode(CAN_STBY_PIN, OUTPUT);
  digitalWrite(CAN_STBY_PIN, LOW);

  if (!CAN.begin(CanBitRate::BR_500k)) {
    Serial.println("CAN.begin failed");
    while (1) {}
  }

  Serial.println("MASTER_READY");
  Serial.println("CSV_FORMAT,k,t_s,theta1_deg,theta2_deg");

  resetRows();
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    handleLine(line);
  }

  if (state == RUNNING) {
    handleRun();
  }
}