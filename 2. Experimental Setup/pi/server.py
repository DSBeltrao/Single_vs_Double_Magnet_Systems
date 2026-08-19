import csv
import json
import socket
import time
import traceback
from pathlib import Path

import serial

from plan_open_loop_pi import plan_from_request

HOST = "0.0.0.0"
PORT = 5000
RECV_CHUNK = 4096

SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUD = 115200


def recv_all_json(conn) -> dict:
    data = b""
    while True:
        packet = conn.recv(RECV_CHUNK)
        if not packet:
            break
        data += packet

    if not data:
        raise ValueError("No data received.")

    return json.loads(data.decode("utf-8"))


def send_line(ser: serial.Serial, s: str) -> None:
    ser.write((s + "\n").encode("utf-8"))
    ser.flush()


def wait_for_line(
    ser: serial.Serial,
    expected_substring: str,
    timeout: float = 10.0,
    log_prefix: str = "ARDUINO:",
) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            print(log_prefix, line)
            if expected_substring in line:
                return line
    raise RuntimeError(
        f"Timed out waiting for '{expected_substring}' from master Arduino."
    )


def drain_startup(ser: serial.Serial, seconds: float = 2.0) -> list[str]:
    lines = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        if ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                print("ARDUINO:", line)
                lines.append(line)
        else:
            time.sleep(0.02)
    return lines


def run_two_motor_csv(
    csv_path: str | Path,
    serial_port: str = SERIAL_PORT,
    baud: int = SERIAL_BAUD,
) -> dict:
    csv_path = Path(csv_path).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"Motor CSV not found: {csv_path}")

    row_count = 0
    upload_echoes = []

    with serial.Serial(serial_port, baud, timeout=1) as ser:
        time.sleep(2.0)  # allow Arduino reset on serial open
        startup_lines = drain_startup(ser, seconds=2.0)

        # Handshake
        send_line(ser, "PING")
        pong_line = wait_for_line(ser, "PONG", timeout=5.0)

        send_line(ser, "START")
        ready_line = wait_for_line(ser, "READY_FOR_ROWS", timeout=5.0)

        # Upload rows
        with csv_path.open("r", newline="") as f:
            reader = csv.reader(f)

            for row in reader:
                if not row:
                    continue

                first = row[0].strip().lower()
                if first in ("k", "#", "index"):
                    continue

                if len(row) < 4:
                    print("Skipping malformed row:", row)
                    continue

                k = row[0].strip()
                t_s = row[1].strip()
                theta1 = row[2].strip()
                theta2 = row[3].strip()

                line = f"{k},{t_s},{theta1},{theta2}"
                send_line(ser, line)
                row_count += 1

                reply = ser.readline().decode("utf-8", errors="ignore").strip()
                if reply:
                    print("ARDUINO:", reply)
                    upload_echoes.append(reply)

        send_line(ser, "END")
        upload_done_line = wait_for_line(ser, "UPLOAD_DONE", timeout=10.0)

        send_line(ser, "RUN")
        print("RUN command sent")

        run_log = []
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                print("ARDUINO:", line)
                run_log.append(line)
                if "RUN_DONE" in line:
                    break

    return {
        "serial_port": serial_port,
        "baud": baud,
        "csv_path": str(csv_path),
        "rows_uploaded": row_count,
        "startup_lines": startup_lines,
        "pong_line": pong_line,
        "ready_line": ready_line,
        "upload_done_line": upload_done_line,
        "upload_echo_count": len(upload_echoes),
        "run_log_tail": run_log[-20:],
    }


def main() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")

        while True:
            conn, addr = s.accept()
            with conn:
                print(f"\nConnected by {addr}")

                try:
                    request = recv_all_json(conn)
                    print("Request received.")
                    print(json.dumps(request, indent=2))

                    # 1) Plan path and write CSV outputs
                    result = plan_from_request(request)

                    motor_csv = Path(result["motor_csv"]).resolve()
                    print(f"Planning completed. Motor CSV: {motor_csv}")

                    # 2) Upload CSV to master and run
                    exec_result = run_two_motor_csv(motor_csv)

                    # 3) Reply only after the run completes
                    reply = {
                        "status": "ok",
                        "message": "Planning, upload, and motor execution completed.",
                        "num_commands": result["num_commands"],
                        "num_motor_rows": result["num_motor_rows"],
                        "full_csv": str(Path(result["full_csv"]).resolve()),
                        "motor_csv": str(motor_csv),
                        "motor_json": str(Path(result["motor_json"]).resolve()),
                        "first_angles_deg": result["first_angles_deg"],
                        "last_angles_deg": result["last_angles_deg"],
                        "execution": exec_result,
                    }

                except Exception as e:
                    traceback.print_exc()
                    reply = {
                        "status": "error",
                        "message": str(e),
                    }

                conn.sendall(json.dumps(reply).encode("utf-8"))
                print("Reply sent.")


if __name__ == "__main__":
    main()
