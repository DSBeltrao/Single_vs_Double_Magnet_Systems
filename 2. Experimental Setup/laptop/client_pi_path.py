import json
import socket

PI_IP = "192.168.1.2"
PORT = 5000


def main() -> None:
    message = {
        "waypoints_cm": [[7.5, 7.5],
             [5, 5]],

        "s_start_cm": 0.0,
        "phi_start_deg": 0.0,

        "command_rate_hz": 10.0,
        "rotation_freq_hz": 0.5,
        "pitch_cm_per_rot": 0.124,
        "max_time_s": 500.0,

        "pool_size_cm": 15.0,
        "output_dir": ".",
        "base_name": "latest",

        "write_full_commands": True,
        "write_motor_only": True,

        "align_steps_per_corner": 0
    }

    payload = json.dumps(message).encode("utf-8")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"Connecting to {PI_IP}:{PORT} ...")
        s.connect((PI_IP, PORT))
        print("Connected.")
        s.sendall(payload)
        s.shutdown(socket.SHUT_WR)
        print("Request sent.")

        data = b""
        while True:
            packet = s.recv(4096)
            if not packet:
                break
            data += packet

    reply = json.loads(data.decode("utf-8"))
    print("\nReply from Pi:")
    print(json.dumps(reply, indent=2))


if __name__ == "__main__":
    main()