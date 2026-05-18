"""
AFSIM File Sender
Monitors JSON files and sends changes to Aeronet via TCP.
Usage: python sender.py [aeronet_ip] [port]
"""
import socket
import json
import time
import os
import sys
import struct

AERONET_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.100"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9600
CHECK_INTERVAL = 1.0

WATCH_FILES = [
    "Performance_Show/platform_info.json",
    "Performance_Show/simulation_stats.json",
]

def get_file_content(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return None

def send_file(sock, filename, content):
    fname_bytes = filename.encode('utf-8')
    content_bytes = content.encode('utf-8')
    header = struct.pack('>I', len(fname_bytes))
    header += fname_bytes
    header += struct.pack('>I', len(content_bytes))
    header += content_bytes
    sock.sendall(header)

def main():
    print("=" * 50)
    print("  AFSIM File Sender")
    print("=" * 50)
    print(f"  Target: {AERONET_IP}:{PORT}")
    print(f"  Interval: {CHECK_INTERVAL}s")
    print(f"  Watch: {', '.join(WATCH_FILES)}")
    print("=" * 50)

    last_contents = {}

    while True:
        changed_files = {}
        for filepath in WATCH_FILES:
            content = get_file_content(filepath)
            if content is None:
                continue
            if filepath not in last_contents or last_contents[filepath] != content:
                fname = os.path.basename(filepath)
                changed_files[fname] = content
                last_contents[filepath] = content

        if changed_files:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                sock.connect((AERONET_IP, PORT))
                for fname, content in changed_files.items():
                    send_file(sock, fname, content)
                    t = time.strftime("%H:%M:%S")
                    print(f"  [{t}] Sent: {fname} ({len(content)} bytes)")
                sock.close()
            except ConnectionRefusedError:
                t = time.strftime("%H:%M:%S")
                print(f"  [{t}] Connection refused - is receiver running? ({AERONET_IP}:{PORT})")
            except Exception as e:
                t = time.strftime("%H:%M:%S")
                print(f"  [{t}] Send error: {e}")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
