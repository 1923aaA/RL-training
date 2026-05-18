#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDS Python Subscriber Test
"""
import pydds
import time
import sys
import threading
from datetime import datetime

# 全局统计
message_count = 0
total_bytes = 0
lock = threading.Lock()


def on_message_received(topic_id, data):
    """消息回调函数"""
    global message_count, total_bytes

    with lock:
        message_count += 1
        total_bytes += len(data)

        # 解码消息
        try:
            msg = data.decode('utf-8')
        except:
            msg = f"<binary data, {len(data)} bytes>"

        # 显示消息
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"\n{'─' * 60}")
        print(f"[MESSAGE #{message_count}]")
        print(f"  Time:     {timestamp}")
        print(f"  Topic ID: {topic_id}")
        print(f"  Length:   {len(data)} bytes")
        print(f"  Content:  \"{msg}\"")
        print(f"{'─' * 60}")
        sys.stdout.flush()


def main():
    global message_count, total_bytes

    print("=" * 60)
    print("  Python DDS Subscriber (Using DDSCOMM SDK)")
    print("=" * 60)

    # 配置
    DOMAIN_ID = 40
    TOPIC_ID = 1005
    RUN_TIME = 60

    print(f"\n[Config]")
    print(f"  Domain ID: {DOMAIN_ID}")
    print(f"  Topic ID:  {TOPIC_ID}")
    print(f"  Run Time:  {RUN_TIME} seconds")
    print("=" * 60)

    # 初始化
    print("\n[Step 1] Initializing DDS...")
    result = pydds.init(DOMAIN_ID)
    if result != 0:
        print(f"[ERROR] Failed to initialize DDS!")
        print(f"Last error: {pydds.get_last_error()}")
        return 1
    print("[SUCCESS] DDS initialized!\n")

    try:
        # 订阅主题
        print(f"[Step 2] Subscribing to Topic {TOPIC_ID}...")
        if not pydds.subscribe(TOPIC_ID, on_message_received):
            print(f"[ERROR] Failed to subscribe!")
            print(f"Last error: {pydds.get_last_error()}")
            pydds.shutdown()
            return 1
        print("[SUCCESS] Subscribed successfully!\n")

        # 等待DDS就绪
        print("[Step 3] Waiting for DDS to stabilize...")
        time.sleep(2)

        # 等待发布者
        print(f"[Step 4] Waiting for publishers (timeout: 5s)...")
        if pydds.wait_for_publishers(TOPIC_ID, 5):
            count = pydds.get_publisher_count(TOPIC_ID)
            print(f"[SUCCESS] Found {count} publisher(s)!\n")
        else:
            print("[WARNING] No publishers found, waiting anyway...\n")

        # 监听消息
        print("=" * 60)
        print("  🎧 Listening for messages...")
        print("  (Press Ctrl+C to stop)")
        print("=" * 60)

        start_time = time.time()
        last_status_time = start_time

        while True:
            time.sleep(1)
            elapsed = time.time() - start_time

            if elapsed >= RUN_TIME:
                break

            # 每5秒显示一次状态
            if elapsed - (last_status_time - start_time) >= 5:
                with lock:
                    print(f"\n[Status] Running {int(elapsed)}/{RUN_TIME}s, "
                          f"received {message_count} messages, "
                          f"{total_bytes} bytes")
                last_status_time = time.time()

    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user")

    finally:
        # 最终统计
        print("\n" + "=" * 60)
        print("  Reception Complete!")
        print("=" * 60)
        print(f"\n[Final Statistics]")
        print(f"  Total Messages: {message_count}")
        print(f"  Total Bytes:    {total_bytes}")
        if message_count > 0:
            print(f"  Average Size:   {total_bytes // message_count} bytes/message")

        # 清理
        print(f"\n[Step 5] Unsubscribing...")
        pydds.unsubscribe(TOPIC_ID)

        print(f"[Step 6] Shutting down DDS...")
        pydds.shutdown()
        print("[SUCCESS] Subscriber shutdown complete!")

    print("\n" + "=" * 60)
    print("  Subscriber Test Completed!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
