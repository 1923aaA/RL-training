#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDS Python Publisher Test
"""
import pydds
import time
import sys


def main():
    print("=" * 60)
    print("  Python DDS Publisher (Using DDSCOMM SDK)")
    print("=" * 60)

    # 配置
    DOMAIN_ID = 40
    TOPIC_ID = 1005
    MESSAGE_COUNT = 10
    INTERVAL_SEC = 1

    print(f"\n[Config]")
    print(f"  Domain ID:      {DOMAIN_ID}")
    print(f"  Topic ID:       {TOPIC_ID}")
    print(f"  Message Count:  {MESSAGE_COUNT}")
    print(f"  Interval:       {INTERVAL_SEC}s")
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
        # 等待稳定
        print("[Step 2] Waiting for DDS to stabilize...")
        time.sleep(2)

        # 等待订阅者
        print(f"[Step 3] Waiting for subscribers (timeout: 5s)...")
        if pydds.wait_for_subscribers(TOPIC_ID, 5):
            count = pydds.get_subscriber_count(TOPIC_ID)
            print(f"[SUCCESS] Found {count} subscriber(s)!\n")
        else:
            print("[WARNING] No subscribers found, continuing anyway...\n")

        # 发布消息
        print("[Step 4] Publishing messages...")
        print("-" * 60)

        success_count = 0
        fail_count = 0

        for i in range(MESSAGE_COUNT):
            msg = f"Hello from Python Publisher #{i}"
            data = msg.encode('utf-8')

            print(f"[{i + 1}/{MESSAGE_COUNT}] Publishing: \"{msg}\"")
            print(f"              Length: {len(data)} bytes")

            if pydds.publish(TOPIC_ID, data):
                print("              Status: ✓ Success")
                success_count += 1
            else:
                error = pydds.get_last_error()
                print(f"              Status: ✗ Failed - {error}")
                fail_count += 1

            print()

            if i < MESSAGE_COUNT - 1:
                time.sleep(INTERVAL_SEC)

        print("-" * 60)
        print(f"\n[Statistics]")
        print(f"  Success: {success_count}")
        print(f"  Failed:  {fail_count}")
        print(f"  Total:   {MESSAGE_COUNT}")

        # 等待消息发送完成
        print(f"\n[Step 5] Waiting for messages to be delivered...")
        time.sleep(2)

    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user")

    finally:
        # 清理
        print("\n[Step 6] Shutting down DDS...")
        pydds.shutdown()
        print("[SUCCESS] Publisher shutdown complete!")

    print("\n" + "=" * 60)
    print("  Publisher Test Completed!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
