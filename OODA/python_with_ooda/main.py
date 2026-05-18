import socket
import struct
import time
import math
import numpy as np
import threading

# 配置主机和端口，需要在这里配一次，在afsim脚本内配一次
HOST = '192.168.5.7'  # 本地IP地址
# HOST = '127.0.0.7'  # 本地IP地址
PORT = 39322  # 端口号
name_list = {'bomber_1': 10, 'bomber_2': 9, 'bomber_3': 6, 'bomber_4' : 7, 'HELICOPTER_1305' : 11,
             'HELICOPTER_2102' : 12, 'HELICOPTER_1303' : 13, 'HELICOPTER_2901' : 14, 'target_1' : 3,
             'target_2' : 4, 'target_3' : 2, 'target_4' : 1, 'flight_lead_south' : 8, 'flight_lead_north' : 5}
platform_packet = [0]*len(name_list)
for key in platform_packet:
    platform_packet[key] = 0
heart_beat = 0
packet_num = 0
# socket for AeroNet
AeroNet_ip = '192.168.1.99'
AeroNet_port = 1111
AeroNet_address = (AeroNet_ip, AeroNet_port)
AeroNet_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#1.定义数据包的头部，包含三个重要字段：xio_version表示版本,buffer_length表示数据包的长度,pkt_id表示数据包标识符
class Header(object):
    def __init__(self, xio_version, buffer_length, pkt_id):
        self.xio_version = xio_version
        self.buffer_length = buffer_length
        self.pkt_id = pkt_id

#定义一个字典send_params，包含默认参数，后续用于发送数据包中
# 默认发送数据的参数
send_params = {
    "xio_version": 1,
    "buffer_length": 300,
    "pkt_id": 60,
    "platform_index": 400,
    "speed": 800,
    "acceleration": 100,
    "altitude": 5000,
    "climb_rate": 100,
    "max_gravity": 1,
    "angle": 180,
    "radial_accel": 10,
    "max_climb_rate": 100,
    "identifier": b'bomber_77',
}

#这里还不完善，提供了修改参数的界面，用于动态修改发送参数
def modify_send_params():
    """
    非阻塞修改发送参数的线程函数。
    """
    while True:
        print("\n当前发送参数：")
        for key, value in send_params.items():
            print(f"{key}: {value}")

        param_name = input("\n输入要修改的参数名 (输入 'done' 完成修改): ").strip()
        if param_name == 'done':
            break
        elif param_name in send_params:
            new_value = input(f"请输入新的值 (当前值: {send_params[param_name]}): ").strip()
            if isinstance(send_params[param_name], bytes):
                send_params[param_name] = new_value.encode('utf-8')
            else:
                send_params[param_name] = type(send_params[param_name])(new_value)
            print(f"参数 {param_name} 已更新为: {send_params[param_name]}")
        else:
            print("参数名无效，请重新输入！")


# 启动非阻塞的参数修改线程
#def start_edit_thread():

    #启动一个线程用于参数修改。

#    thread = threading.Thread(target=modify_send_params, daemon=True)
#    thread.start()
"""
def train_data(params_):
    # 调用train 将参数穿进去
    pass
"""

# 创建 UDP 套接字，绑定到指定的IP地址和端口上，服务器开始监听该端口，等待接收数据包
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server_socket:     #socket.AF_INET指定使用IPv4地址，socket.SOCK_DGRAM 表示使用数据报套接字，即UDP协议
    # 绑定主机和端口
    server_socket.bind((HOST, PORT))                #使用bind方法将服务器的UDP套接字绑定到指定的HOST和PORT上，该服务器将会监听该IP地址和端口号，等待接收来自客户端的数据包
    print(f"UDP 服务端已启动，正在监听 {HOST}:{PORT}")  #成功绑定，输出启动信息
    last_process_time = time.time()                 #记录最后一次处理数据包的时间，确保定时处理每隔一定时间的逻辑。
    application_id = (0, 0, 0)                      #初始化几个数据为默认值
    flags_id = (0,)
    timestamp = (0,)

     #启动参数修改线程
#    start_edit_thread()


    #服务器不断监听并处理接收到的数据包
    while True:
        try:
            # 设置超时时间，recvfrom()方法如果在1秒内没收到数据包会抛出socket.timeout异常，跳到except socket.timeout:进行处理
            server_socket.settimeout(1)
            # 接收数据（缓冲区大小为 1024 字节）
            data, addr = server_socket.recvfrom(1024)   #从套接字中接收最多1024字节的数据，返回值是data（接收到的数据）和addr（发送者的地址）
            # print(f"接收到来自 {addr} 的数据")

            # 解析 header 数据
            header_data = struct.unpack(">III", data[0:12])
            cur_header = Header(header_data[0], header_data[1], header_data[2])
            # print(f"header_data:{header_data}")  # pkt_id

            # 根据 pkt_id 处理不同类型的数据包
            if cur_header.pkt_id == 1:  # 心跳包
                heart_beat += 1
                print('receive heartbeat data')
                application_id = struct.unpack('>III', data[12:24])
                # print(f"application_id：{application_id}")
                flags_id = struct.unpack('>I', data[24:28])
                # print(f"flags_id: {flags_id}")
                timestamp = struct.unpack('>f', data[28:32])
                # print(f"timestamp: {timestamp}")

       #         train_data(111)

            elif cur_header.pkt_id == 200:  # stage包
                # print(f"收到Stage包")
                pos = struct.unpack('>I', data[32:36])
                # print(f'pos:{pos[0]}')
                cur_pos = 36 + pos[0]
                name = struct.unpack(f'>{pos[0]}s', data[36:cur_pos])[0].decode('utf-8')
                # print(f"name:{name}")
                speed = struct.unpack('>I', data[cur_pos:cur_pos + 4])[0]
                # print(f"speed:{speed}")
                cur_pos += 4
                altitude = struct.unpack(f'>I', data[cur_pos:cur_pos + 4])[0]
                # print(f"altitude:{altitude}")
                cur_pos += 4
                length = struct.unpack(f'>d', data[cur_pos:cur_pos + 8])
                # print(f"length:{length}")
                cur_pos += 8
                width = struct.unpack(f'>d', data[cur_pos:cur_pos + 8])
                # print(f"width:{width}")
                cur_pos += 8
                height = struct.unpack(f'>d', data[cur_pos:cur_pos + 8])
                # print(f"height:{height}")
                cur_pos += 8
                heading = struct.unpack(f'>d', data[cur_pos:cur_pos + 8])[0]
                heading_degree = heading * (180 / math.pi)
                # print(f"heading(rad):{heading}")
                # print(f"heading(degree):{heading_degree}")
                cur_pos += 8
                pitch = struct.unpack(f'>d', data[cur_pos:cur_pos + 8])[0]
                # print(f"pitch:{pitch}")
                cur_pos += 8
                roll = struct.unpack(f'>d', data[cur_pos:cur_pos + 8])[0]
                # print(f"roll:{roll}")
                cur_pos += 8
                latitude = struct.unpack(f'>d', data[cur_pos:cur_pos +8])[0]
                # print(f"latitude:{latitude}")
                cur_pos += 8
                longitude = struct.unpack(f'>d', data[cur_pos:cur_pos + 8])[0]
                # print(f"longitude:{longitude}")

                if name in name_list:
                    platform_packet[name_list[name]-1] = 1
                    if name_list[name] > 4:
                        packet_num += 1
                    msg_to_AeroNet = struct.pack('iiddddddd', 5, name_list[name], longitude, latitude, altitude, heading_degree, roll, pitch, speed)
                    print(f"msg_to_AeroNet for platform :{name_list[name]}")
                    AeroNet_socket.sendto(msg_to_AeroNet, AeroNet_address)


            # 每秒发送逻辑
            process_time = time.time() - last_process_time
            # print(f'platform_packet: {platform_packet}, heart_beat: {heart_beat}')

            # if packet_num > 9 or heart_beat == 1:
            if process_time > 0.1:
                heart_beat = -999999999999999
                platform_packet = [0] * len(name_list)
                packet_num = 0
                # if process_time > 1:
                    # print("每10秒钟处理逻辑运行")
                last_process_time = time.time()

                app_name = 'PythonControlAFsim'
                cur_header.pkt_id = 60
                cur_header.buffer_length = 300
                # 设置发送的 scriptlist 包
                packed_data = struct.pack('>IIIIIIIfIIIIIIIIII9s',
                                              cur_header.xio_version,
                                              cur_header.buffer_length,
                                              cur_header.pkt_id,
                                              *application_id,
                                              flags_id[0],
                                              timestamp[0],
                                              send_params["platform_index"],
                                              send_params["speed"],
                                              send_params["acceleration"],
                                              send_params["altitude"],
                                              send_params["climb_rate"],
                                              send_params["max_gravity"],
                                              send_params["angle"],
                                              send_params["radial_accel"],
                                              send_params["max_climb_rate"],
                                              len(send_params["identifier"]),
                                              send_params["identifier"])
                # print(f"发送的 packed_data:{packed_data}")
                print(f"请求一次数据,处理间隔为{process_time}")
                server_socket.sendto(packed_data, addr)
                # time.sleep(0.5)

        except socket.timeout:
            pass
        except Exception as e:
            print(f"发生错误: {e}")
            break



