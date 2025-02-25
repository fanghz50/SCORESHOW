import cv2
import numpy as np
from PySide6.QtGui import QImage, Qt
from PySide6.QtCore import QThread, Signal


class Camera:
    def __init__(self, cam_preset_num=4):#最大检测的摄像头数量
        self.cam_preset_num = cam_preset_num

    def get_cam_num(self):#检测本地连接的摄像头数量，并返回可用摄像头的数量及其设备索引列表。
        cnt = 0
        devices = []
        for device in range(0, self.cam_preset_num):
            stream = cv2.VideoCapture(device)
            grabbed = stream.grab()#尝试从摄像头抓取一帧图像,成功抓取，返回 True
#################################################################################################
            stream.release()#释放摄像头资源，避免占用设备
            if not grabbed:
                continue
            else:
                cnt = cnt + 1
                devices.append(device)
        return cnt, devices


# class WebcamThread(QThread):
#     changePixmap = Signal(np.ndarray)

#     def __init__(self, cam, parent=None):
#         QThread.__init__(self, parent)
#         self.cam = cam

#     def run(self):
#         cap = cv2.VideoCapture(self.cam)
#         ret, frame = cap.read()
#         if ret:
#             self.changePixmap.emit(frame)
#         cap.release()

#读取单个摄像头数据帧
class WebcamThread(QThread):
    # changePixmap = Signal(np.ndarray, int)  # 修改信号，增加摄像头ID
    changePixmap = Signal(np.ndarray)

    def __init__(self, cam, parent=None):
    # def __init__(self, cam, cam_id, parent=None):  # 增加cam_id
        super().__init__(parent)
        self.cam = cam
        # self.cam_id = cam_id  # 保存摄像头ID
        self.running = True  # 添加运行标志

    def run(self):
        cap = cv2.VideoCapture(self.cam)
        while self.running and cap.isOpened(): # 检查运行标志和摄像头是否打开
            ret, frame = cap.read()#一帧图像数据 frame 
            if ret:
                # self.changePixmap.emit(frame, self.cam_id)  # 发送帧和摄像头ID
                self.changePixmap.emit(frame)#发出 changePixmap 信号，并将 frame 数据传递给所有连接到该信号的槽函数
            else:
                break # 如果无法读取帧，退出循环
        cap.release() # 释放摄像头资源

    def stop(self):
        self.running = False # 设置运行标志为False，停止线程

# #读取多个摄像头数据帧
# class CameraReader:
#     def __init__(self, cams):
#         self.threads = []
#         self.window_names = {}  # 存储窗口名称
#         # for cam in cams:
#         for cam_id, cam in enumerate(cams):  # 使用enumerate获取摄像头ID
#             thread = WebcamThread(cam, cam_id)  # 传递摄像头ID
#             # thread = WebcamThread(cam)
#             thread.changePixmap.connect(self.show_frame)
#             self.threads.append(thread)
#             self.window_names[cam_id] = f"Camera {cam_id}"  # 创建窗口名称


#     def start_reading(self):#启动所有摄像头线程。
#         for thread in self.threads:
#             thread.start()

#     def stop_reading(self):#关闭所有摄像头线程。
#         for thread in self.threads:
#             thread.stop()
#             thread.wait() # 等待线程结束

#     def show_frame(self, frame, cam_id):  # 接收帧和摄像头ID
#         # 在这里处理图像数据，例如显示在不同的窗口中
#         window_name = self.window_names[cam_id]  # 获取窗口名称
#         cv2.imshow(window_name, frame)  # 使用窗口名称显示图像
#         cv2.waitKey(1)

# 使用示例
# cams = [0, 1, 2]  # 摄像头索引列表
# camera_reader = CameraReader(cams)
# camera_reader.start_reading()

# 在适当的时候停止读取
# camera_reader.stop_reading()


if __name__ == '__main__':
    cam = Camera()
    cam_num, devices = cam.get_cam_num()
    print(cam_num, devices)
