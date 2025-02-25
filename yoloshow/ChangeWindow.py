from utils import glo


def yoloshowvsSHOW():#显示src_vsmode对应窗口
    yoloshow_glo = glo.get_value('yoloshow')
    yoloshowvs_glo = glo.get_value('yoloshowvs')
    #字符串存储到全局变量管理中，键为 'yoloname1'
    glo.set_value('yoloname1', "yolov5 yolov7 yolov8 yolov9 yolov10 yolov5-seg yolov8-seg rtdetr yolov8-pose yolov8-obb")
    glo.set_value('yoloname2', "yolov5 yolov7 yolov8 yolov9 yolov10 yolov5-seg yolov8-seg rtdetr yolov8-pose yolov8-obb")
    #加载模型
    yoloshowvs_glo.reloadModel()
    #显示 yoloshowvs_glo 对应的窗口
    yoloshowvs_glo.show()
    yoloshow_glo.animation_window = None#清除之前的动画状态
    yoloshow_glo.closed.disconnect()#断开与该信号连接的所有槽函数

def yoloshowSHOW():#显示src_singlemode对应窗口
    yoloshow_glo = glo.get_value('yoloshow')
    yoloshowvs_glo = glo.get_value('yoloshowvs')
    glo.set_value('yoloname', "yolov5 yolov7 yolov8 yolov9 yolov10 yolov5-seg yolov8-seg rtdetr yolov8-pose yolov8-obb")
    yoloshow_glo.reloadModel()#根据全局变量 'yoloname' 的值加载对应的 YOLO 模型，并更新模型的配置。
    yoloshow_glo.show()
    yoloshowvs_glo.animation_window = None
    yoloshowvs_glo.closed.disconnect()
def yoloshow2vs():#从 yoloshow 窗口切换到 yoloshowvs 窗口
    yoloshow_glo = glo.get_value('yoloshow')
    yoloshow_glo.closed.connect(yoloshowvsSHOW)
    yoloshow_glo.close()
def view_to_1():#视角融合函数
    print("视角融合")
    return 0

def vs2yoloshow():#从 yoloshowvs 窗口切换回 yoloshow 窗口
    yoloshowvs_glo = glo.get_value('yoloshowvs')
    yoloshowvs_glo.closed.connect(yoloshowSHOW)
    yoloshowvs_glo.close()