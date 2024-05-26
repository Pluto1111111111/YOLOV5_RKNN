import os
import xml.etree.ElementTree as ET

# yolo的txt文件只有五个数据（第一个代表类别（用数字从0开始），后四个为框的参数）
# 除了对应文件的txt文件，还有一个classes文件罗列了所有类别

# VOC数据集路径
voc_ann_path = r"C:\Users\Pluto\Desktop\VOC2007\Annotations\xml"
#voc_img_path = r"D:\lg\BaiduSyncdisk\project\person_code\project_self\chepai_OCR\data\traindata\VOC\images\train\images\train"
# YOLO数据集路径
yolo_out_path = r"C:\Users\Pluto\Desktop\VOC2007\Annotations\txt"
# VOC类别名称和对应的编号
classes = {"BIPV": 0}  # 根据实际情况修改
# 遍历VOC数据集文件夹
for filename in os.listdir(voc_ann_path):
    # 解析XML文件
    tree = ET.parse(os.path.join(voc_ann_path, filename))
    root = tree.getroot()
    # 获取图片尺寸
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    # 创建YOLO标注文件
    yolo_filename = filename.replace(".xml", ".txt")
    yolo_file = open(os.path.join(yolo_out_path, yolo_filename), "w")
    # 遍历XML文件中的所有目标
    for obj in root.findall("object"):
        # 获取目标类别名称和边界框坐标
        name = obj.find("name").text
        xmin = int(obj.find("bndbox").find("xmin").text)
        ymin = int(obj.find("bndbox").find("ymin").text)
        xmax = int(obj.find("bndbox").find("xmax").text)
        ymax = int(obj.find("bndbox").find("ymax").text)
        # 计算边界框中心点坐标和宽高
        x = (xmin + xmax) / 2 / width
        y = (ymin + ymax) / 2 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        # 将目标写入YOLO标注文件
        class_id = classes[name]
        yolo_file.write(f"{class_id} {x} {y} {w} {h}\n")
    yolo_file.close()

