import os
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.iai.v20200303 import iai_client, models
import base64
import json
from tqdm import tqdm
import argparse

def get_image_base64(image_path):
    """将图片转换为base64编码"""
    with open(image_path, 'rb') as f:
        base64_data = base64.b64encode(f.read())
        return base64_data.decode()

def detect_faces(client, image_path):
    """检测单张图片中的人脸
    
    返回:
        boxes: 人脸框列表 [[x,y,w,h], ...]
    """
    try:
        # 创建请求对象
        req = models.DetectFaceRequest()
        
        # 读取图片并转base64
        image_base64 = get_image_base64(image_path)
        req.Image = image_base64
        
        # 调用接口
        resp = client.DetectFace(req)
        result = json.loads(resp.to_json_string())
        
        # 提取人脸框
        boxes = []
        if 'FaceInfos' in result:
            for face in result['FaceInfos']:
                x = face['X']
                y = face['Y'] 
                w = face['Width']
                h = face['Height']
                boxes.append([x, y, w, h])
                
        return boxes
    
    except Exception as e:
        print(f"检测失败 {image_path}: {str(e)}")
        return []

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='腾讯云人脸检测工具')
    parser.add_argument('--secret_id', required=True, help='腾讯云 SecretId')
    parser.add_argument('--secret_key', required=True, help='腾讯云 SecretKey')
    args = parser.parse_args()
    
    # 使用命令行参数创建认证信息
    cred = credential.Credential(args.secret_id, args.secret_key)
    
    # 创建client
    httpProfile = HttpProfile()
    httpProfile.endpoint = "iai.tencentcloudapi.com"
    clientProfile = ClientProfile()
    clientProfile.httpProfile = httpProfile
    client = iai_client.IaiClient(cred, "ap-guangzhou", clientProfile)
    
    # 图片目录
    image_dir = r"C:\Project\ai\cv-assignment\4. final\dataset\1face"
    
    # 输出文件
    output_file = "face_annotations.txt"
    
    # 遍历处理所有图片
    with open(output_file, 'w') as f:
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        
        for image_file in tqdm(image_files):
            image_path = os.path.join(image_dir, image_file)
            
            # 检测人脸
            boxes = detect_faces(client, image_path)
            
            # 写入结果
            f.write(f"{image_file}\n")
            f.write(f"{len(boxes)}\n")
            for box in boxes:
                f.write(f"{box[0]} {box[1]} {box[2]} {box[3]}\n")

if __name__ == "__main__":
    main()