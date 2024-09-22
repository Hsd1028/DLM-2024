import torch
from torchvision import transforms
from PIL import Image
import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# 确保CUDA可用，如果不可用则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载预训练的Swin Transformer模型
model_name = 'swin_tiny_patch4_window7_224'  # 这里可以根据需要选择不同的模型版本
model = timm.create_model(model_name, pretrained=True)
model = model.to(device)
model.eval()


# 图像预处理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 添加一个批次维度


# 预测函数
def predict(image_path, model):
    image_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
    return outputs


# 使用模型进行预测
image_path = 'test_data/01.jpg'  # 请替换为你的图片路径
outputs = predict(image_path, model)

# 输出预测结果
_, predicted = torch.max(outputs, 1)
print(f'Predicted label: {predicted.item()}')  # 输出预测的类别ID
