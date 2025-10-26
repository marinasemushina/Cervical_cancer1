# 🔧 НОВЫЙ ФАЙЛ - загрузка SSL весов
import torch
import os

def load_ssl_resnet50(ssl_path):
    """Создает ResNet50 с SSL весами"""
    from torchvision.models import resnet50
    
    # 1. Создаем пустую модель
    model = resnet50(pretrained=False)
    
    # 2. Загружаем SSL веса
    if not os.path.exists(ssl_path):
        print(f"❌ SSL weights not found at {ssl_path}")
        return resnet50(pretrained=True)  # Fallback to ImageNet
    
    ssl_state_dict = torch.load(ssl_path, map_location='cpu')
    print(f"📦 Loaded SSL weights, keys: {list(ssl_state_dict.keys())[:3]}...")
    
    # 3. Адаптируем ключи
    model_state_dict = model.state_dict()
    loaded_count = 0
    
    for ssl_key, ssl_value in ssl_state_dict.items():
        # Пробуем разные варианты ключей
        possible_keys = [
            ssl_key,
            ssl_key.replace('module.', ''),
            ssl_key.replace('encoder.', ''),
            ssl_key.replace('backbone.', ''),
            ssl_key.replace('fc.', ''),
        ]
        
        for possible_key in possible_keys:
            if possible_key in model_state_dict:
                if ssl_value.shape == model_state_dict[possible_key].shape:
                    model_state_dict[possible_key] = ssl_value
                    loaded_count += 1
                    break
    
    # 4. Загружаем веса
    model.load_state_dict(model_state_dict, strict=False)
    print(f"✅ Loaded {loaded_count}/{len(model_state_dict)} layers from SSL")
    
    return model
