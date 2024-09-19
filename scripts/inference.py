# scripts/inference.py

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.dataset import CustomDataset
from src.transforms import TransformSelector
from src.models import ModelSelector
from src.inference import inference

def main():
    # 추론에 사용할 장비 선택
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 추론 데이터의 경로와 정보를 가진 파일의 경로 설정
    testdata_dir = "./data/test"
    testdata_info_file = "./data/test.csv"
    save_result_path = "./train_result"
    
    # 추론 데이터의 정보 로드
    test_info = pd.read_csv(testdata_info_file)
    
    # 총 클래스 수 (필요에 따라 수정)
    num_classes = 500
    
    # 변환 설정
    transform_selector = TransformSelector(transform_type="albumentations")
    test_transform = transform_selector.get_transform(is_train=False)
    
    # 테스트 데이터셋 및 데이터로더 설정
    test_dataset = CustomDataset(
        root_dir=testdata_dir,
        info_df=test_info,
        transform=test_transform,
        is_inference=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        shuffle=False,
        drop_last=False
    )
    
    # 모델 설정
    model_selector = ModelSelector(
        model_type='timm', 
        num_classes=num_classes,
        model_name='resnet18', 
        pretrained=False
    )
    model = model_selector.get_model()
    
    # best epoch 모델 로드
    best_model_path = os.path.join(save_result_path, "best_model.pt")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}")
    
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    
    # 추론 수행
    predictions = inference(
        model=model, 
        device=device, 
        test_loader=test_loader
    )
    
    # 예측 결과를 테스트 정보에 추가
    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    
    # 결과를 CSV 파일로 저장
    output_path = os.path.join(save_result_path, "output.csv")
    test_info.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    main()
