# scripts/train.py

import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.dataset import CustomDataset
from src.transforms import TransformSelector
from src.models import ModelSelector
from src.loss import Loss
from src.trainer import Trainer

def main():
    # 학습에 사용할 장비 선택
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 경로 설정
    traindata_dir = "./data/train"
    traindata_info_file = "./data/train.csv"
    save_result_path = "./train_result"

    # 데이터 로드
    train_info = pd.read_csv(traindata_info_file)
    num_classes = len(train_info['target'].unique())

    # 데이터 분할
    train_df, val_df = train_test_split(
        train_info, 
        test_size=0.2,
        stratify=train_info['target'],
        random_state=42
    )

    # 변환 설정
    transform_selector = TransformSelector(transform_type="albumentations")
    train_transform = transform_selector.get_transform(is_train=True)
    val_transform = transform_selector.get_transform(is_train=False)

    # 데이터셋 및 데이터로더
    train_dataset = CustomDataset(
        root_dir=traindata_dir,
        info_df=train_df,
        transform=train_transform
    )
    val_dataset = CustomDataset(
        root_dir=traindata_dir,
        info_df=val_df,
        transform=val_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # 모델 설정
    model_selector = ModelSelector(
        model_type='timm', 
        num_classes=num_classes,
        model_name='resnet18', 
        pretrained=True
    )
    model = model_selector.get_model()
    model.to(device)

    # 옵티마이저 및 스케줄러 설정
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    steps_per_epoch = len(train_loader)
    epochs_per_lr_decay = 2
    scheduler_step_size = steps_per_epoch * epochs_per_lr_decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=0.1)

    # 손실 함수 설정
    loss_fn = Loss()

    # 트레이너 설정
    trainer = Trainer(
        model=model, 
        device=device, 
        train_loader=train_loader,
        val_loader=val_loader, 
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn, 
        epochs=5,
        result_path=save_result_path
    )

    # 학습 시작
    trainer.train()

if __name__ == "__main__":
    main()
