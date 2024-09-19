# src/inference.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import List
import os

def inference(
    model: nn.Module, 
    device: torch.device, 
    test_loader: DataLoader
) -> List[int]:
    """
    모델을 사용하여 테스트 데이터에 대한 예측을 수행합니다.

    Args:
        model (nn.Module): 학습된 모델.
        device (torch.device): 연산을 수행할 디바이스 (CPU 또는 GPU).
        test_loader (DataLoader): 테스트 데이터 로더.

    Returns:
        List[int]: 테스트 데이터에 대한 예측된 레이블 목록.
    """
    model.to(device)
    model.eval()
    
    predictions = []
    progress_bar = tqdm(test_loader, desc="Inference", leave=False)

    with torch.no_grad():
        for images in progress_bar:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())
            progress_bar.set_postfix(predictions=len(predictions))
    
    return predictions
