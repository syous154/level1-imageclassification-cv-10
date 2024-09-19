# src/trainer.py

import os
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

class Trainer:
    """
    모델의 훈련 및 검증을 관리하는 클래스.
    """
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        loss_fn: nn.Module, 
        epochs: int,
        result_path: str
    ):
        """
        Trainer 클래스의 초기화 메서드.

        Args:
            model (nn.Module): 훈련할 모델.
            device (torch.device): 연산을 수행할 디바이스 (CPU 또는 GPU).
            train_loader (DataLoader): 훈련 데이터 로더.
            val_loader (DataLoader): 검증 데이터 로더.
            optimizer (optim.Optimizer): 최적화 알고리즘.
            scheduler (optim.lr_scheduler._LRScheduler): 학습률 스케줄러.
            loss_fn (nn.Module): 손실 함수.
            epochs (int): 총 훈련 에폭 수.
            result_path (str): 모델 저장 경로.
        """
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.result_path = result_path
        self.best_models: List[Tuple[float, int, str]] = []
        self.lowest_loss = float('inf')

    def save_model(self, epoch: int, loss: float) -> None:
        """
        모델을 저장하고, 최상위 3개의 모델을 관리합니다.

        Args:
            epoch (int): 현재 에폭 번호.
            loss (float): 현재 에폭의 검증 손실.
        """
        os.makedirs(self.result_path, exist_ok=True)
        current_model_path = os.path.join(self.result_path, f'model_epoch_{epoch}_loss_{loss:.4f}.pt')
        torch.save(self.model.state_dict(), current_model_path)

        self.best_models.append((loss, epoch, current_model_path))
        self.best_models.sort()
        if len(self.best_models) > 3:
            _, _, path_to_remove = self.best_models.pop(-1)
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        if loss < self.lowest_loss:
            self.lowest_loss = loss
            best_model_path = os.path.join(self.result_path, 'best_model.pt')
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Saved best model at epoch {epoch} with loss {loss:.4f}")

    def train_epoch(self) -> float:
        """
        한 에폭 동안의 훈련을 수행합니다.

        Returns:
            float: 훈련 손실의 평균값.
        """
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        """
        모델의 검증을 수행합니다.

        Returns:
            float: 검증 손실의 평균값.
        """
        self.model.eval()
        total_loss = 0.0
        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)

        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

        return total_loss / len(self.val_loader)

    def train(self) -> None:
        """
        전체 훈련 과정을 관리합니다.
        """
        for epoch in range(1, self.epochs + 1):
            print(f"Epoch {epoch}/{self.epochs}")
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}\n")

            self.save_model(epoch, val_loss)
            # 스케줄러는 이미 train_epoch에서 step을 호출하므로 여기서는 추가 호출이 필요 없을 수 있습니다.
            # 만약 필요하다면 주석을 해제하세요.
            # self.scheduler.step()
