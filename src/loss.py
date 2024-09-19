# src/loss.py

import torch.nn as nn

class Loss(nn.Module):
    """
    모델의 손실 함수를 계산하는 클래스.
    현재는 CrossEntropyLoss를 사용하지만, 필요에 따라 다른 손실 함수로 확장 가능.
    """
    def __init__(self):
        super(Loss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        순전파 단계에서 손실을 계산합니다.

        Args:
            outputs (torch.Tensor): 모델의 출력 logits.
            targets (torch.Tensor): 실제 레이블.

        Returns:
            torch.Tensor: 계산된 손실 값.
        """
        return self.loss_fn(outputs, targets)
