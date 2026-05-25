# -*- coding: utf-8 -*-
"""손실 함수 모음."""

import numpy as np


def cross_entropy_loss(y_pred, y_true):
    """
    Cross Entropy Error (배치 평균).
    y_pred: (batch_size, 10) 확률
    y_true: (batch_size,) 정수 레이블 0~9
    """
    # 정답 클래스 확률의 log 값을 이용해 batch 평균 cross entropy를 계산하세요.
    # 힌트: np.clip으로 log(0)을 피하고, np.arange(batch_size)로 정답 위치를 고릅니다.
    
    if y_pred.ndim == 1:  #두 데이터는 모두 행이 1개이고 열이 데이터 개수인 2차원 행렬(Matrix)로 변환됩니다.
      y_true = y_true.reshape(1, y_true.size)
      y_pred = y_pred.reshape(1, y_pred.size)

    batch_size = y_pred.shape[0] 
    y_pred = np.clip(y_pred,1e-7,1)
    rows = np.arange(batch_size)
  
    # 모델이 예측한 확률 배열(y_pred)에서, 실제 정답 레이블(y_true)이 위치한 곳의 확률만 쏙쏙 골라냅니다.
    target_probabilities = y_pred[rows, y_true]

    log_probabilities = np.log(target_probabilities)

    # 정답을 맞출 확률이 낮았을수록 이 총합(오차)은 엄청나게 커집니다.
    total_loss = -np.sum(log_probabilities)

    # 총 오차를 데이터 개수(batch_size)로 나누어 '평균 오차'를 구한 뒤 반환(return)합니다.
    return total_loss / batch_size
