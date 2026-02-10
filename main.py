import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model import DMVSTNet 

# =====================================================================
# [설정] 데이터 그리드 크기 (고정)
GRID_H = 12
GRID_W = 12
NUM_NODES = GRID_H * GRID_W  # 144개

# ---------------------------------------------------------------------
# 1. 시각화 함수
# ---------------------------------------------------------------------
def evaluate_and_visualize(real_targets, real_preds, node_idxes, save_name="prediction_result.png"):
    plot_len = min(300, len(real_targets))

    valid_idxes = [i for i in node_idxes if i < real_targets.shape[1]]
    
    ts_actual = real_targets[:plot_len, valid_idxes].sum(axis=1)
    ts_pred   = real_preds[:plot_len, valid_idxes].sum(axis=1)

    mae = np.mean(np.abs(ts_actual - ts_pred))

    plt.figure(figsize=(15, 5))
    plt.plot(ts_actual, label='Actual', color='black', alpha=0.7)
    plt.plot(ts_pred, label='Predicted', color='red', linestyle='--', alpha=0.9)
    
    plt.title(f"Taxi Demand Prediction (Nodes: {valid_idxes}) | MAE: {mae:.2f}")
    plt.xlabel("Time Steps")
    plt.ylabel("Demand (Count)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_name)
    plt.close()
    print(f"📈 결과 그래프 저장됨: {save_name}")

# ---------------------------------------------------------------------
# 2. 데이터셋 클래스
# ---------------------------------------------------------------------
class TaxiDemandDataset(Dataset):
    def __init__(self, x, context, y):
        self.x = torch.FloatTensor(x)          
        self.context = torch.FloatTensor(context)  
        self.y = torch.FloatTensor(y)          

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.context[idx], self.y[idx]

# ---------------------------------------------------------------------
# 3. 컨텍스트 생성
# ---------------------------------------------------------------------
def get_extended_context(demand_data, time_strs, num_nodes):
    print("   -> 컨텍스트 생성 중...")
    dates = pd.to_datetime(time_strs)
    
    # 1. 과거 4개 시간 평균 (Trend)
    df_demand = pd.DataFrame(demand_data)
    temporal_feat = df_demand.rolling(window=4, min_periods=1).mean().values
    temporal_feat = temporal_feat[:, :, np.newaxis] 

    # 2. 주말/공휴일
    is_weekend = (dates.weekday >= 5).astype(float)
    kr_holidays = [
        '2024-10-01', '2024-10-03', '2024-10-09', '2024-12-25',
        '2025-01-01', '2025-01-26', '2025-01-28', '2025-01-29', '2025-01-30', 
        '2025-03-01', '2025-03-03' 
    ]
    is_holiday = dates.strftime('%Y-%m-%d').isin(kr_holidays).astype(float)
    holiday_combined = np.maximum(is_weekend, is_holiday)
    holiday_feat = np.tile(holiday_combined[:, np.newaxis, np.newaxis], (1, num_nodes, 1))

    context_2d = np.concatenate([temporal_feat, holiday_feat], axis=-1)
    return context_2d.astype(np.float32)

# ---------------------------------------------------------------------
# 4. 슬라이딩 윈도우 (16x17 Reshape 적용)
# ---------------------------------------------------------------------
def sliding_window_transform(demand_data, context_data, seq_len):
    x_list, c_list, y_list = [], [], []
    num_samples = len(demand_data) - seq_len
    
    for i in range(num_samples):
        x_seq = demand_data[i : i + seq_len]

        x_seq = x_seq.reshape(seq_len, GRID_H, GRID_W) 
        
        y = demand_data[i + seq_len]
        
        x_list.append(x_seq)
        c_list.append(context_data[i : i + seq_len])
        y_list.append(y)
        
    return np.array(x_list), np.array(c_list), np.array(y_list)

# ---------------------------------------------------------------------
# 5. 메인 함수
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="gwn_data1.json")
    parser.add_argument("--vec_path", type=str, default="vec_all_norm.txt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--window", type=int, default=24)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # 1. 데이터 로드
    print(f"📂 데이터 로드: {args.json}")
    with open(args.json, 'r') as f:
        data_json = json.load(f)
        
    X_raw = np.array(data_json['x'], dtype=np.float32) 
    time_strs = data_json['meta']['hours']
    num_timesteps, num_nodes_data = X_raw.shape
    
    # 데이터 검증
    if num_nodes_data != NUM_NODES:
        print(f"⚠️ [주의] 데이터의 노드 수({num_nodes_data})가 설정된 데이터와 다름")

    print(f"   - 데이터 크기: Time={num_timesteps}, Nodes={num_nodes_data}")
    print(f"   - 그리드 설정: {GRID_H} x {GRID_W} (= {NUM_NODES})")

    # 2. Context 생성
    context_raw = get_extended_context(X_raw, time_strs, num_nodes_data) 
    
    # 3. Train/Test 분할
    split_idx = int(num_timesteps * 0.7)
    
    # 4. 슬라이딩 윈도우 변환
    print(f"🔄 데이터 변환 중 (Window={args.window}, Grid={GRID_H}x{GRID_W})...")
    X_train, C_train, Y_train = sliding_window_transform(
        X_raw[:split_idx], context_raw[:split_idx], args.window)
    
    X_test, C_test, Y_test = sliding_window_transform(
        X_raw[split_idx:], context_raw[split_idx:], args.window)

    train_loader = DataLoader(TaxiDemandDataset(X_train, C_train, Y_train), batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(TaxiDemandDataset(X_test, C_test, Y_test), batch_size=args.batch, shuffle=False)

    # 5. 모델 초기화
    try:
        print(f"📥 임베딩 로드: {args.vec_path}")
        embed_vecs = np.loadtxt(args.vec_path, dtype=np.float32)
        embed_vecs = torch.tensor(embed_vecs)
    except:
        print("⚠️ 임베딩 파일 에러 -> 랜덤 초기화")
        embed_vecs = torch.randn(num_nodes_data, 128)

    model = DMVSTNet(
        pretrained_embeddings=embed_vecs,
        num_nodes=num_nodes_data,
        grid_size=GRID_H,
        seq_len=args.window,
        spatial_out_dim=64,
        context_dim=2,
        lstm_hidden=64,
        semantic_out=6
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # 6. 학습 루프
    print("\n🏋️ 학습 시작!")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        custom_metric = 0.0
        
        for bx, bc, by in train_loader:
            bx, bc, by = bx.to(device), bc.to(device), by.to(device)
            
            optimizer.zero_grad()
            preds = model(bx, bc) 
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # [수정] 메모리 누수 방지 (.item())
            with torch.no_grad():
                diff = torch.abs(preds - by)
                metric = 0.1 * diff + diff / (by + 1)
                custom_metric += metric.mean().item()

        if epoch % 5 == 0:
            print(f"Epoch [{epoch:03d}/{args.epochs}] Loss: {train_loss/len(train_loader):.4f} | Custom: {custom_metric/len(train_loader):.4f}")

    # 7. 평가 및 저장
    print("\n🔍 최종 평가...")
    model.eval()
    preds_list, targets_list = [], []
    
    with torch.no_grad():
        for bx, bc, by in test_loader:
            bx, bc, by = bx.to(device), bc.to(device), by.to(device)
            preds = model(bx, bc)
            preds_list.append(preds.cpu().numpy())
            targets_list.append(by.cpu().numpy())
            
    preds_arr = np.concatenate(preds_list)
    targets_arr = np.concatenate(targets_list)
    preds_arr = np.maximum(preds_arr, 0)
    
    rmse = np.sqrt(mean_squared_error(targets_arr, preds_arr))
    mae = mean_absolute_error(targets_arr, preds_arr)
    
    mid_loss = np.abs(preds_arr.flatten() - targets_arr.flatten())
    loss_tensor = 0.1 * mid_loss + mid_loss / (targets_arr.flatten() + 1)
    recommended_loss1 = loss_tensor.mean()
    
    print("="*40)
    print(f"✅ Final RMSE: {rmse:.4f}")
    print(f"✅ Final MAE : {mae:.4f}")
    print(f"Recommended Loss : {recommended_loss1:.4f}")
    print("="*40)
    
    torch.save(model.state_dict(), "dmvst_final.pth")
    
    # 8. 그래프 그리기 (예시 노드)
    # 인덱스 범위 내의 값들로 확인 (예: 100번 노드, 200번 노드 등)
    evaluate_and_visualize(targets_arr, preds_arr, [0, 50, 100], "result_graph.png")

if __name__ == "__main__":
    main()