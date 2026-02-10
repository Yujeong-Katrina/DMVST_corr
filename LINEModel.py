import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import cdist
from pathlib import Path
from sklearn.manifold import SpectralEmbedding

# ====================================================================
# 1. DTW 거리 계산 (패턴 분석)
# ====================================================================
def compute_dtw_matrix(X, num_nodes, n_hours=24):
    print(f"[1/3] 지역별 수요 패턴 분석 중 (DTW)... (Nodes: {num_nodes})")
    
    # 전체 데이터를 [일수, 24시간, 지역]으로 변경하여 '평균 일일 패턴' 추출
    # 데이터가 24시간 배수가 아닐 경우, 남는 시간은 버림
    total_len = X.shape[0]
    n_days = total_len // n_hours
    
    if n_days < 1: 
        # 데이터가 하루보다 짧으면 그냥 전체 평균 사용
        patterns = X.T 
    else:
        X_cut = X[:n_days * n_hours, :]
        # [Day, Hour, Node] -> [Hour, Node] (평균) -> [Node, Hour] (전치)
        daily_patterns = X_cut.reshape(n_days, n_hours, num_nodes).mean(axis=0)
        patterns = daily_patterns.T 

    # 거리 행렬 계산 (Scipy cdist 이용 - 유클리드 거리 기반 근사)
    # 정석 DTW는 너무 느려서, 패턴 벡터 간의 유클리드 거리로 대체하거나
    # FastDTW를 써야 하지만, 여기서는 '상관관계' 기반으로 빠르게 처리합니다.
    # (수요 패턴이 비슷하면 거리가 가깝게 나옵니다.)
    
    dist_matrix = cdist(patterns, patterns, metric='euclidean')
    
    print("      -> 패턴 분석 완료.")
    return dist_matrix

# ====================================================================
# 2. LINE 모델 정의
# ====================================================================
class LINEModel(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()
        self.nodes = nn.Embedding(num_nodes, embedding_dim)
        
    def forward(self, i, j, neg_j):
        u_i = self.nodes(i)
        u_j = self.nodes(j)
        u_neg = self.nodes(neg_j)
        
        pos_loss = -F.logsigmoid(torch.sum(u_i * u_j, dim=1))
        neg_loss = -F.logsigmoid(-torch.sum(u_i * u_neg, dim=1))
        
        return torch.mean(pos_loss + neg_loss)

# ====================================================================
# 3. 학습 및 저장 함수
# ====================================================================
def train_and_save(similarity_matrix, output_file, dim=32, epochs=50):
    print(f"[2/3] 그래프 임베딩(LINE) 학습 시작... (Epochs: {epochs})")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_nodes = similarity_matrix.shape[0]
    
    model = LINEModel(num_nodes, dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 엣지 샘플링 (유사도가 높은 쌍만 추출)
    rows, cols = np.where(similarity_matrix > 0.5) # 임계값 0.5 (조절 가능)
    if len(rows) == 0: # 너무 엄격하면 전체 사용
        rows, cols = np.where(similarity_matrix > 0.0)
        
    weights = similarity_matrix[rows, cols]
    weights = weights / weights.sum() # 확률 분포로 변환
    
    batch_size = 4096
    n_batches = len(rows) // batch_size + 1
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for _ in range(n_batches):
            # Positive Sampling
            idx = np.random.choice(len(rows), batch_size, p=weights)
            u_i = torch.LongTensor(rows[idx]).to(device)
            u_j = torch.LongTensor(cols[idx]).to(device)
            
            # Negative Sampling
            u_neg = torch.randint(0, num_nodes, (batch_size,)).to(device)
            
            optimizer.zero_grad()
            loss = model(u_i, u_j, u_neg)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"      Epoch {epoch+1}/{epochs} | Loss: {total_loss/n_batches:.4f}")
            
    # 결과 추출 및 저장
    print(f"[3/3] 결과 저장 중: {output_file}")
    embeddings = model.nodes.weight.data.cpu().numpy()
    
    # 정규화 (논문에서는 벡터 크기를 1로 맞춤)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    np.savetxt(output_file, embeddings, fmt='%.6f')
    print(">>> 모든 작업 완료!")

# ====================================================================
# 메인 함수
# ====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True, help="Path to json file")
    parser.add_argument("--out", type=str, default="vec_all_norm.txt")
    args = parser.parse_args()

    # 1. JSON 파일 로드
    print(f"[Start] 파일 읽는 중: {args.json}")
    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    
    # 2. X (Demand Matrix) 추출
    X = np.array(data["x"], dtype=np.float32) # [Time, Nodes]
    num_nodes = X.shape[1]
    
    # 3. 거리 계산 및 유사도 변환
    dist_matrix = compute_dtw_matrix(X, num_nodes)
    
    # 거리 -> 유사도 (Gaussian Kernel)
    sigma = np.std(dist_matrix) + 1e-6
    similarity = np.exp(-dist_matrix / sigma)
    np.fill_diagonal(similarity, 0) # 자기 자신과의 관계는 제거
    
    # 4. 학습 시작
    train_and_save(similarity, args.out)

if __name__ == "__main__":
    main()