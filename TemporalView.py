import torch
import torch.nn as nn

class TemporalView(nn.Module):
    def __init__(self, input_dim=64, context_dim=2, hidden_dim=64):
        super(TemporalView, self).__init__()
        
        # LSTM 구조 선언
        # 내부적으로 입력 게이트(i), 망각 게이트(f), 출력 게이트(o), 셀 상태(c) 연산 수행
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,        
            batch_first=True 
        )
        
    def forward(self, spatial_seq, context_seq):
        combined_input = torch.cat([spatial_seq, context_seq], dim=-1)

        hidden_state, (last_hidden_state, last_cell_state) = self.lstm(combined_input)

        return last_hidden_state[-1]