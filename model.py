import torch
import torch.nn as nn
from SpatialView import SpatialView
from SemanticView import SemanticView
from TemporalView import TemporalView
import torch.nn.functional as F

class DMVSTNet(nn.Module):
    def __init__(self, pretrained_embeddings, num_nodes=272, grid_size=16, seq_len=8, 
                spatial_out_dim=64,  # SpatialView의 d_out
                context_dim=2,       # TemporalView의 context_dim
                lstm_hidden=64,      # TemporalView의 hidden_dim
                semantic_out=6):     # SemanticView의 output_dim
        
        super(DMVSTNet, self).__init__()
        
        self.num_nodes = num_nodes
        self.grid_size = grid_size
        self.seq_len = seq_len

        self.spatial_view = SpatialView(
            city=grid_size, 
            d_out=spatial_out_dim
        )
        

        lstm_input_dim = spatial_out_dim + context_dim
        
        self.temporal_view = TemporalView(
            input_dim=lstm_input_dim,
            context_dim=context_dim,
            hidden_dim=lstm_hidden
        )


        self.semantic_view = SemanticView(
            pretrained_embeddings=pretrained_embeddings,
            out_dim=semantic_out
        )

        self.predict_fc = nn.Linear(lstm_hidden + semantic_out, 1)

    def forward(self, x, context):
        batch_size = x.size(0)

        # Step 1: Spatial View (공간적 특징 추출)
        x_reshaped = x.view(batch_size * self.seq_len, 1, self.grid_size, self.grid_size)

        s_feat = self.spatial_view(x_reshaped)
        
        # ==================================================================
        # Step 2: Temporal View (시계열 처리)

        s_feat = s_feat.view(batch_size, self.seq_len, self.num_nodes, -1)

        s_feat = s_feat.permute(0, 2, 1, 3).contiguous()

        s_feat_lstm = s_feat.view(batch_size * self.num_nodes, self.seq_len, -1)

        ctx_lstm = context.permute(0, 2, 1, 3).contiguous()
        ctx_lstm = ctx_lstm.view(batch_size * self.num_nodes, self.seq_len, -1)

        h_t = self.temporal_view(s_feat_lstm, ctx_lstm)
        
        # ==================================================================
        # Step 3: Semantic View (의미적 특징 추출)

        m_hat = self.semantic_view()

        m_hat_expanded = m_hat.unsqueeze(0).repeat(batch_size, 1, 1).view(-1, m_hat.size(1))
        
        # ==================================================================
        # Step 4: Fusion & Prediction
        combined = torch.cat([h_t, m_hat_expanded], dim=-1)
        
        prediction = F.softplus(self.predict_fc(combined))

        prediction = prediction.view(batch_size, self.num_nodes)
        
        return prediction