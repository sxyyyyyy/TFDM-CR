from torch.nn.modules import loss
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from gluonts.core.component import validated
from utils import weighted_average,MeanScaler,NOPScaler
from module import GaussianDiffusion,DiffusionOutput
import numpy as np
from epsilon_theta import EpsilonTheta
from Conv_FT1 import Conv_FT
import torch.nn.functional as F
from CR_compensate import CR_compensate
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
class TrainingNetwork(nn.Module):
    @validated()
    def __init__(
        self,
        series: np.ndarray,
        input_size: int, 
        num_layers: int, 
        num_cells: int, 
        cell_type: str, 
        history_length: int, 
        context_length: int, 
        prediction_length: int,
        dropout_rate: float, 
        lags_seq: List[int], 
        target_dim: int, 
        conditioning_length: int, 
        diff_steps: int, 
        loss_type: str, 
        beta_end: float, 
        beta_schedule: str, 
        residual_layers: int, 
        residual_channels: int, 
        dilation_cycle_length: int,
        cardinality: List[int] = [1],
        embedding_dimension: int = 1,
        scaling: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.target_dim = target_dim 
        self.prediction_length = prediction_length 
        self.context_length = context_length 
        self.history_length = history_length
        self.scaling = scaling
        assert len(set(lags_seq)) == len(lags_seq)
        lags_seq.sort()
        self.lags_seq = lags_seq
        self.weight1 = nn.Parameter(torch.rand(1, 1, 1))
        self.weight2 = nn.Parameter(torch.rand(1, 1, 1))
        self.series = series
        self.cell_type = cell_type

        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type] 
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=num_cells,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=False,
        ) 
        self.fre = Conv_FT(self.target_dim, num_cells, self.context_length).to(device)
        self.CR_compensate = CR_compensate()
        self.denoise_fn = EpsilonTheta(
            target_dim=target_dim,
            cond_length=conditioning_length,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            dilation_cycle_length=dilation_cycle_length,
        ) 

        self.diffusion = GaussianDiffusion(
            self.denoise_fn,
            input_size=target_dim,
            diff_steps=diff_steps,
            loss_type=loss_type,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
        ) 

        self.distr_output = DiffusionOutput(
            self.diffusion, input_size=target_dim, cond_size=conditioning_length
        ) 
        self.proj_dist_args = self.distr_output.get_args_proj(num_cells) 
        self.embed_dim = 1
        self.embed = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.embed_dim
        )

        if self.scaling:
            self.scaler = MeanScaler(keepdim=True) 
        else:
            self.scaler = NOPScaler(keepdim=True)
    

    @staticmethod
    def get_lagged_subsequences(
        sequence: torch.Tensor,
        sequence_length: int,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> torch.Tensor:

        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag "
            f"{max(indices)} while history length is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length 
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...].unsqueeze(1)) 
        return torch.cat(lagged_values, dim=1).permute(0, 2, 3, 1) 
    
    def unroll(
        self,
        lags: torch.Tensor,
        scale: torch.Tensor,
        time_feat: torch.Tensor,
        target_dimension_indicator: torch.Tensor,
        unroll_length: int,
        begin_state: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        lags_scaled = lags / scale.unsqueeze(-1)
        input_lags = lags_scaled.reshape(
            (-1, unroll_length, len(self.lags_seq) * self.target_dim)
        ) 
        
        index_embeddings = self.embed(target_dimension_indicator)

        repeated_index_embeddings = (
            index_embeddings.unsqueeze(1)
            .expand(-1, unroll_length, -1, -1)
            .reshape((-1, unroll_length, self.target_dim * self.embed_dim))
        )

        inputs = torch.cat((input_lags, repeated_index_embeddings, time_feat), dim=-1)
        outputs, state = self.rnn(inputs, begin_state)
        return outputs, state, lags_scaled, inputs

    def unroll_encoder(
        self,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: Optional[torch.Tensor],
        future_target_cdf: Optional[torch.Tensor],
        target_dimension_indicator: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        if future_time_feat is None or future_target_cdf is None:
            time_feat = past_time_feat[:, -self.context_length :, ...]
            sequence = past_target_cdf
            sequence_length = self.history_length
            subsequences_length = self.context_length
        else:
            time_feat = torch.cat(
                (past_time_feat[:, -self.context_length :, ...], future_time_feat),
                dim=1,
            )
            sequence = torch.cat((past_target_cdf, future_target_cdf), dim=1)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        lags = self.get_lagged_subsequences(
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        )

        _, scale = self.scaler(
            past_target_cdf[:, -self.context_length :, ...],
            past_observed_values[:, -self.context_length :, ...],
        )

        outputs, states, lags_scaled, inputs = self.unroll(
            lags=lags,
            scale=scale,
            time_feat=time_feat,
            target_dimension_indicator=target_dimension_indicator,
            unroll_length=subsequences_length,
            begin_state=None,
        )

        return outputs, states, scale, lags_scaled, inputs

    def distr_args(self, rnn_outputs: torch.Tensor):
        (distr_args,) = self.proj_dist_args(rnn_outputs)
        return distr_args

    def forward(
        self,
        target_dimension_indicator: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target_cdf: torch.Tensor,                                                                                                                                                                                                                                                                                                                                                                 
        future_observed_values: torch.Tensor,

    ) -> Tuple[torch.Tensor, ...]:

        rnn_outputs, _, scale, _, _ = self.unroll_encoder(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,
            future_target_cdf=future_target_cdf,
            target_dimension_indicator=target_dimension_indicator,
        )
        
        target = torch.cat(
            (past_target_cdf[:, -self.context_length :, ...], future_target_cdf),
            dim=1,
        )

        freq_feature = self.fre(past_target_cdf[:, -self.context_length :, ...], sign=1)
        rnn_outputs0 = self.weight1 * rnn_outputs[:, :self.context_length,:] + self.weight2 * freq_feature   
        rnn_outputs = torch.cat((rnn_outputs0, rnn_outputs[:, self.context_length:, :]), dim=1)
        distr_args = self.distr_args(rnn_outputs=rnn_outputs)
 
        if self.scaling:
            self.diffusion.scale = scale

        likelihoods = self.diffusion.log_prob(target, distr_args).unsqueeze(-1)  

        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        observed_values = torch.cat(
            (
                past_observed_values[:, -self.context_length :, ...],
                future_observed_values,
            ),
            dim=1,
        )
        loss_weights, _ = observed_values.min(dim=-1, keepdim=True)

        loss = weighted_average(likelihoods, weights=loss_weights, dim=1)  
        
        return (loss.mean(), likelihoods, distr_args)

class PredictionNetwork(TrainingNetwork):
    def __init__(self, num_parallel_samples: int,**kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    def forward(
        self,
        target_dimension_indicator: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
    ) -> torch.Tensor:
        def repeat(tensor, dim=0):
            return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)
        
        series = torch.tensor(self.series, device='cuda:1').permute(1,0).unsqueeze(0)
        repeating_pattern = future_time_feat[:,:self.lags_seq[1],:].cpu().numpy()
        repeat_times = series.shape[1] // self.lags_seq[1]
        expanded_time_feat = np.tile(repeating_pattern, (1, repeat_times, 1))
        future_time_feat = torch.tensor(expanded_time_feat).to(device)
        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )
        series = repeat(series[:,-self.prediction_length:,:])
        past_series = repeat(series[:,:-self.prediction_length,:])

        repeated_past_target_cdf = repeat(past_target_cdf)
        repeated_time_feat = repeat(future_time_feat)

        repeated_target_dimension_indicator = repeat(target_dimension_indicator)

        future_samples = []
        
        repeating_pattern1 = past_time_feat[:,:self.lags_seq[1],:].cpu().numpy()
        repeat_times1 = 882 // self.lags_seq[1]

        expanded_time_feat = np.tile(repeating_pattern1, (1, repeat_times1, 1))
        past_time_feat = torch.tensor(expanded_time_feat).to(device)

        for k in range(self.prediction_length):                
            _, begin_states, scale, _, _ = self.unroll_encoder(
                past_time_feat=past_time_feat,
                past_target_cdf=past_target_cdf,
                past_observed_values=past_observed_values,
                past_is_pad=past_is_pad,
                future_time_feat=None,
                future_target_cdf=None,
                target_dimension_indicator=target_dimension_indicator,
            )
            if self.cell_type == "LSTM":
                repeated_states = [repeat(s, dim=1) for s in begin_states]
            else:
                repeated_states = repeat(begin_states, dim=1)
            repeated_scale = repeat(scale)
            if self.scaling:
                self.diffusion.scale = repeated_scale

            lags = self.get_lagged_subsequences(
                sequence=repeated_past_target_cdf,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            rnn_outputs, repeated_states, _, _ = self.unroll(
                begin_state=repeated_states,
                lags=lags,
                scale=repeated_scale,
                time_feat=repeated_time_feat[:, k : k + 1, ...],
                target_dimension_indicator=repeated_target_dimension_indicator,
                unroll_length=1,
            )

            freq_f = self.fre(repeated_past_target_cdf[:, -self.context_length :, ...], sign=0)
            rnn_outputs = self.weight1 * rnn_outputs + self.weight2 * freq_f   
            distr_args = self.distr_args(rnn_outputs=rnn_outputs)
            new_samples0 = self.diffusion.sample(cond=distr_args)
            new_samples = new_samples0
            future_samples.append(new_samples)
            repeated_past_target_cdf = torch.cat(
                (repeated_past_target_cdf, series[:, k : k + 1,...]), dim=1
            )
            past_target_cdf = torch.cat(
                (past_target_cdf, series[:1, k : k + 1,...]), dim=1
            )
        samples = torch.cat(future_samples, dim=1)
        samples = self.CR_compensate(samples, past_series)

        return samples.reshape(
            (
                -1,
                self.num_parallel_samples,
                self.prediction_length,
                self.target_dim,
            )
        )
    




