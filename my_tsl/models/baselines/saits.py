"""
SAITS model for time-series imputation.

If you use code in this repository, please cite our paper as below. Many thanks.

@article{DU2023SAITS,
title = {{SAITS: Self-Attention-based Imputation for Time Series}},
journal = {Expert Systems with Applications},
volume = {219},
pages = {119619},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.119619},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423001203},
author = {Wenjie Du and David Cote and Yan Liu},
}

or

Wenjie Du, David Cote, and Yan Liu. SAITS: Self-Attention-based Imputation for Time Series. Expert Systems with Applications, 219:119619, 2023. https://doi.org/10.1016/j.eswa.2023.119619

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT

from einops import rearrange

from tsl.nn.models.base_model import BaseModel

from my_tsl.models.assets.saits_layers import *


class SAITSModel(BaseModel):
    """
    Implementation of SAITS model for time-series imputation.
    Adapted from Wenjie Du, David Cote, and Yan Liu. SAITS: Self-Attention-based Imputation for Time Series. 
    Expert Systems with Applications, 219:119619, 2023. https://doi.org/10.1016/j.eswa.2023.119619
    """
    def __init__(
        self,
        input_size,
        n_groups,
        n_group_inner_layers,
        d_time,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout,
        MIT=True,
        input_with_mask=True,
        **kwargs
    ):
        super().__init__()
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.input_with_mask = input_with_mask
        actual_d_feature = input_size * 2 if input_with_mask else input_size
        self.param_sharing_strategy = kwargs["param_sharing_strategy"]
        self.MIT = MIT
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if kwargs["param_sharing_strategy"] == "between_group":
            # For between_group, only need to create 1 group and repeat n_groups times while forwarding
            self.layer_stack_for_first_block = nn.ModuleList(
                [
                    EncoderLayer(
                        d_time,
                        actual_d_feature,
                        d_model,
                        d_inner,
                        n_head,
                        d_k,
                        d_v,
                        dropout,
                        0,
                        **kwargs
                    )
                    for _ in range(n_group_inner_layers)
                ]
            )
            self.layer_stack_for_second_block = nn.ModuleList(
                [
                    EncoderLayer(
                        d_time,
                        actual_d_feature,
                        d_model,
                        d_inner,
                        n_head,
                        d_k,
                        d_v,
                        dropout,
                        0,
                        **kwargs
                    )
                    for _ in range(n_group_inner_layers)
                ]
            )
        else:  # then inner_group，inner_group is the way used in ALBERT
            # For inner_group, only need to create n_groups layers
            # and repeat n_group_inner_layers times in each group while forwarding
            self.layer_stack_for_first_block = nn.ModuleList(
                [
                    EncoderLayer(
                        d_time,
                        actual_d_feature,
                        d_model,
                        d_inner,
                        n_head,
                        d_k,
                        d_v,
                        dropout,
                        0,
                        **kwargs
                    )
                    for _ in range(n_groups)
                ]
            )
            self.layer_stack_for_second_block = nn.ModuleList(
                [
                    EncoderLayer(
                        d_time,
                        actual_d_feature,
                        d_model,
                        d_inner,
                        n_head,
                        d_k,
                        d_v,
                        dropout,
                        0,
                        **kwargs
                    )
                    for _ in range(n_groups)
                ]
            )

        self.dropout = nn.Dropout(p=dropout)
        self.position_enc = PositionalEncoding(d_model, n_position=d_time)
        # for operation on time dim
        self.embedding_1 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_z = nn.Linear(d_model, input_size)
        # for operation on measurement dim
        self.embedding_2 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_beta = nn.Linear(d_model, input_size)
        self.reduce_dim_gamma = nn.Linear(input_size, input_size)
        # for delta decay factor
        self.weight_combine = nn.Linear(input_size + d_time, input_size)

    def impute(self, X, masks):
        # first DMSA block
        input_X_for_first = torch.cat([X, masks], dim=2) if self.input_with_mask else X
        input_X_for_first = self.embedding_1(input_X_for_first)
        enc_output = self.dropout(
            self.position_enc(input_X_for_first)
        )  # namely term e in math algo
        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_first_block:
                    enc_output, _ = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_first_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, _ = encoder_layer(enc_output)

        X_tilde_1 = self.reduce_dim_z(enc_output)
        X_prime = masks * X + (1 - masks) * X_tilde_1

        # second DMSA block
        input_X_for_second = (
            torch.cat([X_prime, masks], dim=2) if self.input_with_mask else X_prime
        )
        input_X_for_second = self.embedding_2(input_X_for_second)
        enc_output = self.position_enc(
            input_X_for_second
        )  # namely term alpha in math algo
        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_second_block:
                    enc_output, attn_weights = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_second_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, attn_weights = encoder_layer(enc_output)

        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

        # attention-weighted combine
        attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in math algo
        if len(attn_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)

        combining_weights = F.sigmoid(
            self.weight_combine(torch.cat([masks, attn_weights], dim=2))
        )  # namely term eta
        # combine X_tilde_1 and X_tilde_2
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1
        X_c = (
            masks * X + (1 - masks) * X_tilde_3
        )  # replace non-missing part with original data
        return X_c, [X_tilde_1, X_tilde_2, X_tilde_3]

    def forward(self, x, mask):
        batches, steps, nodes, features = x.size()

        # Whiten missing values
        x = x * mask 
        # handle nan if still present
        if torch.isnan(x).any():
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        # reshape
        X = rearrange(x, 'b t n f -> (b n) t f')
        masks = rearrange(mask, 'b t n f -> (b n) t f').to(torch.int)

        reconstruction_loss = 0
        imputed_data, [X_tilde_1, X_tilde_2, X_tilde_3] = self.impute(X, masks)

        # reconstruction_loss += masked_mae_cal(X_tilde_1, X, masks)
        # reconstruction_loss += masked_mae_cal(X_tilde_2, X, masks)
        # final_reconstruction_MAE = masked_mae_cal(X_tilde_3, X, masks)
        # reconstruction_loss += final_reconstruction_MAE
        # reconstruction_loss /= 3

        imputed_data = rearrange(imputed_data, '(b n) t f -> b t n f', n=nodes)     

        return {
            "model_name": "SAITS",
            "imputed_data": imputed_data,
            "first": X_tilde_1,
            "second": X_tilde_2,
            "third": X_tilde_3,
            # "reconstruction_loss": reconstruction_loss,
            # "imputation_loss": imputation_MAE,
            # "reconstruction_MAE": final_reconstruction_MAE,
            # "imputation_MAE": imputation_MAE,
        }
