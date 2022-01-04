import math
import torch

from torch import nn


class Boformer(nn.Module):
    def __init__(
        self,
        n_player_ids,
        embedding_dim,
        sigmoid,
        seq_len,
        mlp_layers,
        n_player_labels,
        nhead,
        dim_feedforward,
        num_layers,
        sport_layers,
        dropout,
    ):
        super().__init__()
        self.sigmoid = sigmoid
        self.seq_len = seq_len
        self.max_players = 22
        self.embedding_dim = embedding_dim

        initrange = 0.1
        player_embeddings = {}
        start_mlps = {}
        pos_mlps = {}
        traj_mlps = {}
        missing_start_mlps = {}
        missing_pos_mlps = {}
        missing_traj_mlps = {}
        for sport in ["basketball", "soccer"]:
            player_embedding = nn.Embedding(n_player_ids[sport], embedding_dim)
            player_embedding.weight.data.uniform_(-initrange, initrange)
            player_embeddings[sport] = player_embedding

            start_mlp = nn.Sequential()
            pos_mlp = nn.Sequential()
            traj_mlp = nn.Sequential()
            missing_start_mlp = nn.Sequential()
            missing_pos_mlp = nn.Sequential()
            missing_traj_mlp = nn.Sequential()
            pos_in_feats = embedding_dim + 3
            traj_in_feats = embedding_dim + 5
            for (layer_idx, out_feats) in enumerate(mlp_layers):
                start_mlp.add_module(
                    f"layer{layer_idx}", nn.Linear(pos_in_feats, out_feats)
                )
                pos_mlp.add_module(
                    f"layer{layer_idx}", nn.Linear(pos_in_feats, out_feats)
                )
                traj_mlp.add_module(
                    f"layer{layer_idx}", nn.Linear(traj_in_feats, out_feats)
                )
                if layer_idx == 0:
                    pos_in_feats = embedding_dim + 1
                    traj_in_feats = embedding_dim + 1

                missing_start_mlp.add_module(
                    f"layer{layer_idx}", nn.Linear(pos_in_feats, out_feats)
                )
                missing_pos_mlp.add_module(
                    f"layer{layer_idx}", nn.Linear(pos_in_feats, out_feats)
                )
                missing_traj_mlp.add_module(
                    f"layer{layer_idx}", nn.Linear(traj_in_feats, out_feats)
                )
                if layer_idx < len(mlp_layers) - 1:
                    start_mlp.add_module(f"relu{layer_idx}", nn.ReLU())
                    pos_mlp.add_module(f"relu{layer_idx}", nn.ReLU())
                    traj_mlp.add_module(f"relu{layer_idx}", nn.ReLU())
                    missing_start_mlp.add_module(f"relu{layer_idx}", nn.ReLU())
                    missing_pos_mlp.add_module(f"relu{layer_idx}", nn.ReLU())
                    missing_traj_mlp.add_module(f"relu{layer_idx}", nn.ReLU())

                pos_in_feats = out_feats
                traj_in_feats = out_feats

            start_mlps[sport] = start_mlp
            pos_mlps[sport] = pos_mlp
            traj_mlps[sport] = traj_mlp
            missing_start_mlps[sport] = missing_start_mlp
            missing_pos_mlps[sport] = missing_pos_mlp
            missing_traj_mlps[sport] = missing_traj_mlp

        self.player_embeddings = nn.ModuleDict(player_embeddings)
        self.start_mlps = nn.ModuleDict(start_mlps)
        self.pos_mlps = nn.ModuleDict(pos_mlps)
        self.traj_mlps = nn.ModuleDict(traj_mlps)
        self.missing_start_mlps = nn.ModuleDict(missing_start_mlps)
        self.missing_pos_mlps = nn.ModuleDict(missing_pos_mlps)
        self.missing_traj_mlps = nn.ModuleDict(missing_traj_mlps)

        d_model = mlp_layers[-1]
        self.d_model = d_model

        self.dummy_pos = nn.Parameter(torch.randn(1, d_model))
        self.dummy_start = nn.Parameter(torch.randn(1, d_model))
        self.dummy_traj = nn.Parameter(torch.randn(1, d_model))

        self.register_buffer("mask", self.generate_mask())

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        sport_trannsformers = {}
        for sport in ["basketball", "soccer"]:
            sport_trannsformers[sport] = nn.TransformerEncoder(
                encoder_layer, sport_layers
            )

        self.sport_transformers = nn.ModuleDict(sport_trannsformers)
        self.joint_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers - sport_layers
        )

        self.traj_classifier = nn.Linear(d_model, n_player_labels)
        self.traj_classifier.weight.data.uniform_(-initrange, initrange)
        self.traj_classifier.bias.data.zero_()

    def generate_mask(self):
        tri_sz = 2 * (self.seq_len * self.max_players)
        sz = tri_sz + self.max_players
        mask = torch.zeros(sz, sz)
        mask[:tri_sz, :tri_sz] = torch.tril(torch.ones(tri_sz, tri_sz))
        mask[:, -self.max_players :] = 1
        mask = mask.masked_fill(mask == 0, float("-inf"))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tensors):
        device = list(self.traj_classifier.parameters())[0].device

        n_players = tensors["player_idxs"].shape[1]
        sport = "basketball" if n_players == 10 else "soccer"

        player_embeddings = self.player_embeddings[sport](
            tensors["player_idxs"].flatten().to(device)
        )
        if self.sigmoid == "logistic":
            player_embeddings = torch.sigmoid(player_embeddings)
        elif self.sigmoid == "tanh":
            player_embeddings = torch.tanh(player_embeddings)

        player_xs = tensors["player_xs"].flatten().unsqueeze(1).to(device)
        player_ys = tensors["player_ys"].flatten().unsqueeze(1).to(device)
        player_sides = tensors["player_sides"].flatten().unsqueeze(1).to(device)
        player_pos = torch.cat(
            [
                player_embeddings,
                player_sides,
                player_xs,
                player_ys,
            ],
            dim=1,
        )

        start_feats = self.start_mlps[sport](player_pos[:n_players]) * math.sqrt(
            self.d_model
        )
        player_masks = tensors["player_masks"].flatten().to(device)
        start_feats[:n_players][~player_masks[:n_players]] = self.missing_start_mlps[
            sport
        ](player_pos[:n_players][~player_masks[:n_players], : self.embedding_dim + 1])

        pos_feats = self.pos_mlps[sport](player_pos) * math.sqrt(self.d_model)
        pos_feats[~player_masks] = self.missing_pos_mlps[sport](
            player_pos[~player_masks, : self.embedding_dim + 1]
        )

        player_x_diffs = tensors["player_x_diffs"].flatten().unsqueeze(1).to(device)
        player_y_diffs = tensors["player_y_diffs"].flatten().unsqueeze(1).to(device)
        player_trajs = torch.cat(
            [
                player_embeddings,
                player_sides,
                player_xs + player_x_diffs,
                player_ys + player_y_diffs,
                player_x_diffs,
                player_y_diffs,
            ],
            dim=1,
        )

        trajs_feats = self.traj_mlps[sport](player_trajs) * math.sqrt(self.d_model)
        trajs_feats[~player_masks] = self.missing_traj_mlps[sport](
            player_trajs[~player_masks, : self.embedding_dim + 1]
        )
        missing_nexts = player_x_diffs.flatten().abs() > 500
        trajs_feats[missing_nexts] = self.missing_traj_mlps[sport](
            player_trajs[missing_nexts, : self.embedding_dim + 1]
        )

        (seq_len, max_players) = (self.seq_len, self.max_players)
        combined = torch.zeros(
            2 * seq_len * max_players + max_players, self.d_model
        ).to(device)
        if sport == "basketball":
            dummy_pos_feats = self.dummy_pos.repeat(seq_len * max_players, 1)
            dummy_start_feats = self.dummy_start.repeat(max_players, 1)
            dummy_trajs_feats = self.dummy_traj.repeat(seq_len * max_players, 1)

            dummy_start_feats[:n_players] = start_feats
            for time_step in range(seq_len):
                start = time_step * n_players
                end = start + n_players
                dummy_start = time_step * max_players
                dummy_end = dummy_start + n_players
                dummy_pos_feats[dummy_start:dummy_end] = pos_feats[start:end]
                dummy_trajs_feats[dummy_start:dummy_end] = trajs_feats[start:end]

            pos_feats = dummy_pos_feats
            start_feats = dummy_start_feats
            trajs_feats = dummy_trajs_feats

        combined[:-max_players:2] = pos_feats
        combined[1:-max_players:2] = trajs_feats
        combined[-max_players:] = start_feats

        outputs = self.sport_transformers[sport](combined.unsqueeze(1), self.mask)
        outputs = self.joint_transformer(outputs, self.mask)
        outputs = outputs.squeeze(1)[:-max_players:2]
        if sport == "basketball":
            preds = torch.zeros(self.seq_len * n_players, self.d_model).to(device)
            for time_step in range(seq_len):
                start = time_step * max_players
                end = start + n_players
                preds_start = time_step * n_players
                preds_end = preds_start + n_players
                preds[preds_start:preds_end] = outputs[start:end]

        else:
            preds = outputs

        preds = self.traj_classifier(preds)

        return preds
