import numpy as np
import torch

from settings import (
    COURT_LENGTH,
    COURT_WIDTH,
    GAMES_DIR,
    MATCHES_DIR,
    PITCH_LENGTH,
    PITCH_WIDTH,
)
from torch.utils.data import Dataset

START_ENDS = {
    "basketball": {"idxs": (10, 20), "xs": (20, 30), "ys": (30, 40), "sides": (40, 50)},
    "soccer": {"idxs": (5, 27), "xs": (27, 49), "ys": (49, 71), "sides": (71, 93)},
}


class BoformerDataset(Dataset):
    def __init__(
        self,
        hz,
        secs,
        N,
        player_traj_n,
        max_player_move,
        comp_ids,
        starts,
        mode,
        n_player_ids,
    ):
        self.hz = hz
        self.secs = secs
        self.N = N
        self.comp_ids = comp_ids
        self.starts = starts
        self.mode = mode
        assert mode in {"train", "valid", "test"}
        self.n_player_ids = n_player_ids
        self.seq_len = int(25 * secs) // int(25 / hz)

        self.player_traj_n = player_traj_n
        self.max_player_move = max_player_move
        self.player_traj_bins = np.linspace(
            -max_player_move, max_player_move, player_traj_n - 1
        )

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if self.mode == "train":
            comp_id = np.random.choice(self.comp_ids)
        else:
            comp_id = self.comp_ids[idx]

        sport = "basketball" if len(comp_id.split("_")) == 1 else "soccer"

        comp_dir = GAMES_DIR if sport == "basketball" else MATCHES_DIR
        X = np.load(f"{comp_dir}/{comp_id}_X.npy")

        raw_data_hz = 25
        if sport == "soccer":
            raw_data_hz = int(comp_id.split("_")[1])

        skip = int(raw_data_hz / self.hz)
        chunk_size = int(raw_data_hz * self.secs)

        if self.mode == "train":
            start = np.random.randint(len(X) - chunk_size)
        else:
            start = self.starts[idx]

        # Downsample.
        seq_data = X[start : start + chunk_size : skip]

        n_players = 10 if sport == "basketball" else 22
        keep_players = np.random.choice(np.arange(n_players), n_players, False)
        if self.mode in {"valid", "test"}:
            keep_players.sort()

        # End sequence early if there is a position glitch. Often happens when there was
        # a break in the game, but glitches also happen for other reasons. See
        # glitch_example.py for an example.
        (x_start, x_end) = START_ENDS[sport]["xs"]
        (y_start, y_end) = START_ENDS[sport]["ys"]
        player_xs = seq_data[:, x_start:x_end][:, keep_players]
        player_ys = seq_data[:, y_start:y_end][:, keep_players]
        player_x_diffs = np.diff(player_xs, axis=0)
        player_y_diffs = np.diff(player_ys, axis=0)
        if sport == "basketball":
            try:
                glitch_x_break = np.where(
                    np.abs(player_x_diffs) > 1.2 * self.max_player_move
                )[0].min()
            except ValueError:
                glitch_x_break = len(seq_data)

            try:
                glitch_y_break = np.where(
                    np.abs(player_y_diffs) > 1.2 * self.max_player_move
                )[0].min()
            except ValueError:
                glitch_y_break = len(seq_data)

        else:
            glitch_x_break = len(seq_data)
            glitch_y_break = len(seq_data)

        seq_break = min(glitch_x_break, glitch_y_break)
        seq_data = seq_data[:seq_break]

        (idx_start, idx_end) = START_ENDS[sport]["idxs"]
        (side_start, side_end) = START_ENDS[sport]["sides"]
        player_idxs = seq_data[:, idx_start:idx_end][:, keep_players].astype(int)
        player_xs = seq_data[:, x_start:x_end][:, keep_players]
        player_ys = seq_data[:, y_start:y_end][:, keep_players]
        player_sides = seq_data[:, side_start:side_end][:, keep_players].astype(int)
        player_masks = player_xs != 999

        # Randomly rotate the competition area because the direction is arbitrary.
        if (self.mode == "train") and (np.random.random() < 0.5):
            if sport == "basketball":
                (length, width) = (COURT_LENGTH, COURT_WIDTH)
            else:
                (length, width) = (PITCH_LENGTH, PITCH_WIDTH)

            player_xs = length - player_xs
            player_ys = width - player_ys
            player_sides = (player_sides + 1) % 2

        # Get player trajectories.
        player_x_diffs = np.diff(player_xs, axis=0)
        player_y_diffs = np.diff(player_ys, axis=0)

        player_traj_rows = np.digitize(player_y_diffs, self.player_traj_bins)
        player_traj_cols = np.digitize(player_x_diffs, self.player_traj_bins)
        player_trajs = player_traj_rows * self.player_traj_n + player_traj_cols

        return {
            "player_idxs": torch.LongTensor(player_idxs[: seq_break - 1]),
            "player_xs": torch.Tensor(player_xs[: seq_break - 1]),
            "player_ys": torch.Tensor(player_ys[: seq_break - 1]),
            "player_x_diffs": torch.Tensor(player_x_diffs),
            "player_y_diffs": torch.Tensor(player_y_diffs),
            "player_sides": torch.Tensor(player_sides[: seq_break - 1]),
            "player_trajs": torch.LongTensor(player_trajs),
            "player_masks": torch.BoolTensor(player_masks[: seq_break - 1]),
        }
