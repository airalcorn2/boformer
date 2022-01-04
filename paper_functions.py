import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

from boformer_dataset import START_ENDS
from settings import MATCHES_DIR


def soccer_traj_heat_map():
    hz = 5
    (x_start, x_end) = START_ENDS["soccer"]["xs"]
    (y_start, y_end) = START_ENDS["soccer"]["ys"]
    all_player_x_diffs = []
    all_player_y_diffs = []
    for match_f in os.listdir(MATCHES_DIR):
        X = np.load(f"{MATCHES_DIR}/{match_f}")

        raw_data_hz = int(match_f.split(".")[0].split("_")[1])
        skip = int(raw_data_hz / hz)
        skip_secs = 1 / skip

        seq_data = X[::skip]

        player_xs = seq_data[:, x_start:x_end]
        player_ys = seq_data[:, y_start:y_end]
        player_x_diffs = np.diff(player_xs, axis=0)
        player_y_diffs = np.diff(player_ys, axis=0)

        game_clock_diffs = np.diff(seq_data[:, 0] / 1000)[None].T

        not_missing = (player_xs[:-1] != 999) & (player_xs[1:] != 999)
        keep_diffs = (game_clock_diffs <= 1.2 * skip_secs) & not_missing
        all_player_x_diffs.append(player_x_diffs[keep_diffs].flatten())
        all_player_y_diffs.append(player_y_diffs[keep_diffs].flatten())

    all_player_x_diffs = np.concatenate(all_player_x_diffs)
    all_player_y_diffs = np.concatenate(all_player_y_diffs)

    max_player_move = 4.5 + 1
    player_traj_n = 11 + 2
    player_traj_bins = np.linspace(-max_player_move, max_player_move, player_traj_n - 1)
    (heatmap, xedges, yedges) = np.histogram2d(
        all_player_x_diffs, all_player_y_diffs, player_traj_bins
    )
    probs = (heatmap / heatmap.sum()).flatten()
    entropy = -(probs * np.log(probs)).sum()
    pp = np.exp(entropy)
    print(pp)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    heatmap += 1
    norm = matplotlib.colors.LogNorm(vmin=heatmap.min(), vmax=heatmap.max())

    home_dir = os.path.expanduser("~")
    os.makedirs(f"{home_dir}/test", exist_ok=True)
    plt.imshow(heatmap.T, extent=extent, origin="lower", norm=norm)
    plt.savefig(f"{home_dir}/test/player_traj_heat_map.png")
    plt.clf()
    shutil.make_archive(f"{home_dir}/test", "zip", f"{home_dir}/test")
    shutil.rmtree(f"{home_dir}/test")
