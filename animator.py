import matplotlib.pyplot as plt
import numpy as np
import pickle

from matplotlib import animation
from settings import *


class Match:
    def __init__(self, data_dir, games_dir, match_id):
        soccer_config = pickle.load(open(f"{data_dir}/soccer_config.pydict", "rb"))
        self.player_idx2props = soccer_config["player_idx2props"]

        X = np.load(f"{games_dir}/{match_id}_X.npy")

        self.periods = X[:, 1].astype(int)
        self.period_times = X[:, 0] / 1000

        self.player_idxs = X[:, 5:27].astype(int)

        self.ball_xs = X[:, 2]
        self.ball_ys = X[:, 3]
        self.ball_zs = X[:, 4]

        self.player_xs = X[:, 27:49]
        self.player_ys = X[:, 49:71]
        self.player_goal_sides = X[:, 71:93]

    def update_radius(
        self,
        i,
        start,
        player_circles,
        ball_circle,
        annotations,
        clock_info,
        player_idx2circle_idx,
    ):
        time_step = start + i
        for (idx, player_idx) in enumerate(self.player_idxs[time_step]):
            circle_idx = player_idx2circle_idx[player_idx]
            player_circles[circle_idx].center = (
                self.player_xs[time_step, idx],
                Y_MAX - self.player_ys[time_step, idx],
            )
            annotations[circle_idx].set_position(player_circles[circle_idx].center)
            name = self.player_idx2props[player_idx]["name"].split()
            initials = name[0][0] + name[-1][0]
            annotations[circle_idx].set_text(initials)
            if self.player_goal_sides[time_step, idx]:
                player_circles[circle_idx].set_facecolor("white")
                player_circles[circle_idx].set_edgecolor("white")
            else:
                player_circles[circle_idx].set_facecolor("gray")
                player_circles[circle_idx].set_edgecolor("gray")

        clock_str = f"{self.periods[time_step]}/"
        period_time = int(self.period_times[time_step])
        clock_str += f"{period_time // 60:02}:{period_time % 60:02}"
        clock_info.set_text(clock_str)

        ball_circle.center = (self.ball_xs[time_step], Y_MAX - self.ball_ys[time_step])
        # ball_circle.radius = self.ball_zs[time_step] / NORMALIZATION_COEF
        ball_circle.radius = 12 / 7

        return (player_circles, ball_circle)

    def show_seq(
        self, start_period, start_time, stop_period, stop_time, save_gif=False
    ):
        ax = plt.axes(xlim=(X_MIN, X_MAX), ylim=(Y_MIN, Y_MAX))
        ax.axis("off")
        fig = plt.gcf()
        # Remove grid.
        ax.grid(False)

        clock_info = ax.annotate(
            "",
            xy=[X_CENTER, Y_CENTER],
            color="black",
            horizontalalignment="center",
            verticalalignment="center",
        )

        (start_min, start_sec) = start_time.split(":")
        (start_min, start_sec) = (int(start_min), int(start_sec))
        period_time = 60 * start_min + start_sec
        start = np.argwhere(
            (self.period_times > period_time) & (self.periods == start_period)
        ).min()

        (stop_min, stop_sec) = stop_time.split(":")
        (stop_min, stop_sec) = (int(stop_min), int(stop_sec))
        period_time = 60 * stop_min + stop_sec
        stop = np.argwhere(
            (self.period_times > period_time) & (self.periods == stop_period)
        ).min()

        player_idxs = set(self.player_idxs[start])
        for time_step in range(start + 1, stop):
            # End sequence early at lineup change.
            if len(player_idxs & set(self.player_idxs[time_step])) != 22:
                stop = time_step
                break

        annotations = []
        player_idx2circle_idx = {}
        team_a_players = []
        team_b_players = []
        player_circles = []
        for (circle_idx, player_idx) in enumerate(self.player_idxs[start]):
            player_idx2circle_idx[player_idx] = circle_idx
            name = self.player_idx2props[player_idx]["name"].split()
            initials = name[0][0] + name[-1][0]
            annotations.append(
                ax.annotate(
                    initials,
                    xy=[0, 0],
                    color="black",
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontweight="bold",
                )
            )
            if self.player_goal_sides[start, circle_idx]:
                team_a_players.append(
                    self.player_idx2props[player_idx]["name"] + f": {initials}"
                )
                player_circles.append(
                    plt.Circle((0, 0), PLAYER_CIRCLE_SIZE, color="white")
                )
            else:
                team_b_players.append(
                    self.player_idx2props[player_idx]["name"] + f": {initials}"
                )
                player_circles.append(
                    plt.Circle((0, 0), PLAYER_CIRCLE_SIZE, color="gray")
                )

        # Prepare table.
        # column_labels = tuple(["Team A", "Team B"])
        # players_data = list(zip(team_a_players, team_b_players))
        #
        # table = plt.table(
        #     cellText=players_data,
        #     colLabels=column_labels,
        #     colWidths=[COL_WIDTH, COL_WIDTH],
        #     loc="bottom",
        #     fontsize=FONTSIZE,
        #     cellLoc="center",
        # )
        # table.scale(1, SCALE)

        # Add animated objects.
        for circle in player_circles:
            ax.add_patch(circle)

        ball_circle = plt.Circle((0, 0), PLAYER_CIRCLE_SIZE, color=BALL_COLOR)
        ax.add_patch(ball_circle)

        anim = animation.FuncAnimation(
            fig,
            self.update_radius,
            fargs=(
                start,
                player_circles,
                ball_circle,
                annotations,
                clock_info,
                player_idx2circle_idx,
            ),
            frames=stop - start,
            interval=INTERVAL,
        )
        pitch = plt.imread("pitch.png")
        plt.imshow(pitch, zorder=0, extent=[X_MIN, X_MAX, Y_MAX, Y_MIN])
        if save_gif:
            anim.save("animation.mp4", writer="imagemagick", fps=25)

        plt.show()
