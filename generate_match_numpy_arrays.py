import json
import multiprocessing
import numpy as np
import pandas as pd
import pickle
import shutil
import unidecode

from settings import *

HALF_COURT_LENGTH = COURT_LENGTH // 2
THRESHOLD = 1.0

MATCH2SAME_LAST_NAMES = {
    "a2zaylcacc8460vihjy4ebzf8": {"balikwisha"},
    "8at0ll1fsfe20otu6wwhl6a96": {"balikwisha"},
    "7ovr1cz4h3fu0zr9zrxus8bxm": {"hubert"},
    "86pfq1y8zupv3b380ay5ymzca": {"hubert"},
    "8owyooimjx6ib6oler6z0e2y2": {"ito"},
    "879nmex2z03sioxmbnj54w6sq": {"gueye"},
    "8cgnx5orliiin2rt9ipzg9v8q": {"gueye"},
    "7zkn1b5z0onzs6ts6n25niaui": {"gueye"},
    "85e1f1vq49whp54r1d2wtg3fu": {"suzuki"},
    "84a6spv03gjeik5fuitw9o0oa": {"seck"},
    "7z5o2e02xzco7wmww1ydf3f56": {"kayembe"},
    "8qlldkdfxcojvmx2hlsuvxiei": {"kayembe"},
}
REPLACEMENT_NAMES = {
    "lourenco": "mata",
    "hongla": "yma",
    "mananga": "buatu",
    "montes": "castro-montes",
    "benson": "manuel",
    "pedersen": "maehle",
    "bonilla": "lucumi",
    "tshiend": "nkuba",
    "felix-eppiah": "eppiah",
    "ekango": "limbombe",
    "yao": "raux-yao",
    "nwambeben": "mmaee",
    "lee": "seung-woo",
    "bermudez": "murillo",
    "dovillabichus": "avenatti",
    "vha": "balongo",
    "kisonga": "bushiri",
    "bezua": "mbokani",
    "mbenza": "kamboleke",
    "humphreys": "humphreys-grant",
    "batombo": "bongonda",
    "dhli": "daehli",
    "malangu": "malungu",
    "mbamba": "mbamba-muanda",
}
MATCH_REPLACEMENT_NAMES = {
    "sowah": {
        "match_ids": {"7ka0j66rt1jlyy84tjaoc8tbe", "874la9kk0yncyuo4mjlvzlysq"},
        "replacement": "adjei",
    },
    "costa": {
        "match_ids": {
            "8660v6hiro4bkmx989gd1ek4q",
            "7xqxtprfn65uws9tz6jfga4q2",
            "7i0cnknsrjqv1q97enqd7sqy2",
            "7jw9s2pa5b1o0ba63lpjcl4kq",
            "7monyrre59ix2hvp55ytdzqlm",
            "8kudrsc8e2k7wn1sf80aamaui",
        },
        "replacement": "ribeiro",
    },
    "schrijvers": {
        "match_ids": {
            "7myyrf3ponm5rzrmm7if967vu",
            "7iz6mawg2xj2u9m0mccdloa5m",
            "7kjt8ls8q34cqcuh4fp745ze2",
            "7l78l3r4g23n2h5vuthr8xm5m",
        },
        "replacement": "schryvers",
    },
    "garcia": {
        "match_ids": {"7p0dvgbfguoy02dq7gvarwnl6", "7nhqaf9wfnhkr8k0zly76ck16"},
        "replacement": "govea",
    },
    "shamshudin": {
        "match_ids": {"7u1fvun06x96w9pgbi5nezfca"},
        "replacement": "shamsudin",
    },
}
WEIRD_MATCHES = {
    "7ovr1cz4h3fu0zr9zrxus8bxm",
    "7wui5bwfeusaz2y7t2ruq860a",
    "7wzgofamy7y2cki9eexphr42y",
    "8bn6s9jmnd7cp4h6le0xwrtka",
}
MATCH2HOME_GOALKEEPER = {
    "7nwj6kc2frfctmjx5ol9uly7e": "Vanhamel",
    "7ovr1cz4h3fu0zr9zrxus8bxm": "Hubert",
    "7peq1nddb11m7a74mwys4i92i": "Jackers",
    "7qheim25berh9ij7e4lxwz3vu": "Jakubech",
}


def get_missing_tracking_ids():
    df = pd.read_csv(f"{SOCCER_DATA_DIR}/2021-08-19-jpl-season-2020-2021-squads.csv")
    stats_ids = set(df["stats_id"])

    player_ids = set()
    for filename in os.listdir(f"{SOCCER_DATA_DIR}/data"):
        if "tracking-metadata" in filename:
            with open(f"{SOCCER_DATA_DIR}/data/{filename}") as f:
                for line in f:
                    parts = line.strip().split(",")
                    player_ids.add(int(parts[9]))

    print(player_ids - stats_ids)
    print(len(player_ids - stats_ids))


def get_same_last_name_matches():
    same_last_name2matches = {}
    for filename in os.listdir(f"{SOCCER_DATA_DIR}/data"):
        if "tracking-metadata" in filename:
            match_id = filename.split(".")[0].split("-")[-1]
            with open(f"{SOCCER_DATA_DIR}/data/{filename}") as f:
                last_names = set()
                for line in f:
                    parts = line.strip().split(",")
                    last_name = parts[3].split(" ")[-1].lower()
                    if last_name in last_names:
                        same_last_name2matches.setdefault(last_name, []).append(
                            match_id
                        )

                    last_names.add(last_name)

    match2same_last_names = {}
    for (last_name, matches) in same_last_name2matches.items():
        for match in matches:
            # tracking-metadata-7q1u4zertpe7lz4j35abr7yp6.csv has the goalies listed
            # among the main roster. All of the other matches have multiple players with
            # the same last name.
            if match == "7q1u4zertpe7lz4j35abr7yp6":
                continue

            match2same_last_names.setdefault(match, set()).add(last_name)


def get_metadata_name_info(match_id, player_tracking_id2props):
    id_names = set()
    name2player_tracking_id = {}
    with open(f"{SOCCER_DATA_DIR}/data/tracking-metadata-{match_id}.csv") as f:
        for line in f:
            parts = line.strip().split(",")

            id_name = unidecode.unidecode(parts[3].split(" ")[-1].lower())
            id_name = id_name.replace("'", "")
            id_name = id_name.replace("?", "")

            # Process last names that vary between the metadata and event stream
            # files.
            id_name = REPLACEMENT_NAMES.get(id_name, id_name)
            if (id_name in MATCH_REPLACEMENT_NAMES) and (
                match_id in MATCH_REPLACEMENT_NAMES[id_name]["match_ids"]
            ):
                id_name = MATCH_REPLACEMENT_NAMES[id_name]["replacement"]

            if (match_id in MATCH2SAME_LAST_NAMES) and (
                id_name in MATCH2SAME_LAST_NAMES[match_id]
            ):
                id_name = f"{parts[2][0].lower()}. {id_name}"
                id_name = "h. gueye" if id_name == "p. gueye" else id_name
                id_name = "a. gueye" if id_name == "e. gueye" else id_name

            # Incorrect roster for this match.
            if (match_id == "7wkte145a8aosm12inv1otpii") and (id_name == "ritiere"):
                id_name = "bossche"
                player_tracking_id = "1229633"
            else:
                player_tracking_id = parts[9]

            if f"{parts[2]} {parts[3]}" == "Moussa Al-Taamari":
                # This file has the wrong tracking ID for some reason.
                if match_id == "7yvdmfhf2ic08xf6g1n1iqkx6":
                    player_tracking_id = "1129652"

            id_names.add(id_name)
            # stats_id in squads CSV.
            name2player_tracking_id[id_name] = player_tracking_id

            name = f"{parts[2]} {parts[3]}"
            player_tracking_id2props[player_tracking_id] = {"name": name}

    return (id_names, name2player_tracking_id)


def get_event_names(match_id, name2player_tracking_id, player_tracking_id2props):
    event_names = set()
    name_counts = {}
    with open(f"{SOCCER_DATA_DIR}/data/events-ma3-{match_id}.json") as f:
        events = json.load(f)["liveData"]["event"]
        for event in events:
            if "playerId" in event:
                playerName = unidecode.unidecode(event["playerName"].lower())
                id_name = playerName.split()[-1]
                id_name = id_name.replace("'", "")
                id_name = id_name.replace("?", "")
                if (match_id in MATCH2SAME_LAST_NAMES) and (
                    id_name in MATCH2SAME_LAST_NAMES[match_id]
                ):
                    id_name = playerName

                event_names.add(id_name)
                name_counts[id_name] = name_counts.get(id_name, 0) + 1
                if id_name not in name2player_tracking_id:
                    continue

                statsperform_uuid = event["playerId"]
                player_tracking_id = name2player_tracking_id[id_name]
                if (
                    "statsperform_uuid" in player_tracking_id2props[player_tracking_id]
                ) and (
                    player_tracking_id2props[player_tracking_id]["statsperform_uuid"]
                    != statsperform_uuid
                ):
                    raise ValueError

                player_tracking_id2props[player_tracking_id][
                    "statsperform_uuid"
                ] = statsperform_uuid

    return (event_names, name_counts)


def get_playerid2player_idx_map():
    tracking_metadata_filenames = [
        filename
        for filename in os.listdir(f"{SOCCER_DATA_DIR}/data")
        if "tracking-metadata" in filename
    ]
    tracking_metadata_filenames.sort()

    player_tracking_id2props = {}
    not_perfect = 0
    for (idx, filename) in enumerate(tracking_metadata_filenames):
        match_id = filename.split(".")[0].split("-")[-1]
        (id_names, name2player_tracking_id) = get_metadata_name_info(
            match_id, player_tracking_id2props
        )
        (event_names, name_counts) = get_event_names(
            match_id, name2player_tracking_id, player_tracking_id2props
        )

        try:
            assert len(event_names - id_names) == 0
        except AssertionError:
            print(filename)
            print(event_names - id_names)
            print(id_names - event_names)
            not_perfect += 1
            if len(id_names - event_names) == 0:
                for name in event_names - id_names:
                    # These are players who show up in the event stream once, but are
                    # not listed on either of the rosters.
                    if name_counts[name] > 1:
                        # There are some exceptions where the non-rostered player shows
                        # up more than once.
                        if match_id not in WEIRD_MATCHES:
                            raise ValueError
            else:
                raise ValueError

    playerid2player_idx = {}
    player_idx2props = {}
    playerids = list(player_tracking_id2props)
    playerids.sort()
    for (player_idx, playerid) in enumerate(playerids):
        playerid2player_idx[playerid] = player_idx
        player_idx2props[player_idx] = player_tracking_id2props[playerid]
        player_idx2props[player_idx]["playerid"] = playerid

    return (playerid2player_idx, player_idx2props)


def find_files_without_goalies():
    tracking_metadata_filenames = [
        filename
        for filename in os.listdir(f"{SOCCER_DATA_DIR}/data")
        if "tracking-metadata" in filename
    ]
    have_goalies = set()
    for filename in tracking_metadata_filenames:
        with open(f"{SOCCER_DATA_DIR}/data/{filename}") as f:
            for line in f:
                parts = line.split(",")
                if parts[1] == "3":
                    have_goalies.add(filename)

    missing_goalies = set(tracking_metadata_filenames) - have_goalies
    print(len(missing_goalies))
    print(missing_goalies)


def get_home_goal_sides_worker(tracking_fs, queue):
    home_goal_sides = {}
    for tracking_f in tracking_fs:

        match_id = tracking_f.split("-")[2]

        home_goalie_ids = set()
        try:
            with open(f"{SOCCER_DATA_DIR}/data/tracking-metadata-{match_id}.csv") as f:
                for line in f:
                    parts = line.split(",")
                    if match_id in MATCH2HOME_GOALKEEPER:
                        goalie = MATCH2HOME_GOALKEEPER[match_id]
                        if goalie == parts[3]:
                            home_goalie_ids.add(parts[9])

                    else:
                        if parts[1] == "3":
                            home_goalie_ids.add(parts[9])

        except FileNotFoundError:
            print(f"{match_id} has no metadata file.", flush=True)
            continue

        assert home_goalie_ids

        match_home_goal_side_counts = {}
        tracking_periods = set()

        with open(f"{SOCCER_DATA_DIR}/data/{tracking_f}") as f:
            for line in f:
                try:
                    (info, players, ball) = line.split(":")
                except ValueError:
                    continue

                period = int(info.split(";")[1].split(",")[1])
                tracking_periods.add(period)

                if period not in match_home_goal_side_counts:
                    match_home_goal_side_counts[period] = {"left": 0, "right": 0}

                for player_info in players.split(";")[:-1]:
                    (_, player_tracking_id, _, x, _) = player_info.split(",")
                    if player_tracking_id in home_goalie_ids:
                        home_goalie_x = float(x)

                try:
                    if home_goalie_x < 50:
                        match_home_goal_side_counts[period]["left"] += 1

                    else:
                        match_home_goal_side_counts[period]["right"] += 1

                except UnboundLocalError:
                    print(f"No goalie in {match_id}.", flush=True)
                    raise UnboundLocalError

        match_home_goal_sides = {}
        for (period, side_counts) in match_home_goal_side_counts.items():
            (max_side, max_count) = (None, 0)
            for (side, count) in side_counts.items():
                if count > max_count:
                    (max_side, max_count) = (side, count)

            match_home_goal_sides[period] = max_side

        for period_a in tracking_periods:
            if period_a <= 2:
                period_b = 2 - period_a + 1
            else:
                period_b = 4 - period_a + 3

            if (period_a in match_home_goal_sides) and (
                period_b not in match_home_goal_sides
            ):
                side_a = match_home_goal_sides[period_a]
                side_b = "right" if side_a == "left" else "left"
                match_home_goal_sides[period_b] = side_b

        for tracking_period in tracking_periods:
            try:
                assert tracking_period in match_home_goal_sides
            except AssertionError:
                print(f"{match_id}: {match_home_goal_side_counts}", flush=True)
                break

        home_goal_sides[match_id] = match_home_goal_sides

    queue.put(home_goal_sides)


def get_home_goal_sides():
    q = multiprocessing.Queue()

    df = pd.read_csv(f"{SOCCER_DATA_DIR}/2021-08-19-jpl-season-2020-2021-overview.csv")
    all_tracking_fs = []
    for (_, row) in df.iterrows():
        match_id = row["statsperform_uuid"]
        if row["tracking_25fps"]:
            all_tracking_fs.append(f"tracking-data-{match_id}-25fps.txt")
        elif row["tracking_10fps"]:
            all_tracking_fs.append(f"tracking-data-{match_id}-10fps.txt")

    all_tracking_fs.sort()
    processes = multiprocessing.cpu_count()
    tracking_fs_per_process = int(np.ceil(len(all_tracking_fs) / processes))
    jobs = []
    for i in range(processes):
        start = i * tracking_fs_per_process
        end = start + tracking_fs_per_process
        tracking_fs = all_tracking_fs[start:end]
        p = multiprocessing.Process(
            target=get_home_goal_sides_worker, args=(tracking_fs, q)
        )
        jobs.append(p)
        p.start()

    home_goal_sides = {}
    for _ in jobs:
        home_goal_sides.update(q.get())

    for p in jobs:
        p.join()

    print("Matches without ball data.\n")
    for (match_id, match_home_goal_sides) in home_goal_sides.items():
        if len(match_home_goal_sides) == 0:
            print(match_id, flush=True)

    return home_goal_sides


def save_match_numpy_array(tracking_f):
    # tracking_f = "tracking-data-7wzgofamy7y2cki9eexphr42y-25fps.txt"
    X = []
    total_lines = 0
    missing_ball_lines = 0
    missing_players_lines = 0
    too_many_players_lines = 0
    match_id = tracking_f.split("-")[2]
    prev_teams = {"home": set(), "away": set()}
    with open(f"{SOCCER_DATA_DIR}/data/{tracking_f}") as f:
        for line in f:
            total_lines += 1
            try:
                (info, players, ball) = line.split(":")
            except ValueError:
                missing_ball_lines += 1
                continue

            if (len(players.split(";")[:-1]) != 22) and (not prev_teams["home"]):
                missing_players_lines += 1
                continue

            (_, period_info) = info.split(";")
            (period_time, period, in_play) = period_info.split(",")

            if in_play == "1":
                continue

            period = int(period)
            (ball_x, ball_y, ball_z) = ball.split(";")[0].split(",")
            data = [
                float(period_time),
                period,
                float(ball_x),
                float(ball_y),
                float(ball_z),
            ]

            player_idxs = {"home": [], "away": []}
            player_xs = {"home": [], "away": []}
            player_ys = {"home": [], "away": []}
            player_goal_sides = {"home": [], "away": []}

            home_goal_side_left = home_goal_sides[match_id][period] == "left"
            current_teams = {"home": set(), "away": set()}
            for player in players.split(";")[:-1]:
                (team_id, playerid, _, player_x, player_y) = player.split(",")

                # Handle Moussa Al-Taamari.
                playerid = "1129652" if playerid == "1194894" else playerid
                if team_id in {"0", "1", "3", "4"}:
                    if team_id in {"0", "3"}:
                        team = "home"
                        goal_side = int(home_goal_side_left)
                    elif team_id in {"1", "4"}:
                        team = "away"
                        goal_side = int(not home_goal_side_left)

                    try:
                        player_idx = playerid2player_idx[playerid]

                    except KeyError:
                        raise KeyError(f"Bad player_id ({playerid}) for {match_id}.")

                    player_idxs[team].append(player_idx)
                    player_xs[team].append(float(player_x))
                    player_ys[team].append(float(player_y))
                    player_goal_sides[team].append(goal_side)
                    current_teams[team].add(player_idx)

                else:
                    continue

            # Handle lines that are missing players from one or both teams.
            for (team, team_player_idxs) in current_teams.items():
                if len(team_player_idxs) < 11:
                    missing_players_lines += 1
                    for player_idx in prev_teams[team] - team_player_idxs:
                        if len(team_player_idxs) == 11:
                            continue

                        if team == "home":
                            goal_side = int(home_goal_side_left)

                        else:
                            goal_side = int(not home_goal_side_left)

                        player_idxs[team].append(player_idx)
                        player_xs[team].append(999)
                        player_ys[team].append(999)
                        player_goal_sides[team].append(goal_side)
                        team_player_idxs.add(player_idx)

            # Handle lines that have too many players for one or both teams.
            for (team, team_player_idxs) in current_teams.items():
                if len(team_player_idxs) > 11:
                    too_many_players_lines += 1
                    exclude_players = list(
                        set(team_player_idxs) - set(prev_teams[team])
                    )
                    exclude_players.sort()
                    n_exclude = len(team_player_idxs) - 11
                    exclude_players = set(exclude_players[:n_exclude])
                    new_player_idxs = []
                    new_player_xs = []
                    new_player_ys = []
                    new_player_goal_sides = []
                    new_team_player_idxs = set()
                    for (idx, player_idx) in enumerate(player_idxs[team]):
                        if len(new_team_player_idxs) == 11:
                            continue

                        if player_idx in exclude_players:
                            continue

                        new_player_idxs.append(player_idx)
                        new_player_xs.append(player_xs[team][idx])
                        new_player_ys.append(player_ys[team][idx])
                        new_player_goal_sides.append(player_goal_sides[team][idx])
                        new_team_player_idxs.add(player_idx)

                    player_idxs[team] = new_player_idxs
                    player_xs[team] = new_player_xs
                    player_ys[team] = new_player_ys
                    player_goal_sides[team] = new_player_goal_sides
                    current_teams[team] = new_team_player_idxs

            try:
                assert len(current_teams["home"]) == 11
                assert len(current_teams["away"]) == 11
            except AssertionError:
                # continue
                raise AssertionError(f"{match_id} has the wrong number of players.")

            player_idxs = player_idxs["home"] + player_idxs["away"]
            order = np.argsort(player_idxs)
            for idx in order:
                data.append(player_idxs[idx])

            player_xs = player_xs["home"] + player_xs["away"]
            for idx in order:
                data.append(player_xs[idx])

            player_ys = player_ys["home"] + player_ys["away"]
            for idx in order:
                data.append(player_ys[idx])

            player_goal_sides = player_goal_sides["home"] + player_goal_sides["away"]
            for idx in order:
                data.append(player_goal_sides[idx])

            assert len(data) == 93

            X.append(np.array(data))
            prev_teams = current_teams

    print(f"{match_id} missing ball: {100 * missing_ball_lines / total_lines:.4f}%")
    print(
        f"{match_id} missing players: {100 * missing_players_lines / total_lines:.4f}%",
        flush=True,
    )
    print(
        f"{match_id} too many players: {100 * too_many_players_lines / total_lines:.4f}%",
        flush=True,
    )
    if len(X) > 0:
        X = np.stack(X)
        fps = tracking_f.split(".")[0].split("-")[-1][:2]
        np.save(f"{MATCHES_DIR}/{match_id}_{fps}_X.npy", X)


def save_numpy_arrays_worker(tracking_fs):
    for tracking_f in tracking_fs:
        save_match_numpy_array(tracking_f)


def save_numpy_arrays():
    df = pd.read_csv(f"{SOCCER_DATA_DIR}/2021-08-19-jpl-season-2020-2021-overview.csv")
    all_tracking_fs = []
    for (_, row) in df.iterrows():
        match_id = row["statsperform_uuid"]
        if match_id not in home_goal_sides:
            continue

        if row["tracking_25fps"]:
            all_tracking_fs.append(f"tracking-data-{match_id}-25fps.txt")
        elif row["tracking_10fps"]:
            all_tracking_fs.append(f"tracking-data-{match_id}-10fps.txt")

    all_tracking_fs.sort()
    processes = multiprocessing.cpu_count()
    tracking_fs_per_process = int(np.ceil(len(all_tracking_fs) / processes))
    jobs = []
    for i in range(processes):
        start = i * tracking_fs_per_process
        end = start + tracking_fs_per_process
        tracking_fs = all_tracking_fs[start:end]
        p = multiprocessing.Process(
            target=save_numpy_arrays_worker, args=(tracking_fs,)
        )
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()


if __name__ == "__main__":
    os.makedirs(MATCHES_DIR, exist_ok=True)

    (playerid2player_idx, player_idx2props) = get_playerid2player_idx_map()
    home_goal_sides = get_home_goal_sides()
    save_numpy_arrays()

    soccer_config = {
        "player_idx2props": player_idx2props,
    }
    pickle.dump(soccer_config, open(f"{DATA_DIR}/soccer_config.pydict", "wb"))
