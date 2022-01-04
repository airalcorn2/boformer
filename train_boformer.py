import numpy as np
import pickle
import sys
import time
import torch
import yaml

from boformer import Boformer
from boformer_dataset import BoformerDataset
from settings import *
from torch import nn, optim
from torch.utils.data import DataLoader

SEED = 2010
torch.manual_seed(SEED)
torch.set_printoptions(linewidth=160)
np.random.seed(SEED)


def worker_init_fn(worker_id):
    # See: https://pytorch.org/docs/stable/notes/faq.html#my-data-loader-workers-return-identical-random-numbers
    # and: https://pytorch.org/docs/stable/data.html#multi-process-data-loading
    # and: https://pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading.
    # NumPy seed takes a 32-bit unsigned integer.
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2 ** 32 - 1))


def get_train_valid_test_gameids():
    with open("train_gameids.txt") as f:
        train_gameids = f.read().split()

    with open("valid_gameids.txt") as f:
        valid_gameids = f.read().split()

    with open("test_gameids.txt") as f:
        test_gameids = f.read().split()

    return (train_gameids, valid_gameids, test_gameids)


def get_train_valid_test_match_ids(opts):
    try:
        with open("train_match_ids.txt") as f:
            train_match_ids = f.read().split()

        with open("valid_match_ids.txt") as f:
            valid_match_ids = f.read().split()

        with open("test_match_ids.txt") as f:
            test_match_ids = f.read().split()

    except FileNotFoundError:
        print("No {train/valid/test}_gameids.txt files found. Generating new ones.")

        match_ids = list(set([np_f.split("_X")[0] for np_f in os.listdir(MATCHES_DIR)]))
        match_ids.sort()
        np.random.seed(SEED)
        np.random.shuffle(match_ids)
        n_train_valid = int(opts["train"]["train_valid_prop"] * len(match_ids))
        n_train = int(opts["train"]["train_prop"] * n_train_valid)
        train_valid_match_ids = match_ids[:n_train_valid]

        train_match_ids = train_valid_match_ids[:n_train]
        valid_match_ids = train_valid_match_ids[n_train:]
        test_match_ids = match_ids[n_train_valid:]
        train_valid_test_match_ids = {
            "train": train_match_ids,
            "valid": valid_match_ids,
            "test": test_match_ids,
        }
        for (train_valid_test, match_ids) in train_valid_test_match_ids.items():
            with open(f"{train_valid_test}_match_ids.txt", "w") as f:
                for match_id in match_ids:
                    f.write(f"{match_id}\n")

    np.random.seed(SEED)

    return (train_match_ids, valid_match_ids, test_match_ids)


def init_datasets(opts):
    baller2vec_config = pickle.load(open(f"{DATA_DIR}/baller2vec_config.pydict", "rb"))
    n_player_ids = {"basketball": len(baller2vec_config["player_idx2props"])}
    soccer_config = pickle.load(open(f"{DATA_DIR}/soccer_config.pydict", "rb"))
    n_player_ids["soccer"] = len(soccer_config["player_idx2props"])

    if opts["train"]["use_basketball"]:
        (train_gameids, valid_gameids, test_gameids) = get_train_valid_test_gameids()
    else:
        (train_gameids, valid_gameids, test_gameids) = ([], [], [])

    if opts["train"]["use_soccer"]:
        (
            train_match_ids,
            valid_match_ids,
            test_match_ids,
        ) = get_train_valid_test_match_ids(opts)
    else:
        (train_match_ids, valid_match_ids, test_match_ids) = ([], [], [])

    train_comp_ids = train_gameids + train_match_ids
    valid_comp_ids = valid_gameids + valid_match_ids
    test_comp_ids = test_gameids + test_match_ids

    dataset_config = opts["dataset"]
    dataset_config["comp_ids"] = train_comp_ids
    dataset_config["N"] = opts["train"]["train_samples_per_epoch"]
    dataset_config["starts"] = []
    dataset_config["mode"] = "train"
    dataset_config["n_player_ids"] = n_player_ids
    train_dataset = BoformerDataset(**dataset_config)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=None,
        num_workers=opts["train"]["workers"],
        worker_init_fn=worker_init_fn,
    )

    samps_per_valid_id = opts["train"]["samps_per_valid_id"]
    starts = []
    for comp_id in valid_comp_ids:
        sport = "basketball" if len(comp_id.split("_")) == 1 else "soccer"
        comp_dir = GAMES_DIR if sport == "basketball" else MATCHES_DIR
        X = np.load(f"{comp_dir}/{comp_id}_X.npy")
        raw_data_hz = 25
        if sport == "soccer":
            raw_data_hz = int(comp_id.split("_")[1])

        chunk_size = int(raw_data_hz * train_dataset.secs)
        max_start = len(X) - chunk_size
        gaps = max_start // samps_per_valid_id
        starts.append(gaps * np.arange(samps_per_valid_id))

    dataset_config["comp_ids"] = np.repeat(valid_comp_ids, samps_per_valid_id)
    dataset_config["N"] = len(dataset_config["comp_ids"])
    dataset_config["starts"] = np.concatenate(starts)
    dataset_config["mode"] = "valid"
    valid_dataset = BoformerDataset(**dataset_config)
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=None,
        num_workers=opts["train"]["workers"],
    )

    samps_per_test_id = opts["train"]["samps_per_test_id"]
    starts = []
    for comp_id in test_comp_ids:
        sport = "basketball" if len(comp_id.split("_")) == 1 else "soccer"
        comp_dir = GAMES_DIR if sport == "basketball" else MATCHES_DIR
        X = np.load(f"{comp_dir}/{comp_id}_X.npy")
        raw_data_hz = 25
        if sport == "soccer":
            raw_data_hz = int(comp_id.split("_")[1])

        chunk_size = int(raw_data_hz * train_dataset.secs)
        max_start = len(X) - chunk_size
        gaps = max_start // samps_per_test_id
        starts.append(gaps * np.arange(samps_per_test_id))

    dataset_config["comp_ids"] = np.repeat(test_comp_ids, samps_per_test_id)
    dataset_config["N"] = len(dataset_config["comp_ids"])
    dataset_config["starts"] = np.concatenate(starts)
    dataset_config["mode"] = "test"
    test_dataset = BoformerDataset(**dataset_config)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=None,
        num_workers=opts["train"]["workers"],
    )

    return (
        train_dataset,
        train_loader,
        valid_dataset,
        valid_loader,
        test_dataset,
        test_loader,
    )


def init_model(opts, train_dataset):
    model_config = opts["model"]
    model_config["n_player_ids"] = train_dataset.n_player_ids
    model_config["seq_len"] = train_dataset.seq_len - 1
    model_config["n_player_labels"] = train_dataset.player_traj_n ** 2
    model = Boformer(**model_config)

    return model


def get_preds_labels(tensors):
    player_masks = tensors["player_masks"].flatten().to(device)
    preds = model(tensors)[player_masks]
    labels = tensors["player_trajs"].flatten().to(device)[player_masks]
    return (preds, labels)


def train_model():
    # Initialize optimizer.
    train_params = [params for params in model.parameters()]
    optimizer = optim.Adam(train_params, lr=opts["train"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Continue training on a prematurely terminated model.
    try:
        model.load_state_dict(torch.load(f"{JOB_DIR}/best_params.pth"))

        try:
            state_dict = torch.load(f"{JOB_DIR}/optimizer.pth")
            if opts["train"]["learning_rate"] == state_dict["param_groups"][0]["lr"]:
                optimizer.load_state_dict(state_dict)

        except ValueError:
            print("Old optimizer doesn't match.")

    except FileNotFoundError:
        pass

    train_loss = None
    best_train_loss = float("inf")
    best_basketball_valid_loss = float("inf")
    basketball_test_loss = float("inf")
    best_soccer_valid_loss = float("inf")
    soccer_test_loss = float("inf")
    no_improvement = 0
    for epoch in range(1000000):
        print(f"\nepoch: {epoch}", flush=True)

        model.eval()
        with torch.no_grad():
            n_basketball = 0
            n_soccer = 0
            basketball_valid_loss = 0.0
            soccer_valid_loss = 0.0
            valid_comp_ids = valid_dataset.comp_ids
            for (valid_idx, valid_tensors) in enumerate(valid_loader):
                # Skip bad sequences.
                if len(valid_tensors["player_idxs"]) < seq_len:
                    continue

                (preds, labels) = get_preds_labels(valid_tensors)
                loss = criterion(preds, labels)
                if len(valid_comp_ids[valid_idx].split("_")) == 1:
                    basketball_valid_loss += loss.item()
                    n_basketball += 1
                    n_players = 10

                else:
                    soccer_valid_loss += loss.item()
                    n_soccer += 1
                    n_players = 22

            probs = torch.softmax(preds, dim=1)
            (probs, preds) = probs.max(1)
            print(probs.view(seq_len, n_players), flush=True)
            print(preds.view(seq_len, n_players), flush=True)
            print(labels.view(seq_len, n_players), flush=True)

            if use_basketball:
                basketball_valid_loss /= n_basketball
            else:
                basketball_valid_loss = float("inf")

            if use_soccer:
                soccer_valid_loss /= n_soccer
            else:
                soccer_valid_loss = float("inf")

        if (basketball_valid_loss < best_basketball_valid_loss) or (
            soccer_valid_loss < best_soccer_valid_loss
        ):

            if use_basketball and (basketball_valid_loss < best_basketball_valid_loss):
                best_basketball_valid_loss = basketball_valid_loss
                torch.save(model.state_dict(), f"{JOB_DIR}/best_params_basketball.pth")

            if use_soccer and (soccer_valid_loss < best_soccer_valid_loss):
                best_soccer_valid_loss = soccer_valid_loss
                torch.save(model.state_dict(), f"{JOB_DIR}/best_params_soccer.pth")

            no_improvement = 0
            torch.save(optimizer.state_dict(), f"{JOB_DIR}/optimizer.pth")
            torch.save(model.state_dict(), f"{JOB_DIR}/best_params.pth")

            with torch.no_grad():
                n_basketball = 0
                n_soccer = 0
                basketball_test_loss = 0.0
                soccer_test_loss = 0.0
                test_comp_ids = test_dataset.comp_ids
                for (test_idx, test_tensors) in enumerate(test_loader):
                    # Skip bad sequences.
                    if len(test_tensors["player_idxs"]) < seq_len:
                        continue

                    (preds, labels) = get_preds_labels(test_tensors)
                    loss = criterion(preds, labels)
                    if len(test_comp_ids[test_idx].split("_")) == 1:
                        basketball_test_loss += loss.item()
                        n_basketball += 1
                    else:
                        soccer_test_loss += loss.item()
                        n_soccer += 1

            if use_basketball:
                basketball_test_loss /= n_basketball

            if use_soccer:
                soccer_test_loss /= n_soccer

        elif no_improvement < opts["train"]["patience"]:
            no_improvement += 1
            if no_improvement == opts["train"]["patience"]:
                print("Reducing learning rate.")
                for g in optimizer.param_groups:
                    g["lr"] *= 0.1

        print(f"train_loss: {train_loss}")
        print(f"best_train_loss: {best_train_loss}")
        if use_basketball:
            print(f"basketball_valid_loss: {basketball_valid_loss}")
            print(f"best_basketball_valid_loss: {best_basketball_valid_loss}")
            print(f"basketball_test_loss: {basketball_test_loss}")

        if use_soccer:
            print(f"soccer_valid_loss: {soccer_valid_loss}")
            print(f"best_soccer_valid_loss: {best_soccer_valid_loss}")
            print(f"soccer_test_loss: {soccer_test_loss}")

        model.train()
        train_loss = 0.0
        n_train = 0
        start_time = time.time()
        for (train_idx, train_tensors) in enumerate(train_loader):
            if train_idx % 1000 == 0:
                print(train_idx, flush=True)

            # Skip bad sequences.
            if len(train_tensors["player_idxs"]) < seq_len:
                continue

            optimizer.zero_grad()
            (preds, labels) = get_preds_labels(train_tensors)
            # For some soccer data, all the players are missing.
            if len(preds) == 0:
                continue

            loss = criterion(preds, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            n_train += 1

        epoch_time = time.time() - start_time

        train_loss /= n_train
        if train_loss < best_train_loss:
            best_train_loss = train_loss

        print(f"epoch_time: {epoch_time:.2f}", flush=True)


if __name__ == "__main__":
    JOB = sys.argv[1]
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    except IndexError:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))

    # Initialize datasets.
    (
        train_dataset,
        train_loader,
        valid_dataset,
        valid_loader,
        test_dataset,
        test_loader,
    ) = init_datasets(opts)

    # Initialize model.
    device = torch.device("cuda:0")

    model = init_model(opts, train_dataset).to(device)
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params}")

    seq_len = model.seq_len
    use_basketball = opts["train"]["use_basketball"]
    use_soccer = opts["train"]["use_soccer"]

    train_model()
