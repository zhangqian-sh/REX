import os
from typing import Callable

import equinox as eqx
import numpy as np
import optax
from absl import logging
from ml_collections import ConfigDict


def data_loader(data: list[np.ndarray], batch_size: int, shuffle: bool = False):
    def all_equal(lst):
        return all(x == lst[0] for x in lst)

    if not all_equal([len(d) for d in data]):
        raise ValueError("All data should have the same length")
    N = len(data[0])

    idx = np.random.permutation(N) if shuffle else np.arange(N)
    for i in range(0, N, batch_size):
        yield [x[idx[i : i + batch_size]] for x in data]


def train_model(
    cfg: ConfigDict,
    dataset: list[np.ndarray],
    model: eqx.Module,
    optimizer: optax.GradientTransformation,
    batch_iter: Callable,
    save_results: Callable,
):
    logging.info("Start training")
    # train loop
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    logging.info(f"Optimizer initialized. Start training loop.")
    loss_history = []
    best_loss, best_epoch, best_model = 1e5, 0, None
    for epoch in range(1, cfg.train.n_epochs + 1):
        loss_epoch = []
        train_loader = data_loader(dataset, cfg.train.batch_size, True)
        for batch_idx, batch_input in enumerate(train_loader):
            model, opt_state, loss = batch_iter(cfg, batch_input, model, opt_state=opt_state)
            logging.info(f"Epoch {epoch}, batch {batch_idx}, loss: {loss:.4e}")
            loss_epoch.append(loss)
        loss = np.mean(loss_epoch)
        loss_history.append(loss)
        logging.info(f"Epoch {epoch} loss: {loss}")
        if loss < best_loss:
            best_loss, best_epoch, best_model = loss, epoch, model
        if epoch % (cfg.train.n_epochs // cfg.train.n_save) == 0:
            eval_loader = data_loader(dataset, cfg.train.batch_size, False)
            eval_iter = lambda x: batch_iter(cfg, x, model)
            eval_output, loss = zip(*map(eval_iter, eval_loader))
            save_results(cfg, model, eval_output, epoch=epoch)
    model = best_model
    eqx.tree_serialise_leaves(os.path.join(cfg.io.model_folder, "best.eqx"), best_model)
    np.savetxt(f"{cfg.io.result_folder}/loss_history.csv", loss_history, delimiter=",")
    logging.info(f"Training finished. Best loss: {best_loss} at epoch {best_epoch}")
    return model


def load_model(model_init: eqx.Module, model_path: str):
    return eqx.tree_deserialise_leaves(model_path, model_init)


def test_model_and_save_results(
    cfg: dict, dataset: np.ndarray, model: eqx.Module, batch_iter: Callable, save_results: Callable
):
    logging.info("Start testing")
    eval_loader = data_loader(dataset, cfg.train.batch_size, False)
    eval_iter = lambda x: batch_iter(cfg, x, model)
    eval_output, loss = zip(*map(eval_iter, eval_loader))
    save_results(cfg, model, eval_output)
    logging.info(f"Testing finished. Test loss: {np.mean(loss)}.")
