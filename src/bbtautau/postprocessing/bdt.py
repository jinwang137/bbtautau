from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from bdt_config import bdt_config
from boostedhh import hh_vars, plotting, utils
from boostedhh.utils import Sample
from Samples import CHANNELS, SAMPLES
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from bbtautau.postprocessing.postprocessing import (
    base_filters_default,
    bbtautau_assignment,
    derive_variables,
    get_columns,
)
from bbtautau.postprocessing.utils import LoadedSample


class Trainer:

    loaded_dmatrix = False

    sample_names: ClassVar[list[str]] = [
        "qcd",
        "ttbarhad",
        "ttbarsl",
        "ttbarll",
        "dyjets",
        "bbtt",
    ]

    samples: ClassVar[dict[str, Sample]] = {name: SAMPLES[name] for name in sample_names}

    del sample_names

    def __init__(self, years, modelname=None) -> None:
        self.years = years
        self.data_path = {
            "signal": Path(
                "/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr17bbpresel_v12_private_signal/"
            ),
            "bg": Path(
                "/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr17bbpresel_v12_private_signal/"
            ),
        }
        self.bdt_config = bdt_config
        if modelname is not None:
            self.modelname = modelname
        else:
            self.modelname = "test"
        self.model_dir = Path(
            f"/home/users/lumori/bbtautau/src/bbtautau/postprocessing/classifier/trained_models/{self.modelname}_{'-'.join(self.years)}"
        )
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, base_filters=True, force_reload=False):
        # Check if data buffer file exists
        if self.model_dir / "dtrain.buffer" in self.model_dir.glob("*.buffer") and not force_reload:
            print("Loading data from buffer file")
            self.dtrain = xgb.DMatrix(self.model_dir / "dtrain.buffer")
            self.dval = xgb.DMatrix(self.model_dir / "dval.buffer")
            self.loaded_dmatrix = True
        else:
            self.events_dict = {year: {} for year in self.years}
            for year in self.years:
                for key, sample in self.samples.items():
                    if sample.selector is not None:
                        sample.load_columns = get_columns(year)[sample.get_type()]
                        events = utils.load_sample(
                            sample,
                            year,
                            self.data_path,
                            base_filters_default if base_filters else None,
                        )
                        self.events_dict[year][key] = LoadedSample(sample=sample, events=events)
                        print(f"Successfully imported sample {sample.label} (key: {key}) to memory")
                derive_variables(
                    self.events_dict[year], CHANNELS["hm"]
                )  # legacy issue, muon branches are misnamed
                bbtautau_assignment(self.events_dict[year], agnostic=True)

            for ch, channel in CHANNELS.items():
                for year in self.years:
                    self.events_dict[year][f"bbtt{ch}"] = LoadedSample(
                        sample=SAMPLES[f"bbtt{ch}"],
                        events=self.events_dict[year]["bbtt"].events[
                            self.events_dict[year]["bbtt"].get_var(f"GenTau{ch}")
                        ],
                    )
                self.samples[f"bbtt{ch}"] = SAMPLES[f"bbtt{ch}"]
                bbtautau_assignment(self.events_dict[year], channel)
            del self.samples["bbtt"]
            for year in self.years:
                del self.events_dict[year]["bbtt"]

    @staticmethod
    def shorten_df(df, N, seed=42):
        if len(df) < N:
            return df
        return df.sample(n=N, random_state=seed)

    def prepare_training_set(self, train=False):
        """Prepare features and labels using LabelEncoder for multiclass classification."""
        # Get hyperparameters and training variables from config
        self.hyperpars = self.bdt_config[self.modelname]["hyperpars"]
        self.train_vars = self.bdt_config[self.modelname]["train_vars"]

        if self.loaded_dmatrix:
            # TODO need to have stored the sample names somewhere
            return

        # Initialize lists for features, labels, and weights
        X_list = []
        weights_list = []
        sample_names_labels = []  # Store sample names for each event

        # Process each sample
        for year in self.years:
            total_signal_weight = np.concatenate(
                [
                    np.abs(self.events_dict[year][sig_sample].events["finalWeight"].to_numpy())
                    for sig_sample in self.samples
                    if self.samples[sig_sample].isSignal
                ]
            ).sum()
            total_ttbar_weight = np.concatenate(
                [
                    np.abs(self.events_dict[year][ttbar_sample].events["finalWeight"].to_numpy())
                    for ttbar_sample in self.samples
                    if "ttbar" in ttbar_sample
                ]
            ).sum()
            print("total_signal_weight", total_signal_weight)
            print("total_ttbar_weight", total_ttbar_weight)

            for sample_name, sample in self.events_dict[year].items():

                X_sample = sample.events[self.train_vars["misc"]["feats"]].assign(
                    **{
                        feat: sample.events[feat].where(sample.tt_mask, 0).sum(axis=1)
                        for feat in self.train_vars["fatjet"]["feats"]
                    }
                )

                # Get weights
                weights = np.abs(sample.events["finalWeight"].to_numpy())

                if sample.sample.isSignal:
                    weights = weights * 1e5
                elif "ttbar" in sample_name:
                    weights = weights / total_ttbar_weight * total_signal_weight * 1e5
                else:  # qcd, dy
                    weights = weights * total_signal_weight / np.sum(weights) * 1e5

                X_list.append(X_sample)
                weights_list.append(weights)

                sample_names_labels.extend([sample_name] * len(sample.events))

        # Combine all samples
        X = pd.concat(X_list, axis=0)
        weights = np.concatenate(weights_list)

        # Use LabelEncoder to convert sample names to numeric labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(sample_names_labels)
        self.classes = self.label_encoder.classes_

        # Print class mapping
        print("\nClass mapping:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"Class {i}: {class_name}")

        # Split into training and validation sets
        X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
            X,
            y,
            weights,
            test_size=0.8,
            random_state=self.bdt_config[self.modelname]["random_seed"],
            stratify=y,
        )

        print(f"X_train.shape: {X_train.shape}, X_val.shape: {X_val.shape}")

        # Create DMatrix objects
        self.dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights_train, nthread=-1)
        self.dval = xgb.DMatrix(X_val, label=y_val, weight=weights_val, nthread=-1)

        # save buffer for quicker loading
        if train:
            self.dtrain.save_binary(self.model_dir / "dtrain.buffer")
            self.dval.save_binary(self.model_dir / "dval.buffer")

    def train_model(self):
        """Trains BDT. ``classifier_params`` are hyperparameters for the classifier"""

        # early_stopping_callback = xgb.callback.EarlyStopping(rounds=5, min_delta=0.0)
        # classifier_params = {
        #     **self.bdt_config[self.modelname]["hyperpars"],
        #     "callbacks": [early_stopping_callback],
        # }

        evals_result = {}

        evallist = [(self.dtrain, "train"), (self.dval, "eval")]
        self.bst = xgb.train(
            self.hyperpars,
            self.dtrain,
            self.bdt_config[self.modelname]["num_rounds"],
            evals=evallist,
            evals_result=evals_result,
        )
        self.bst.save_model(self.model_dir / f"{self.modelname}.json")

        # Save evaluation results as JSON
        with (self.model_dir / "evals_result.json").open("w") as f:
            json.dump(evals_result, f, indent=2)

        return

    def load_model(self):
        self.bst = xgb.Booster()
        print(f"loading model {self.modelname}")
        try:
            self.bst.load_model(self.model_dir / f"{self.modelname}.json")
            print("loading successful")
        except Exception as e:
            print(e)
        return self.bst

    def evaluate_training(self):
        # Load evaluation results from JSON
        with (self.model_dir / "evals_result.json").open("r") as f:
            evals_result = json.load(f)

        plt.figure(figsize=(10, 6))
        plt.plot(evals_result["train"][self.hyperpars["eval_metric"]], label="Train")
        plt.plot(evals_result["eval"][self.hyperpars["eval_metric"]], label="Validation")
        plt.xlabel("Iteration")
        plt.ylabel(self.hyperpars["eval_metric"])
        plt.legend()
        plt.savefig(self.model_dir / "training_history.png")
        plt.close()

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        xgb.plot_importance(self.bst, max_num_features=20)
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(self.model_dir / "feature_importance.png")
        plt.close()

    def complete_train(self, training_info=True, force_reload=False, **kwargs):
        """Train a multiclass BDT model.

        Args:
            year (str): Year of data to use
            modelname (str): Name of the model configuration to use
            save_dir (str, optional): Directory to save the model and plots. If None, uses default location.

        """
        # out-of-the-box for training
        self.load_data(force_reload=force_reload, **kwargs)
        self.prepare_training_set(train=True, **kwargs)
        self.train_model(**kwargs)
        if training_info:
            self.evaluate_training()
        self.compute_rocs()

    def complete_load(self, force_reload=False, **kwargs):
        self.load_data(force_reload=force_reload, **kwargs)
        self.prepare_training_set(**kwargs)
        self.load_model(**kwargs)
        self.compute_rocs()
        self.evaluate_training()

        # add other info: precision, accuracy, other metrics

    def compute_rocs(self, discs=None):

        y_pred = self.bst.predict(self.dval, output_margin=True)

        # TODO: de-hardcode this
        sigs_strs = {"hh": 0, "he": 1, "hm": 2}
        bg_strs = {"QCD": [4], "all": [3, 4, 5, 6, 7]}

        if not hasattr(self, "rocs"):
            self.rocs = {}
        self.rocs = {sig: {} for sig in sigs_strs}

        if discs is None:
            for sig in sigs_strs:
                for bg in bg_strs:
                    disc = f"bbtt{sig}vs{bg}"

                    bg_truth = np.isin(self.dval.get_label(), bg_strs[bg])
                    sig_truth = self.dval.get_label() == sigs_strs[sig]
                    weights = self.dval.get_weight()

                    truth = np.zeros_like(bg_truth)
                    truth[sig_truth] = 1

                    sig_score = y_pred[:, sigs_strs[sig]]
                    bg_scores = np.sum([y_pred[:, b] for b in bg_strs[bg]], axis=0)
                    disc_score = np.divide(
                        sig_score,
                        sig_score + bg_scores,
                        out=np.zeros_like(sig_score),
                        where=(sig_score + bg_scores != 0),
                    )

                    # print(score[sig_truth])
                    # print(score[self.dval.get_label() == 3])

                    (self.model_dir / "scores").mkdir(parents=True, exist_ok=True)
                    (self.model_dir / "outputs").mkdir(parents=True, exist_ok=True)

                    plotting.plot_hist(
                        [disc_score[sig_truth]]
                        + [disc_score[self.dval.get_label() == b] for b in bg_strs["all"]],
                        [sig] + [bg_strs["all"]],
                        nbins=100,
                        xlim=(-0.7, 0.7),
                        weights=[weights[sig_truth]]
                        + [weights[self.dval.get_label() == b] for b in bg_strs["all"]],
                        xlabel=f"BDT {disc} score",
                        lumi=f"{np.sum([hh_vars.LUMI[year] for year in self.years]) / 1000:.1f}",
                        density=True,
                        year="-".join(self.years) if len(self.years) < 4 else "2022-2023",
                        saveas=self.model_dir / f"scores/bdt_scores_{sig}_{bg}.png",
                    )

                    # exclude events that are not signal or background
                    mask = bg_truth | sig_truth
                    disc_score = disc_score[mask]
                    truth = truth[mask]
                    weights_all = weights.copy()
                    weights = weights[mask]
                    fpr, tpr, thresholds = roc_curve(truth, disc_score, sample_weight=weights)
                    # print(fpr, tpr, thresholds)
                    roc_auc = auc(fpr, tpr)

                    self.rocs[sig][bg] = {
                        "fpr": fpr,
                        "tpr": tpr,
                        "thresholds": thresholds,
                        "label": disc,
                        "auc": roc_auc,
                    }

                plotting.plot_hist(
                    [sig_score[sig_truth]]
                    + [bg_scores[self.dval.get_label() == b] for b in bg_strs["all"]],
                    [sig] + bg_strs["all"],  # need to have a label mapping in case quick load
                    nbins=100,
                    weights=[weights_all[sig_truth]]
                    + [weights_all[self.dval.get_label() == b] for b in bg_strs["all"]],
                    xlabel="BDT output score",
                    lumi=f"{np.sum([hh_vars.LUMI[year] for year in self.years]) / 1000:.1f}",
                    density=True,
                    year="-".join(self.years) if len(self.years) < 4 else "2022-2023",
                    saveas=self.model_dir / f"outputs/bdt_outputs_{sig}.png",
                )

                # Plot ROC curve
                (self.model_dir / "rocs").mkdir(parents=True, exist_ok=True)
                plotting.multiROCCurve(
                    {"": {b: self.rocs[sig][b] for b in bg_strs}},
                    thresholds=[0.3, 0.5, 0.9, 0.99],
                    show=True,
                    plot_dir=self.model_dir / "rocs",
                    lumi=f"{np.sum([hh_vars.LUMI[year] for year in self.years]) / 1000:.1f}",
                    year="-".join(self.years) if len(self.years) < 4 else "2022-2023",
                    name=f"roc_{sig}",
                )


if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a multiclass BDT model")

    parser.add_argument("--years", type=list[str], default=["2022"], help="Year of data to use")
    parser.add_argument(
        "--model", type=str, default="test", help="Name of the model configuration to use"
    )
    parser.add_argument(
        "--save-dir", type=str, default=None, help="Directory to save model and plots"
    )
    parser.add_argument(
        "--force-reload", action="store_true", default=False, help="Force reload of data"
    )

    # Add mutually exclusive group for train/load
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", action="store_true", help="Train a new model")
    group.add_argument("--load", action="store_true", default=True, help="Load model from file")

    args = parser.parse_args()
    trainer = Trainer(args.years, args.model)

    if args.train:
        trainer.complete_train(force_reload=args.force_reload)
    else:
        trainer.complete_load(force_reload=args.force_reload)
