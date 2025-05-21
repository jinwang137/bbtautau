from __future__ import annotations

import argparse
from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from bdt_config import bdt_config
from boostedhh import hh_vars, plotting, utils
from Samples import SAMPLES
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from bbtautau.postprocessing.postprocessing import base_filters_default, get_columns
from bbtautau.postprocessing.utils import LoadedSample


class Trainer:

    loaded_dmatrix = False

    sample_names: ClassVar[list[str]] = [
        "bbtt",
        "qcd",
        "ttbarhad",
        "ttbarsl",
        "ttbarll",
        "dyjets",
    ]

    samples: ClassVar[dict[str, any]] = {name: SAMPLES[name] for name in sample_names}

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

    def load_data(self, base_filters=True):
        # Check if data buffer file exists
        if self.model_dir / "dtrain.buffer" in self.model_dir.glob("*.buffer"):
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
                    # print(sample.load_columns)
                    events = utils.load_sample(
                        sample, year, self.data_path, base_filters_default if base_filters else None
                    )

                    self.events_dict[year][key] = LoadedSample(sample=sample, events=events)
                    print(f"Successfully imported sample {sample.label} (key: {key}) to memory")

            for year in self.years:
                for ch in ["hh", "he", "hm"]:
                    self.events_dict[year][f"bbtt{ch}"] = LoadedSample(
                        sample=SAMPLES[f"bbtt{ch}"],
                        events=self.events_dict[year]["bbtt"].events[
                            self.events_dict[year]["bbtt"].get_var(f"GenTau{ch}")
                        ],
                    )
                    self.samples[f"bbtt{ch}"] = SAMPLES[f"bbtt{ch}"]
                del self.events_dict[year]["bbtt"]
            del self.samples["bbtt"]

    @staticmethod
    def shorten_df(df, N, seed=42):
        if len(df) < N:
            return df
        return df.sample(n=N, random_state=seed)

    def prepare_training_set(self, train=False, njets=2):
        """Prepare features and labels using LabelEncoder for multiclass classification."""

        # Get hyperparameters and training variables from config
        self.hyperpars = self.bdt_config[self.modelname]["hyperpars"]
        self.train_vars = self.bdt_config[self.modelname]["train_vars"]

        if self.loaded_dmatrix:
            return None

        # Initialize lists for features, labels, and weights
        X_list = []
        weights_list = []
        sample_names = []  # Store sample names for each event

        # Process each sample
        for year in self.years:
            for sample_name, sample in self.events_dict[year].items():
                # Flatten multi-entry branches into individual columns
                flattened_events = {}

                # filter by simple preselection
                # twojets = (sample.events["ak8FatJetPt"][0] > 250) & (sample.events["ak8FatJetPt"][1] > 200)
                # sample.events = sample.events[twojets]

                # flatten fatjet variables
                for key in self.train_vars["fatjet"]["feats"]:
                    for jet_i in range(njets):
                        flattened_events[f"{key}_{jet_i}"] = (
                            sample.events[key].iloc[:, jet_i].to_numpy().flatten()
                        )

                # misc variables
                for key in self.train_vars["misc"]["feats"]:
                    flattened_events[key] = sample.events[key].to_numpy().flatten()

                # Convert to DataFrame
                X_sample = pd.DataFrame(data=flattened_events)

                # Get weights
                weights = np.abs(sample.events["finalWeight"].to_numpy())

                # Store features and weights
                X_list.append(X_sample)
                weights_list.append(weights)

                # Store sample names for each event
                sample_names.extend([sample_name] * len(sample.events))

        # Combine all samples
        X = pd.concat(X_list, axis=0)
        weights = np.concatenate(weights_list)

        # Use LabelEncoder to convert sample names to numeric labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(sample_names)

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
        self.dval = xgb.DMatrix(X_val, label=y_val, weight=weights_val)

        # # save buffer for quicker loading
        if train:
            self.dtrain.save_binary(self.model_dir / "dtrain.buffer")
            self.dval.save_binary(self.model_dir / "dval.buffer")

        # Update hyperparameters with number of classes (for now keep in config)
        # self.hyperpars = hyperpars.copy()
        # self.hyperpars["num_class"] = len(self.label_encoder.classes_)

        return X_train, X_val, y_train, y_val, weights_train, weights_val

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

        with (self.model_dir / "evals_result.txt").open("w") as f:
            f.write(str(evals_result))

        return evals_result

    def load_model(self):
        self.bst = xgb.Booster()
        print(f"loading model {self.modelname}")
        try:
            self.bst.load_model(self.model_dir / f"{self.modelname}.json")
            print("loading successful")
        except Exception as e:
            print(e)
        return self.bst

    def evaluate_training(self, evals_result):
        plt.figure(figsize=(10, 6))
        plt.plot(evals_result["train"]["auc"], label="Train")
        plt.plot(evals_result["eval"]["auc"], label="Validation")
        plt.xlabel("Iteration")
        plt.ylabel("AUC Score")
        plt.title("Training History")
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

        # Print summary
        print("\nTraining Summary:")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        print("\nClass mapping:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"Class {i}: {class_name}")

    def complete_train(self, training_info=True, **kwargs):
        """Train a multiclass BDT model.

        Args:
            year (str): Year of data to use
            modelname (str): Name of the model configuration to use
            save_dir (str, optional): Directory to save the model and plots. If None, uses default location.

        """
        # out-of-the-box for training
        self.load_data(**kwargs)
        self.prepare_training_set(train=True, **kwargs)
        evals_result = self.train_model(**kwargs)
        if training_info:
            self.evaluate_training(evals_result)
        self.compute_rocs()

    def complete_load(self, **kwargs):
        self.load_data(**kwargs)
        self.prepare_training_set(**kwargs)
        self.load_model(**kwargs)
        self.compute_rocs()

        # add other info: precision, accuracy, other metrics

    def compute_rocs(self, discs=None):

        y_pred = self.bst.predict(self.dval, output_margin=True)

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
                    score = np.divide(
                        sig_score,
                        sig_score + bg_scores,
                        out=np.zeros_like(sig_score),
                        where=(sig_score + bg_scores != 0),
                    )

                    plotting.plot_hist(
                        [sig_score[sig_truth], bg_scores[bg_truth]],
                        [sig, bg],
                        nbins=100,
                        weights=[weights[sig_truth], weights[bg_truth]],
                        xlabel="BDT score",
                        lumi=f"{np.sum([hh_vars.LUMI[year] for year in self.years]) / 1000:.1f}",
                        density=True,
                        year="-".join(self.years) if len(self.years) < 4 else "2022-2023",
                        saveas=self.model_dir / f"bdt_scores_{sig}_{bg}.png",
                    )

                    # exclude events that are not signal or background
                    mask = bg_truth | sig_truth
                    score = score[mask]
                    truth = truth[mask]
                    weights = weights[mask]
                    fpr, tpr, thresholds = roc_curve(truth, score, sample_weight=weights)
                    # print(fpr, tpr, thresholds)
                    roc_auc = auc(fpr, tpr)

                    self.rocs[sig][bg] = {
                        "fpr": fpr,
                        "tpr": tpr,
                        "thresholds": thresholds,
                        "label": disc,
                        "auc": roc_auc,
                    }

                    # Plot ROC curve
                plotting.multiROCCurve(
                    {"": {b: self.rocs[sig][b] for b in bg_strs}},
                    thresholds=[0.3, 0.5, 0.9, 0.99],
                    show=True,
                    plot_dir=self.model_dir,
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

    # Add mutually exclusive group for train/load
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", action="store_true", help="Train a new model")
    group.add_argument("--load", action="store_true", default=True, help="Load model from file")

    args = parser.parse_args()
    trainer = Trainer(args.years, args.model)

    if args.train:
        trainer.complete_train()
    else:
        trainer.complete_load()
