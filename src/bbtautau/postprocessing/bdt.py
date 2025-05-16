from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from boostedhh import utils
from Samples import SAMPLES
from sklearn.metrics import accuracy_score, precision_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from bbtautau.postprocessing.postprocessing import get_columns
from bbtautau.postprocessing.utils import LoadedSample


class Trainer:

    sample_names: ClassVar[list[str]] = [
        "bbtt",
        "qcd",
        "ttbarhad",
        "ttbarsl",
        "ttbarll",
        "dyjets",
    ]

    samples: ClassVar[dict[str, any]] = {name: SAMPLES[name] for name in sample_names}

    # temporary
    bdt_config: ClassVar[dict[str, dict]] = {
        "test": {
            "modelname": "test",
            "hyperpars": {
                "objective": "multi:softmax",
                "max_depth": 8,
                "subsample": 0.3,
                "alpha": 8.0,
                "gamma": 2.0,
                "lambda": 2.0,
                "min_child_weight": 0,
                "colsample_bytree": 1.0,
                "num_parallel_tree": 50,
                "eval_metric": "auc",
                "tree_method": "hist",
                # "device": "cuda",
            },
            "num_rounds": 8,
            "random_seed": 42,
            "train_vars": {
                "fatjet": {
                    # will probably only keep the "chosen" tt and bb jets
                    "feats": [
                        "ak8FatJetPt",
                        "ak8FatJetPNetXbbLegacy",
                        "ak8FatJetPNetQCDLegacy",
                        "ak8FatJetPNetmassLegacy",
                        "ak8FatJetParTmassResApplied",
                        "ak8FatJetParTmassVisApplied",
                        "ak8FatJetMsd",
                        "ak8FatJetParTQCD0HF",
                        "ak8FatJetParTQCD1HF",
                        "ak8FatJetParTQCD2HF",
                        "ak8FatJetParTTopW",
                        "ak8FatJetParTTopbW",
                        "ak8FatJetParTXtauhtauh",
                        "ak8FatJetParTXtauhtaue",
                        "ak8FatJetParTXtauhtaum",
                        "ak8FatJetParTXbb",
                        "ak8FatJetEta",
                    ],
                    "len": 3,
                },
                "misc": {
                    "feats": [
                        "METPt",
                    ],
                    "len": 1,
                },
            },
        }
    }

    def __init__(self, year, modelname=None) -> None:
        self.year = year
        self.data_path = {
            "signal": Path(
                "/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr17bbpresel_v12_private_signal/"
            ),
            "bg": Path(
                "/ceph/cms/store/user/rkansal/bbtautau/skimmer/25Apr17bbpresel_v12_private_signal/"
            ),
        }
        if modelname is not None:
            self.modelname = modelname
        else:
            self.modelname = "test"
        self.model_dir = Path(
            f"/home/users/lumori/bbtautau/src/bbtautau/postprocessing/classifier/trained_models/{self.modelname}_{self.year}"
        )
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        self.events_dict = {}
        for key, sample in self.samples.items():
            if sample.selector is not None:
                sample.load_columns = get_columns(self.year)[sample.get_type()]
                # print(sample.load_columns)
                events = utils.load_sample(sample, self.year, self.data_path)

                self.events_dict[key] = LoadedSample(sample=sample, events=events)
                print(f"Successfully imported sample {sample.label} to memory")

        for ch in ["hh", "he", "hm"]:
            self.events_dict[f"bbtt{ch}"] = LoadedSample(
                sample=SAMPLES[f"bbtt{ch}"],
                events=self.events_dict["bbtt"].events[
                    self.events_dict["bbtt"].get_var(f"GenTau{ch}")
                ],
            )
            self.samples[f"bbtt{ch}"] = SAMPLES[f"bbtt{ch}"]
        del self.events_dict["bbtt"]
        del self.samples["bbtt"]

    @staticmethod
    def shorten_df(df, N, seed=42):
        if len(df) < N:
            return df
        return df.sample(n=N, random_state=seed)

    def prepare_training_set(self):
        """Prepare features and labels using LabelEncoder for multiclass classification."""
        # Get hyperparameters and training variables from config
        hyperpars = self.bdt_config[self.modelname]["hyperpars"]
        self.train_vars = self.bdt_config[self.modelname]["train_vars"]

        # Initialize lists for features, labels, and weights
        X_list = []
        weights_list = []
        sample_names = []  # Store sample names for each event

        # Process each sample
        for sample_name, sample in self.events_dict.items():
            # Flatten multi-entry branches into individual columns
            flattened_events = {}

            # flatten fatjet variables
            for key in self.train_vars["fatjet"]["feats"]:
                for jet_i in range(3):
                    flattened_events[f"{key}_{jet_i}"] = (
                        sample.events[key].iloc[:, jet_i].to_numpy().flatten()
                    )
                    # probably kill jets with padded values here

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
        le = LabelEncoder()
        y = le.fit_transform(sample_names)

        # Print class mapping
        print("\nClass mapping:")
        for i, class_name in enumerate(le.classes_):
            print(f"Class {i}: {class_name}")

        # Split into training and validation sets
        X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
            X,
            y,
            weights,
            test_size=0.2,
            random_state=self.bdt_config[self.modelname]["random_seed"],
            stratify=y,
        )

        print(X_train.shape, X_val.shape)

        # Create DMatrix objects
        self.dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights_train, nthread=-1)
        self.dval = xgb.DMatrix(X_val, label=y_val, weight=weights_val)

        # save buffer for quicker loading
        self.dtrain.save_binary(self.model_dir / "dtrain.buffer")
        self.dval.save_binary(self.model_dir / "dval.buffer")

        # Update hyperparameters with number of classes
        self.hyperpars = hyperpars.copy()
        self.hyperpars["num_class"] = len(le.classes_)

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

        return self.bst, evals_result

    def load_model(self):
        self.bst = xgb.Booster()
        self.bst.load_model(self.model_dir / f"{self.modelname}.json")
        return self.bst


#     @staticmethod
#     def plot_hist(
#         data,
#         names,
#         nbins=100,
#         weights=None,
#         xlabel=None,
#         saveas=None,
#         text=None,
#         xlim=None,
#         log=False,
#         density=False,
#         int_xticks=False,
#     ):
#         hep.style.use("CMS")
#         colors = plt.cm.tab10.colors
#         fig, ax = plt.subplots(figsize=(12, 9))
#         hep.cms.label("Preliminary", data=True, lumi=config["lumi"]["offline"], com=config["com"])
#         if weights is not None:
#             for d, w, name, c in zip(data, weights, names, colors[: len(data)]):
#                 # print(d,w,name)
#                 ax.hist(
#                     d,
#                     bins=nbins,
#                     weights=w,
#                     range=xlim,
#                     label=name,
#                     color=c,
#                     density=density,
#                     log=log,
#                     histtype="step",
#                     linewidth=2,
#                 )
#                 # ax.hist(d, bins = nbins, range = xlim, color=c, density = density, log=log, alpha = 0.5)# hatch = '*',
#         else:
#             for d, name, c in zip(data, names, colors[: len(data)]):
#                 ax.hist(
#                     d,
#                     bins=nbins,
#                     range=xlim,
#                     label=name,
#                     color=c,
#                     density=density,
#                     log=log,
#                     histtype="step",
#                     linewidth=2,
#                 )
#                 # ax.hist(d, bins = nbins, range = xlim, color=c, density = density, log=log, alpha = 0.5)# hatch = '*',
#         if xlabel:
#             ax.set_xlabel(xlabel)
#         if text != None:
#             ax.text(
#                 0.02,
#                 0.6,
#                 text,
#                 fontsize=13,
#                 bbox=dict(facecolor="white", edgecolor="black"),
#                 transform=ax.transAxes,
#             )
#         if int_xticks:
#             ax.xaxis.get_major_locator().set_params(integer=True)
#         ax.set_ylabel("Normalized frequency")
#         ax.set_xlim(xlim)
#         ax.legend()  # (fontsize=13,frameon=True,edgecolor='black',fancybox=False)
#         # ax.grid(True)
#         if saveas:
#             plt.savefig(saveas)
#             print(f"saved figure as {saveas}")
#         return

#     @staticmethod
#     def plot_scatter(
#         xs,
#         ys,
#         names,
#         xlabel=None,
#         ylabel=None,
#         saveas=None,
#         text=None,
#         xlim=None,
#         log=False,
#         int_xticks=False,
#     ):
#         hep.style.use("CMS")
#         colors = plt.cm.tab10.colors
#         fig, ax = plt.subplots(figsize=(12, 9))
#         hep.cms.text("Preliminary")
#         for x, y, name, c in zip(xs, ys, names, colors[: len(xs)]):
#             ax.scatter(x, y, label=name, color=c)
#             ax.plot(x, y, color=c, alpha=0.5)
#         if xlabel:
#             ax.set_xlabel(xlabel)
#         if ylabel:
#             ax.set_ylabel(ylabel)
#         if text != None:
#             ax.text(
#                 0.02,
#                 0.8,
#                 text,
#                 fontsize=13,
#                 bbox=dict(facecolor="white", edgecolor="black"),
#                 transform=ax.transAxes,
#             )
#         if int_xticks:
#             ax.xaxis.get_major_locator().set_params(integer=True)
#         ax.set_xlim(xlim)
#         ax.legend(fontsize=13, frameon=True, edgecolor="black", fancybox=False)
#         # ax.grid(True)
#         if saveas:
#             plt.savefig(saveas)
#             print(f"saved figure as {saveas}")
#         return


# def plot_ROC(
#     trainers,
#     labels,
#     evals,
#     n_points=50,
#     tmva=None,
#     text="",
#     textpos=(0.02, 0.6),
#     textsize=14,
#     log=False,
# ):
#     hep.style.use("CMS")
#     colors = plt.cm.tab10.colors
#     fig, ax = plt.subplots(figsize=(9, 9))
#     hep.cms.label("Preliminary", data=True, lumi=config["lumi"]["offline"], com=config["com"])

#     dis = np.linspace(0.001, 0.999, n_points)

#     for trainer, eval, c1, l in zip(trainers, evals, colors, labels):

#         if len(eval.get_weight()) == 0:
#             eval.set_weight(np.ones((len(eval.get_label()),)))
#         true_test_sig = eval.get_label() == 1
#         true_test_bkg = ~true_test_sig

#         preds_test = trainer.bst.predict(eval)

#         w = eval.get_weight()
#         w_sig = np.sum(eval.get_weight()[true_test_sig])
#         w_bkg = np.sum(eval.get_weight()[true_test_bkg])

#         @jit(nopython=True, parallel=True)
#         def compute_ROC_point(d):
#             pred_test_sig = preds_test > d
#             pred_test_bkg = ~pred_test_sig
#             return (
#                 np.sum(w[true_test_sig & pred_test_sig]) / w_sig,
#                 np.sum(w[true_test_bkg & pred_test_bkg]) / w_bkg,
#             )

#         sig_eff_test, bkg_rej_test = np.vectorize(compute_ROC_point)(dis)
#         ax.scatter(sig_eff_test, bkg_rej_test, color=c1, zorder=0, label=l)
#         ax.plot(sig_eff_test, bkg_rej_test, lw=1.3, color=c1)

#     ax.text(*textpos, text, fontsize=textsize, transform=ax.transAxes)
#     ax.set_xlabel("Signal efficiency")
#     ax.set_ylabel("Background rejection")
#     # ax.grid(True)

#     if tmva:
#         ax.scatter(
#             tmva["sig_eff_test_tmva"],
#             tmva["bkg_rej_test_tmva"],
#             color="blue",
#             zorder=0,
#             label="TMVA test",
#         )
#         ax.plot(tmva["sig_eff_test_tmva"], tmva["bkg_rej_test_tmva"], lw=1.3, color="blue")
#         ax.scatter(
#             tmva["sig_eff_train_tmva"],
#             tmva["bkg_rej_train_tmva"],
#             color="red",
#             zorder=0,
#             label="TMVA train",
#         )
#         ax.plot(tmva["sig_eff_train_tmva"], tmva["bkg_rej_train_tmva"], lw=1.3, color="red")

#     ax.legend()  # (fontsize=13,frameon=True,edgecolor='black',fancybox=False)
#     plt.show()
#     return


def train_multiclass_bdt(year="2022", modelname="test", save_dir=None):
    """Train a multiclass BDT model.

    Args:
        year (str): Year of data to use
        modelname (str): Name of the model configuration to use
        save_dir (str, optional): Directory to save the model and plots. If None, uses default location.

    Returns:
        tuple: (Trainer instance, trained model, evaluation results)
    """
    # Initialize trainer
    trainer = Trainer(year=year, modelname=modelname)

    # Load data
    print("\nLoading data...")
    trainer.load_data()

    # Prepare training set
    print("\nPreparing training set...")
    trainer.prepare_training_set()

    # Train model
    print("\nTraining model...")
    model, evals_result = trainer.train_model()

    # Generate and save plots
    print("\nGenerating plots...")
    plot_dir = Path(save_dir) if save_dir else trainer.model_dir
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(evals_result["validation_0"]["auc"], label="Train")
    plt.plot(evals_result["validation_1"]["auc"], label="Validation")
    plt.xlabel("Iteration")
    plt.ylabel("AUC Score")
    plt.title("Training History")
    plt.legend()
    plt.savefig(plot_dir / "training_history.png")
    plt.close()

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(model, max_num_features=20)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(plot_dir / "feature_importance.png")
    plt.close()

    # Calculate and save metrics
    y_pred = model.predict(trainer.X_val)
    metrics = {
        "accuracy": accuracy_score(trainer.y_val, y_pred),
        "precision": precision_score(trainer.y_val, y_pred, average=None),
        "classes": trainer.label_encoder.classes_.tolist(),
    }

    fpr, tpr, thresholds = roc_curve(trainer.y_val, y_pred)

    # Calculate the area under the ROC curve (AUC)
    # roc_auc = auc(fpr, tpr)

    # plotting.multiROCCurve(
    #             {"": {"fpr": fpr, "tpr": tpr, "label": "test"}},
    #             title=title,
    #             thresholds=[0.3, 0.5, 0.9, 0.99, 0.998],
    #             show=True,
    #             plot_dir=plot_dir,
    #             lumi=f"{np.sum([hh_vars.LUMI[year] for year in years]) / 1000:.1f}",
    #             year="2022-23" if len(years) == 4 else "+".join(years),
    #             name=f"roc_test.png",
    #         )

    with (plot_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTraining complete! Results saved to {plot_dir}")
    return trainer, model, evals_result


if __name__ == "__main__":
    import argparse
    import json

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a multiclass BDT model")

    parser.add_argument("--year", type=str, default="2022", help="Year of data to use")
    parser.add_argument(
        "--model", type=str, default="test", help="Name of the model configuration to use"
    )
    parser.add_argument(
        "--save-dir", type=str, default=None, help="Directory to save model and plots"
    )

    # Parse arguments
    args = parser.parse_args()

    # Train model
    trainer, model, evals_result = train_multiclass_bdt(
        year=args.year, modelname=args.model, save_dir=args.save_dir
    )

    # Print summary
    print("\nTraining Summary:")
    print(f"Number of classes: {len(trainer.label_encoder.classes_)}")
    print("\nClass mapping:")
    for i, class_name in enumerate(trainer.label_encoder.classes_):
        print(f"Class {i}: {class_name}")

    # Print final metrics
    y_pred = model.predict(trainer.X_val)
    accuracy = accuracy_score(trainer.y_val, y_pred)
    print(f"\nValidation accuracy: {accuracy:.3f}")

    precision = precision_score(trainer.y_val, y_pred, average=None)
    print("\nPer-class precision:")
    for class_name, prec in zip(trainer.label_encoder.classes_, precision):
        print(f"{class_name}: {prec:.3f}")
