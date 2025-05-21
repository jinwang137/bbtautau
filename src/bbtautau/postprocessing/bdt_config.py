from __future__ import annotations

bdt_config = {
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
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "num_class": 8,
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
