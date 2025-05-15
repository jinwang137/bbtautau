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
            "num_parallel_tree": 500,
            "eval_metric": "auc",
            "tree_method": "hist",
            "device": "cuda",
        },
        "num_rounds": 30,
        "random_seed": 42,
        "train_vars": [
            "Mm_kin_lxy",
            "Mm_kin_l3d",
            "Mm_kin_sl3d",
            "Mm_kin_vtx_chi2dof",
            "Mm_kin_vtx_prob",
            "Mm_kin_alpha",
            "Mm_kin_alphaBS",
            "Mm_closetrk",
            "Mm_closetrks1",
            "Mm_closetrks2",
            "Mm_kin_pvip",
            "Mm_kin_spvip",
            "Mm_kin_eta",
            "Mm_kin_pvlip",
            "Mm_kin_slxy",
            "Mm_iso",
            "Mm_otherVtxMaxProb",
            "Mm_otherVtxMaxProb1",
            "Mm_otherVtxMaxProb2",
        ],
    }
}
