from __future__ import annotations

from pathlib import Path

from boostedhh import utils
from Samples import SAMPLES, SIGNALS

from bbtautau.HLTs import HLTs

MAIN_DIR = Path("/home/users/lumori/bbtautau/")

exclude = {"highptMu": False}

data_paths = {
    "2022": "/ceph/cms/store/user/lumori/bbtautau/skimmer/25Apr16AddVBF_v12_private_signal",
    "2022EE": "/ceph/cms/store/user/lumori/bbtautau/skimmer/25Apr16AddVBF_v12_private_signal",
    "2023": "/ceph/cms/store/user/lumori/bbtautau/skimmer/25Apr16AddVBF_v12_private_signal",
    "2023BPix": "/ceph/cms/store/user/lumori/bbtautau/skimmer/25Apr16AddVBF_v12_private_signal",
}


data_paths = {
    "2022": {
        "signal": Path(
            "/ceph/cms/store/user/lumori/bbtautau/skimmer/25Apr16AddVBF_v12_private_signal"
        ),
    },
    "2022EE": {
        "signal": Path(
            "/ceph/cms/store/user/lumori/bbtautau/skimmer/25Apr16AddVBF_v12_private_signal"
        ),
    },
    "2023": {
        "signal": Path(
            "/ceph/cms/store/user/lumori/bbtautau/skimmer/25Apr16AddVBF_v12_private_signal"
        ),
    },
    "2023BPix": {
        "signal": Path(
            "/ceph/cms/store/user/lumori/bbtautau/skimmer/25Apr16AddVBF_v12_private_signal"
        ),
    },
}

"""
Objectives:
- trigger study with all sig channels
- change trigger choice, add met
- correlation between triggers (subtract 1 at the time)
"""


class Analyser:
    """
    Process signal data and perform trigger study for (year, sig_key). sig_key in SIGNALS
    """

    def __init__(self, year, sig_key, test_mode=False):
        assert sig_key in SIGNALS, f"sig_key {sig_key} not in SIGNALS"
        self.sample = SAMPLES[sig_key]
        self.year = year
        self.test_mode = test_mode
        self.sig_path = data_paths[year]
        self.plot_dir = MAIN_DIR / f"plots/SensitivityStudy/25Apr16/{year}{sig_key}"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.events_dict = {}
        self.triggers_dict = HLTs.hlt_list()[year]

    def load_data(self):
        self.events_dict = utils.load_sample(self.sample, self.year, self.sig_path)
        print(f"Loaded {self.sample} for year {self.year} from {self.sig_path}")

    def trigger_plots(self):
        triggers_dict = {
            "tauhh": {
                "mask": self.events_dict["GenTauhh"][0],
                "label": r"$\tau_h\tau_h$",
                "triggers": {
                    "All": HLTs["pnet"]
                    | HLTs["pfjet"]
                    | HLTs["quadjet"]
                    | HLTs["singletau"]
                    | HLTs["ditau"],
                    "PNetBB": HLTs["pnetbb"],
                    "PNetTauTau": HLTs["pnettt"],
                    "PNet": HLTs["pnet"],
                    "PFJet": HLTs["pfjet"],
                    "PNet | SingleTau| Di-tau": HLTs["pnet"] | HLTs["singletau"] | HLTs["ditau"],
                    "PNet | PFJet | Quad-jet": HLTs["pnet"] | HLTs["pfjet"] | HLTs["quadjet"],
                    "PNet | PFJet | Quad-jet | SingleTau": HLTs["pnet"]
                    | HLTs["pfjet"]
                    | HLTs["quadjet"]
                    | HLTs["singletau"],
                    "PNet | PFJet | Quad-jet | Di-tau": HLTs["pnet"]
                    | HLTs["pfjet"]
                    | HLTs["quadjet"]
                    | HLTs["ditau"],
                    "PNet | PFJet | SingleTau| Di-tau": HLTs["pnet"]
                    | HLTs["pfjet"]
                    | HLTs["singletau"]
                    | HLTs["ditau"],
                },
            },
            "tauhh_minus": {
                "mask": self.events_dict["GenTauhh"][0],
                "label": r"$\tau_h\tau_h$",
                "triggers": {
                    "All": HLTs["pnet"]
                    | HLTs["pfjet"]
                    | HLTs["quadjet"]
                    | HLTs["singletau"]
                    | HLTs["ditau"],
                    "-PNetBB": HLTs["pnettt"]
                    | HLTs["pfjet"]
                    | HLTs["quadjet"]
                    | HLTs["singletau"]
                    | HLTs["ditau"],
                    "-PNetTauTau": HLTs["pnetbb"]
                    | HLTs["pfjet"]
                    | HLTs["quadjet"]
                    | HLTs["singletau"]
                    | HLTs["ditau"],
                    "-PFJet": HLTs["pnet"] | HLTs["quadjet"] | HLTs["singletau"] | HLTs["ditau"],
                    "-Quad-jet": HLTs["pnet"] | HLTs["pfjet"] | HLTs["singletau"] | HLTs["ditau"],
                    "-SingleTau": HLTs["pnet"] | HLTs["pfjet"] | HLTs["quadjet"] | HLTs["ditau"],
                    "-Di-tau": HLTs["pnet"] | HLTs["pfjet"] | HLTs["quadjet"] | HLTs["singletau"],
                },
            },
            "tauhmu": {
                "mask": self.events_dict["GenTauhmu"][0],
                "label": r"$\tau_h\mu$",
                "triggers": {
                    "All": HLTs["pnet"]
                    | HLTs["singlemuon"]
                    | HLTs["mutau"]
                    | HLTs["singletau"]
                    | HLTs["ditau"]
                    | HLTs["pfjet"],
                    "PNetBB": HLTs["pnetbb"],
                    "PNetTauTau": HLTs["pnettt"],
                    "PNetBB | TauTau": HLTs["pnet"],
                    "Muon": HLTs["singlemuon"],
                    "Mu-tau": HLTs["mutau"],
                    "SingleTau": HLTs["singletau"],
                    "Di-tau": HLTs["ditau"],
                    "PFJet": HLTs["pfjet"],
                },
            },
            "tauhmu_minus": {
                "mask": self.events_dict["GenTauhmu"][0],
                "label": r"$\tau_h\mu$",
                "triggers": {
                    "All": HLTs["pnet"]
                    | HLTs["singlemuon"]
                    | HLTs["mutau"]
                    | HLTs["singletau"]
                    | HLTs["ditau"]
                    | HLTs["pfjet"],
                    "-PNetBB": HLTs["pnettt"]
                    | HLTs["singlemuon"]
                    | HLTs["mutau"]
                    | HLTs["singletau"]
                    | HLTs["ditau"]
                    | HLTs["pfjet"],
                    "-PNetTauTau": HLTs["pnetbb"]
                    | HLTs["singlemuon"]
                    | HLTs["mutau"]
                    | HLTs["singletau"]
                    | HLTs["ditau"]
                    | HLTs["pfjet"],
                    "-Muon": HLTs["pnet"]
                    | HLTs["mutau"]
                    | HLTs["singletau"]
                    | HLTs["ditau"]
                    | HLTs["pfjet"],
                    "-Mu-tau": HLTs["pnet"]
                    | HLTs["singlemuon"]
                    | HLTs["singletau"]
                    | HLTs["ditau"]
                    | HLTs["pfjet"],
                    "-SingleTau": HLTs["pnet"]
                    | HLTs["singlemuon"]
                    | HLTs["mutau"]
                    | HLTs["ditau"]
                    | HLTs["pfjet"],
                    "-Di-tau": HLTs["pnet"]
                    | HLTs["singlemuon"]
                    | HLTs["mutau"]
                    | HLTs["singletau"]
                    | HLTs["pfjet"],
                    "-PFJet": HLTs["pnet"]
                    | HLTs["singlemuon"]
                    | HLTs["mutau"]
                    | HLTs["singletau"]
                    | HLTs["ditau"],
                },
            },
            "tauhe": {
                "mask": self.events_dict["GenTauhe"][0],
                "label": r"$\tau_he$",
                "triggers": {
                    "All": HLTs["pnet"]
                    | HLTs["egamma"]
                    | HLTs["etau"]
                    | HLTs["singletau"]
                    | HLTs["ditau"]
                    | HLTs["pfjet"],
                    "PNetBB": HLTs["pnetbb"],
                    "PNetTauTau": HLTs["pnettt"],
                    "PNetBB | TauTau": HLTs["pnet"],
                    "EGamma": HLTs["egamma"],
                    "e-tau": HLTs["etau"],
                    "SingleTau": HLTs["singletau"],
                    "Di-tau": HLTs["ditau"],
                },
            },
            "tauhe_minus": {
                "mask": self.events_dict["GenTauhe"][0],
                "label": r"$\tau_he$",
                "triggers": {
                    "All": HLTs["pnet"]
                    | HLTs["egamma"]
                    | HLTs["etau"]
                    | HLTs["singletau"]
                    | HLTs["ditau"]
                    | HLTs["pfjet"],
                    "-PNetBB": HLTs["pnettt"]
                    | HLTs["egamma"]
                    | HLTs["etau"]
                    | HLTs["singletau"]
                    | HLTs["ditau"]
                    | HLTs["pfjet"],
                    "-PNetTauTau": HLTs["pnetbb"]
                    | HLTs["egamma"]
                    | HLTs["etau"]
                    | HLTs["singletau"]
                    | HLTs["ditau"]
                    | HLTs["pfjet"],
                    "-EGamma": HLTs["pnet"]
                    | HLTs["etau"]
                    | HLTs["singletau"]
                    | HLTs["ditau"]
                    | HLTs["pfjet"],
                    "-e-tau": HLTs["pnet"]
                    | HLTs["egamma"]
                    | HLTs["singletau"]
                    | HLTs["ditau"]
                    | HLTs["pfjet"],
                    "-SingleTau": HLTs["pnet"]
                    | HLTs["egamma"]
                    | HLTs["etau"]
                    | HLTs["ditau"]
                    | HLTs["pfjet"],
                    "-Di-tau": HLTs["pnet"]
                    | HLTs["egamma"]
                    | HLTs["etau"]
                    | HLTs["singletau"]
                    | HLTs["pfjet"],
                    "-PFJet": HLTs["pnet"]
                    | HLTs["egamma"]
                    | HLTs["etau"]
                    | HLTs["singletau"]
                    | HLTs["ditau"],
                },
            },
        }
        if year in ["2023", "2023BPix"]:
            triggers_dict["tauhh"]["triggers"].update(
                {
                    "All": HLTs["pnet"]
                    | HLTs["pnetbb_lowpt"]
                    | HLTs["pfjet"]
                    | HLTs["quadjet"]
                    | HLTs["singletau"]
                    | HLTs["ditau"]
                    | HLTs["pk_quadjet_loose"],
                    "PNet | Parking ": HLTs["pnet"] | HLTs["pk_quadjet_loose"],
                    "Parking Quad-jet": HLTs["pk_quadjet_loose"],
                    "PNetBB (low pt)": HLTs["pnetbb_lowpt"],
                }
            )

            for key in triggers_dict["tauhh_minus"]["triggers"]:
                triggers_dict["tauhh_minus"]["triggers"][key] = (
                    triggers_dict["tauhh_minus"]["triggers"][key] | HLTs["pk_quadjet_loose"]
                )

            triggers_dict["tauhh_minus"]["triggers"].update(
                {
                    "-Parking Quad-jet": HLTs["pnet"]
                    | HLTs["pfjet"]
                    | HLTs["quadjet"]
                    | HLTs["singletau"]
                    | HLTs["ditau"],
                }
            )
            triggers_dict["tauhmu"]["triggers"].update(
                {
                    "PNetBB (low pt)": HLTs["pnetbb_lowpt"],
                }
            )
            triggers_dict["tauhh_minus"]["triggers"].update(
                {
                    "PNetBB (low pt)": HLTs["pnetbb_lowpt"],
                }
            )
        return triggers_dict

    # def plot_channel(self):
    #     plt.rcParams.update({"font.size": 14})

    #     plot_vars = [
    #         (mhh, "mhh", r"$m_{HH}$ [GeV]", np.linspace(250, 1500, 30)),
    #         (hbbpt, "hbbpt", r"Hbb $p_{T}$ [GeV]", np.linspace(230, 500, 20)),
    #         (httpt, "httpt", r"H$\tau\tau$ $p_{T}$ [GeV]", np.linspace(230, 500, 20)),
    #     ]

    #     for cat, vals in triggers_dict.items():
    #         for kinvar, kinname, kinlabel, bins in plot_vars:
    #             (mask, label, triggers) = vals.values()

    #             fig, (ax, rax) = plt.subplots(
    #                 2,
    #                 1,
    #                 figsize=(9, 11),
    #                 gridspec_kw=dict(height_ratios=[4, 1], hspace=0.07),
    #                 sharex=True,
    #             )

    #             hists = {
    #                 "Preselection": np.histogram(kinvar[mask], bins=bins, weights=weights[mask])
    #             }
    #             ratios = {}

    #             hep.histplot(
    #                 hists["Preselection"],
    #                 yerr=False,
    #                 label="Preselection",
    #                 ax=ax,
    #             )

    #             colours = plt.cm.tab20.colors[1:]

    #             for key, c in zip(triggers.keys(), colours):
    #                 # print(key)
    #                 hists[key] = np.histogram(
    #                     kinvar[mask & triggers[key]],
    #                     bins=bins,
    #                     weights=weights[mask & triggers[key]],
    #                 )
    #                 ratios[key] = hists[key][0] / hists["Preselection"][0]

    #                 hep.histplot(
    #                     hists[key],
    #                     yerr=False,
    #                     label=key,
    #                     ax=ax,
    #                     color=c,
    #                 )

    #                 hep.histplot(
    #                     (ratios[key], bins),
    #                     yerr=False,
    #                     label=key,
    #                     ax=rax,
    #                     histtype="errorbar",
    #                     color=c,
    #                     # markersize=20,
    #                     linestyle="--",
    #                     # markeredgewidth=3,
    #                     # fillstyle='none',
    #                 )

    #             ax.set_ylabel("Events [A.U.]")
    #             ax.legend()
    #             ax.set_title(label)
    #             ax.set_xlim(bins[0], bins[-1])
    #             ax.set_ylim(0)

    #             # rax.legend()
    #             rax.grid(axis="y")
    #             rax.set_xlabel(kinlabel)
    #             rax.set_ylabel("Triggered / Preselection")

    #             ylims = [0.5, 1] if (cat.endswith("minus") and kinname != "mhh") else [0, 1]
    #             rax.set_ylim(ylims)

    #             hep.cms.label(ax=ax, data=False, year=year, com="13.6")

    #             plt.savefig(plot_dir / f"{kinname}_{cat}.pdf", bbox_inches="tight")
    #             plt.savefig(plot_dir / f"{kinname}_{cat}.png", bbox_inches="tight")
    #             plt.show()


if __name__ == "__main__":
    # Load the data
    years = ["2022", "2022EE", "2023", "2023BPix"]
    for year in years:
        for sig_key in SIGNALS:
            analyser = Analyser(year, sig_key)
            analyser.load_data()

            break
