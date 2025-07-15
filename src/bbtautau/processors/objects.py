"""
Object definitions.

Author(s): Cristina Suarez, Raghav Kansal
"""

from __future__ import annotations

import awkward as ak
import numpy as np
from boostedhh.processors.objects import jetid_v12
from boostedhh.processors.utils import PDGID
from coffea.nanoevents.methods.nanoaod import (
    ElectronArray,
    FatJetArray,
    JetArray,
    MuonArray,
    TauArray,
)

from bbtautau.HLTs import HLTs


def trig_match_sel(events, leptons, trig_leptons, year, trigger, filterbit, ptcut, trig_dR=0.2):
    """
    Returns selection for leptons which are trigger matched to the specified trigger.
    """
    trigger = HLTs.hlts_by_type(year, trigger, hlt_prefix=False)[0]  # picking first trigger in list
    trig_fired = events.HLT[trigger]
    # print(f"{trigger} rate: {ak.mean(trig_fired)}")

    filterbit = 2**filterbit

    pass_trig = (trig_leptons.filterBits & filterbit) == filterbit
    trig_l = trig_leptons[pass_trig]
    trig_l_matched = ak.any(leptons.metric_table(trig_l) < trig_dR, axis=2)
    trig_l_sel = trig_fired & trig_l_matched & (leptons.pt > ptcut)
    return trig_l_sel


def get_ak8jets(fatjets: FatJetArray):
    """
    Add extra variables to FatJet collection
    """
    fatjets["t32"] = ak.nan_to_num(fatjets.tau3 / fatjets.tau2, nan=-1.0)
    fatjets["t21"] = ak.nan_to_num(fatjets.tau2 / fatjets.tau1, nan=-1.0)

    fatjets["pt_raw"] = (1 - fatjets.rawFactor) * fatjets.pt
    fatjets["mass_raw"] = (1 - fatjets.rawFactor) * fatjets.mass

    fatjets["globalParT_QCD"] = (
        fatjets.globalParT_QCD0HF + fatjets.globalParT_QCD1HF + fatjets.globalParT_QCD2HF
    )
    fatjets["globalParT_Top"] = fatjets.globalParT_TopW + fatjets.globalParT_TopbW

    fatjets["particleNetLegacy_XbbvsQCD"] = fatjets.particleNetLegacy_Xbb / (
        fatjets.particleNetLegacy_Xbb + fatjets.particleNetLegacy_QCD
    )
    fatjets["globalParT_XbbvsQCD"] = fatjets.globalParT_Xbb / (
        fatjets.globalParT_Xbb + fatjets["globalParT_QCD"]
    )
    fatjets["globalParT_XbbvsQCDTop"] = fatjets.globalParT_Xbb / (
        fatjets.globalParT_Xbb + fatjets["globalParT_QCD"] + fatjets["globalParT_Top"]
    )

    for tautau in ["tauhtauh", "tauhtaue", "tauhtaum"]:
        fatjets[f"globalParT_X{tautau}vsQCD"] = fatjets[f"globalParT_X{tautau}"] / (
            fatjets[f"globalParT_X{tautau}"] + fatjets["globalParT_QCD"]
        )
        fatjets[f"globalParT_X{tautau}vsQCDTop"] = fatjets[f"globalParT_X{tautau}"] / (
            fatjets[f"globalParT_X{tautau}"] + fatjets["globalParT_QCD"] + fatjets["globalParT_Top"]
        )

    fatjets["globalParT_massResCorr"] = fatjets.globalParT_massRes
    fatjets["globalParT_massVisCorr"] = fatjets.globalParT_massVis
    fatjets["globalParT_massResApplied"] = (
        fatjets.globalParT_massRes * (1 - fatjets.rawFactor) * fatjets.mass
    )
    fatjets["globalParT_massVisApplied"] = (
        fatjets.globalParT_massVis * (1 - fatjets.rawFactor) * fatjets.mass
    )
    return fatjets


# ak8 jet definition
def good_ak8jets(
    fatjets: FatJetArray,
    object_pt: float,  # select objects based on this
    pt: float,  # make event selections based on this  # noqa: ARG001
    eta: float,
    msd: float,  # noqa: ARG001
    mreg: float,  # noqa: ARG001
    nano_version: str,  # noqa: ARG001
    mreg_str: str = "particleNet_mass_legacy",  # noqa: ARG001
):
    # if nano_version.startswith("v12"):
    #     jetidtight, jetidtightlepveto = jetid_v12(fatjets)  # v12 jetid fix
    # else:
    #     raise NotImplementedError(f"Jet ID fix not implemented yet for {nano_version}")

    # Data does not have .neHEF etc. fields for fatjets, so above recipe doesn't work
    # Either way, doesn't matter since we only use tightID, and it is correct for eta < 2.7
    jetidtight = fatjets.isTight

    fatjet_sel = (
        jetidtight
        & (fatjets.pt > object_pt)
        & (abs(fatjets.eta) < eta)
        # & ((fatjets.msoftdrop > msd) | (fatjets[mreg_str] > mreg))
    )
    return fatjets[fatjet_sel]


def good_ak4jets(jets: JetArray, nano_version: str):
    if nano_version.startswith("v12"):
        jetidtight, jetidtightlepveto = jetid_v12(jets)  # v12 jetid fix
    else:
        raise NotImplementedError(f"Jet ID fix not implemented yet for {nano_version}")
    jet_sel = (jets.pt > 15) & (np.abs(jets.eta) < 4.7) & jetidtight & jetidtightlepveto

    return jets[jet_sel]


"""
Trigger quality bits in NanoAOD v12
0 => CaloIdL_TrackIdL_IsoVL,
1 => 1e (WPTight),
2 => 1e (WPLoose),
3 => OverlapFilter PFTau,
4 => 2e,
5 => 1e-1mu,
6 => 1e-1tau,
7 => 3e,
8 => 2e-1mu,
9 => 1e-2mu,
10 => 1e (32_L1DoubleEG_AND_L1SingleEGOr),
11 => 1e (CaloIdVT_GsfTrkIdT),
12 => 1e (PFJet),
13 => 1e (Photon175_OR_Photon200) for Electron;
"""


def good_electrons(events, leptons: ElectronArray, year: str):
    # from https://indico.cern.ch/event/1495537/contributions/6355656/attachments/3012754/5312393/2025.02.11_Run3HHbbtautau_CMSweek.pdf
    trigobj = events.TrigObj

    # baseline kinematic selection
    lsel = (
        leptons.mvaIso_WP90
        & (leptons.pt > 20)
        & (abs(leptons.eta) < 2.5)
        & (abs(leptons.dz) < 0.2)
        & (abs(leptons.dxy) < 0.045)
    )
    leptons = leptons[lsel]

    # Trigger: (filterbit, ptcut for matched lepton)
    triggers = {"EGamma": (1, 31), "ETau": (6, 25)}
    trig_leptons = trigobj[trigobj.id == PDGID.e]

    TrigMatchDict = {
        f"ElectronTrigMatch{trigger}": trig_match_sel(
            events, leptons, trig_leptons, year, trigger, filterbit, ptcut
        )
        for trigger, (filterbit, ptcut) in triggers.items()
    }

    return leptons, TrigMatchDict


"""
Trigger quality bits in NanoAOD v12
0 => TrkIsoVVL,
1 => Iso,
2 => OverlapFilter PFTau,
3 => 1mu,
4 => 2mu,
5 => 1mu-1e,
6 => 1mu-1tau,
7 => 3mu,
8 => 2mu-1e,
9 => 1mu-2e,
10 => 1mu (Mu50),
11 => 1mu (Mu100),
12 => 1mu-1photon for Muon;
"""


def good_muons(events, leptons: MuonArray, year: str):
    # from https://indico.cern.ch/event/1495537/contributions/6355656/attachments/3012754/5312393/2025.02.11_Run3HHbbtautau_CMSweek.pdf
    trigobj = events.TrigObj

    lsel = (
        leptons.tightId
        & (leptons.pt > 20)
        & (abs(leptons.eta) < 2.4)
        & (abs(leptons.dz) < 0.2)
        & (abs(leptons.dxy) < 0.045)
    )
    leptons = leptons[lsel]

    # Trigger: (filterbit, ptcut for matched lepton)
    triggers = {"Muon": (3, 26), "MuonTau": (6, 22)}
    trig_leptons = trigobj[trigobj.id == PDGID.mu]

    TrigMatchDict = {
        f"MuonTrigMatch{trigger}": trig_match_sel(
            events, leptons, trig_leptons, year, trigger, filterbit, ptcut
        )
        for trigger, (filterbit, ptcut) in triggers.items()
    }

    return leptons, TrigMatchDict


"""
Trigger quality bits in NanoAOD v12
0 => LooseChargedIso,
1 => MediumChargedIso,
2 => TightChargedIso,
3 => DeepTau,
4 => TightID OOSC photons,
5 => HPS,
6 => charged iso di-tau,
7 => deeptau di-tau,
8 => e-tau,
9 => mu-tau,
10 => single-tau/tau+MET,
11 => run 2 VBF+ditau,
12 => run 3 VBF+ditau,
13 => run 3 double PF jets + ditau,
14 => di-tau + PFJet,
15 => Displaced Tau,
16 => Monitoring,
17 => regional paths,
18 => L1 seeded paths,
19 => 1 prong tau paths for Tau;
"""


def good_taus(events, leptons: TauArray, year: str):
    # from https://indico.cern.ch/event/1495537/contributions/6355656/attachments/3012754/5312393/2025.02.11_Run3HHbbtautau_CMSweek.pdf
    trigobj = events.TrigObj

    lsel = (
        (leptons.idDeepTau2018v2p5VSjet >= 5)
        # & (leptons.idDeepTau2018v2p5VSe >= 3)
        & (leptons.pt > 20)
        & (abs(leptons.eta) < 2.5)
        & (abs(leptons.dz) < 0.2)
    )
    leptons = leptons[lsel]

    # Trigger: (filterbit, ptcut for matched lepton)
    triggers = {"SingleTau": (10, 185), "DiTau": (7, 37), "ETau": (8, 32), "MuonTau": (9, 30)}
    trig_leptons = trigobj[trigobj.id == PDGID.tau]

    TrigMatchDict = {
        f"TauTrigMatch{trigger}": trig_match_sel(
            events, leptons, trig_leptons, year, trigger, filterbit, ptcut
        )
        for trigger, (filterbit, ptcut) in triggers.items()
    }

    return leptons, TrigMatchDict


"""
Trigger quality bits in NanoAOD v12
0 => HLT_AK8PFJetX_SoftDropMass40_PFAK8ParticleNetTauTau0p30,
1 => hltAK8SinglePFJets230SoftDropMass40PNetTauTauTag0p03 for BoostedTau;
"""


def good_boostedtaus(events, taus: TauArray):  # noqa: ARG001
    # from https://indico.cern.ch/event/1495537/contributions/6355656/attachments/3012754/5312393/2025.02.11_Run3HHbbtautau_CMSweek.pdf

    tau_sel = (taus.pt > 20) & (abs(taus.eta) < 2.5)
    return taus[tau_sel]

##add CA Method by jin
def delta_r(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = phi1 - phi2
    dphi = ak.where(dphi > np.pi, dphi - 2*np.pi, dphi)
    dphi = ak.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
    return np.sqrt(deta**2 + dphi**2)

# def get_Matching_old(fatjets: FatJetArray, taus: TauArray):

#     if taus is not None:
#         fatjet_boostedtau_pairs = ak.cartesian([fatjets, taus], nested=True)
#         fatjets_in_pairs = fatjet_boostedtau_pairs["0"]
#         boostedtaus_in_pairs = fatjet_boostedtau_pairs["1"]

#         dR = delta_r(fatjets_in_pairs.eta, fatjets_in_pairs.phi, boostedtaus_in_pairs.eta, boostedtaus_in_pairs.phi)

#         close_matches = dR < 0.8
#         num_close_matches = ak.sum(close_matches, axis = -1)
#         fatjets["CA_matched_2BoostedTaus"] = ak.where(num_close_matches >= 2, 1, 0)

#         matched_fatjet_mask = fatjets["CA_matched_2BoostedTaus"] == 1
#         has_any_match = ak.any(matched_fatjet_mask, axis=1)
        
#         # 对有匹配的事件进行处理
#         tautau_scores = fatjets.globalParT_XtauhtauhvsQCD
#         valid_scores = ak.where(matched_fatjet_mask, tautau_scores, -1)
#         computed_best_idx = ak.argmax(valid_scores, axis=1)
        
#         # 使用循环来避免复杂的索引操作
#         best_fatjet_matches = []
#         for i in range(len(close_matches)):
#             if has_any_match[i]:
#                 best_fatjet_matches.append(close_matches[i][computed_best_idx[i]])
#             else:
#                 # 对于没有匹配的事件，创建一个全False的数组
#                 best_fatjet_matches.append(ak.zeros_like(close_matches[i][0], dtype=bool))
        
#         best_fatjet_matches = ak.Array(best_fatjet_matches)
#         matched_taus = taus[best_fatjet_matches]
        
#         # 排序和选择前两个
#         sorted_indices = ak.argsort(matched_taus.pt, axis=1, ascending=False)
#         sorted_taus = matched_taus[sorted_indices]
#         top2_taus = sorted_taus[:, :2]
        
#         # 计算结果
#         computed_pt_sum = ak.sum(top2_taus.pt, axis=1)
        
#         # 计算 tau 索引
#         matched_tau_indices = ak.local_index(taus, axis=1)[best_fatjet_matches]
#         sorted_tau_indices = matched_tau_indices[sorted_indices]
#         computed_tau_indices = sorted_tau_indices[:, :2]
        
#         # 只对有匹配的事件保留计算结果
#         event_matched_tau_pt_sum = ak.where(has_any_match, computed_pt_sum, 0)
#         final_tau_indices = ak.where(has_any_match[:, np.newaxis], computed_tau_indices, -1)
#         best_fatjet_idx = ak.where(has_any_match, computed_best_idx, -1)

#     else:
#         fatjets["CA_matched_2BoostedTaus"] = ak.full_like(fatjets.pt, -1, dtype=int)
#         event_matched_tau_pt_sum = ak.full_like(fatjets.pt[:, 0], -999, dtype=float)
#         final_tau_indices = ak.full_like(fatjets.pt[:, 0:2], -999, dtype=int)
#         best_fatjet_idx = ak.full_like(fatjets.pt[:, 0], -999, dtype=int)


def get_Matching(fatjets: FatJetArray, taus: TauArray):

    if taus is not None:
        
        n_events = len(fatjets)
        n_fatjets = ak.num(fatjets, axis=1)
        n_taus = ak.num(taus, axis=1)
        
        
        has_fatjets = n_fatjets > 0
        has_taus = n_taus > 0
        can_match = has_fatjets & has_taus
        
        
        fatjets["CA_matched_2BoostedTaus"] = ak.full_like(fatjets.pt, -1, dtype=int)
        event_matched_tau_pt_sum = ak.zeros_like(n_fatjets, dtype=float)  
        final_tau_indices = ak.full_like(ak.broadcast_arrays(n_fatjets, ak.Array([[0, 0]]))[1], -1, dtype=int)
        best_fatjet_idx = ak.full_like(n_fatjets, -1, dtype=int)  
        
        
        if ak.any(can_match):
            
            fatjet_boostedtau_pairs = ak.cartesian([fatjets, taus], nested=True)
            fatjets_in_pairs = fatjet_boostedtau_pairs["0"]
            boostedtaus_in_pairs = fatjet_boostedtau_pairs["1"]

            dR = delta_r(fatjets_in_pairs.eta, fatjets_in_pairs.phi, boostedtaus_in_pairs.eta, boostedtaus_in_pairs.phi)

            close_matches = dR < 0.8
            num_close_matches = ak.sum(close_matches, axis = -1)
            
            
            ca_matched = ak.where(num_close_matches >= 2, 1, 0)
            fatjets["CA_matched_2BoostedTaus"] = ak.where(has_fatjets[:, np.newaxis], ca_matched, -1)

            matched_fatjet_mask = fatjets["CA_matched_2BoostedTaus"] == 1
            has_any_match = ak.any(matched_fatjet_mask, axis=1)
            
           
            tautau_scores = fatjets.globalParT_XtauhtauhvsQCD
            valid_scores = ak.where(matched_fatjet_mask, tautau_scores, -1)
            computed_best_idx = ak.argmax(valid_scores, axis=1)
            
            
            best_fatjet_matches = []
            for i in range(n_events):
                if has_any_match[i]:
                    best_fatjet_matches.append(close_matches[i][computed_best_idx[i]])
                else:
                    if n_taus[i] > 0:
                        best_fatjet_matches.append(ak.Array([False] * n_taus[i]))
                    else:
                        best_fatjet_matches.append(ak.Array([]))
            
            best_fatjet_matches = ak.Array(best_fatjet_matches)
            matched_taus = taus[best_fatjet_matches]
            
            has_matched_taus = ak.num(matched_taus, axis=1) > 0
            
            computed_pt_sum = ak.zeros_like(n_fatjets, dtype=float)  # 用 n_fatjets 数组
            computed_tau_indices = ak.full_like(ak.broadcast_arrays(n_fatjets, ak.Array([[0, 0]]))[1], -1, dtype=int)
            

            if ak.any(has_matched_taus):
         
                sorted_indices = ak.argsort(matched_taus.pt, axis=1, ascending=False)
                sorted_taus = matched_taus[sorted_indices]
                
                padded_taus = ak.pad_none(sorted_taus, 2, axis=1)
                top2_taus = padded_taus[:, :2]
                
                valid_pt = ak.fill_none(top2_taus.pt, 0)
                temp_pt_sum = ak.sum(valid_pt, axis=1)
                
                matched_tau_indices = ak.local_index(taus, axis=1)[best_fatjet_matches]
                sorted_tau_indices = matched_tau_indices[sorted_indices]
                padded_indices = ak.pad_none(sorted_tau_indices, 2, axis=1)
                temp_tau_indices = ak.fill_none(padded_indices[:, :2], -1)
                
                computed_pt_sum = ak.where(has_matched_taus, temp_pt_sum, 0)
                computed_tau_indices = ak.where(has_matched_taus[:, np.newaxis], temp_tau_indices, -1)
            
            event_matched_tau_pt_sum = ak.where(has_any_match, computed_pt_sum, 0)
            final_tau_indices = ak.where(has_any_match[:, np.newaxis], computed_tau_indices, -1)
            best_fatjet_idx = ak.where(has_any_match, computed_best_idx, -1)


    else:
        n_fatjets = ak.num(fatjets, axis=1)
        fatjets["CA_matched_2BoostedTaus"] = ak.full_like(fatjets.pt, -1, dtype=int)
        event_matched_tau_pt_sum = ak.full_like(n_fatjets, -999, dtype=float)
        final_tau_indices = ak.full_like(ak.broadcast_arrays(n_fatjets, ak.Array([[0, 0]]))[1], -999, dtype=int)
        best_fatjet_idx = ak.full_like(n_fatjets, -999, dtype=int)

    return fatjets, event_matched_tau_pt_sum, final_tau_indices, best_fatjet_idx
