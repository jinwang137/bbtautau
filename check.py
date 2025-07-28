import coffea.nanoevents.methods.nanoaod
print(dir(coffea.nanoevents.methods.nanoaod))


# import awkward as ak
# import numpy as np

# def delta_r(eta1, phi1, eta2, phi2):
#     # deta = eta1 - eta2
#     # dphi = phi1 - phi2
#     # dphi = ak.where(dphi > np.pi, dphi - 2 * np.pi, dphi)
#     # dphi = ak.where(dphi < -np.pi, dphi + 2 * np.pi, dphi)
#     # return np.sqrt(deta ** 2 + dphi ** 2)
#     return eta2

# fatjets = ak.Array([
#     [{"eta": 1.0, "phi": 2.0, "mass": 1.0}, {"eta": 3.0, "phi": 4.0, "mass": 1.0}],  # 事件0
#     [{"eta": 0.9, "phi": 1.8, "mass": 2.0}, {"eta": -0.8, "phi": 1.8, "mass": 1.0}]   # 事件1
# ])

# taus = ak.Array([
#     [{"eta": 5.0, "phi": 6.0, "pt": 1.0}, {"eta": 7.0, "phi": 8.0, "pt": 1.0},  {"eta": 9.0, "phi": 10.0, "pt": 1.0}],  # 事件0
#     [{"eta": -1.2, "phi": 1.9, "pt": 1.0}, {"eta": 0.9, "phi": 1.78, "pt": 1.0}, {"eta": 0.5, "phi": 1.82, "pt": 1.0}]   # 事件1
# ])

# fatjet_boostedtau_pairs = ak.cartesian([fatjets, taus], nested=True)
# fatjets_in_pairs = fatjet_boostedtau_pairs["0"]
# boostedtaus_in_pairs = fatjet_boostedtau_pairs["1"]

# dR = delta_r(fatjets_in_pairs.eta, fatjets_in_pairs.phi, boostedtaus_in_pairs.eta, boostedtaus_in_pairs.phi)

# print(fatjet_boostedtau_pairs)

# print(boostedtaus_in_pairs)

# print(dR)

# close_matches = dR < 6

# matched_taus_per_fatjet = boostedtaus_in_pairs[close_matches]
# n_matched = ak.num(matched_taus_per_fatjet, axis=-1)
# sorted_indices = ak.argsort(matched_taus_per_fatjet.eta, axis=-1, ascending=False)
# sorted_taus = matched_taus_per_fatjet[sorted_indices]
# top2_taus = ak.pad_none(sorted_taus, 2, axis=-1)[..., :2]

# tau0_eta = ak.fill_none(top2_taus.eta[..., 0], -999)
# tau1_eta = ak.fill_none(top2_taus.eta[..., 1], -999)
# tau0_phi = ak.fill_none(top2_taus.phi[..., 0], -999)
# tau1_phi = ak.fill_none(top2_taus.phi[..., 1], -999)

# tau0_pt = ak.fill_none(top2_taus.pt[..., 0], -999)
# tau1_pt = ak.fill_none(top2_taus.pt[..., 1], -999)

# def CA_got(met_pt, met_phi, fatjets_mass, tau0_eta, tau1_eta, tau0_phi, tau1_phi, tau0_pt, tau1_pt):
#     invalid = (
#         (met_pt == -999)
#         | (met_phi == -999)
#         | (fatjets_mass == -999)
#         | (tau0_eta == -999)
#         | (tau1_eta == -999)
#         | (tau0_phi == -999)
#         | (tau1_phi == -999)
#         | (tau0_pt == -999)
#         | (tau1_pt == -999)
#     )

#     dphi1 = met_phi - tau0_phi
#     dphi0 = tau1_phi - met_phi
#     dphi = tau0_phi - tau1_phi
    
#     sin_dphi0 = np.sin(dphi0)
#     sin_dphi1 = np.sin(dphi1)
#     sin_dphi = np.sin(dphi)
    
#     pmet_tau0 = np.abs(met_pt * sin_dphi0 / sin_dphi)
#     pmet_tau1 = np.abs(met_pt * sin_dphi1 / sin_dphi)

#     denom = np.sqrt(np.abs(tau0_pt/(tau0_pt + pmet_tau0)) * np.abs(tau1_pt/(tau1_pt + pmet_tau1)))
#     denom = ak.where(denom == 0, 1, denom)

#     mass = fatjets_mass / denom
#     mass = ak.where(invalid, -999, mass)
#     return mass

# met_pt = [1.0,2.0]
# met_phi = [1.0,1.8]
# fatjets_mass = fatjets.mass
# mass_boostedtau = CA_got(met_pt, met_phi, fatjets_mass, tau0_eta, tau1_eta, tau0_phi, tau1_phi, tau0_pt, tau1_pt)

# print(matched_taus_per_fatjet)
# print(n_matched)
# # print(sorted_indices)
# # print(sorted_taus)
# # print(top2_taus)
# # print(tau0_eta)
# # print(tau1_eta)
# # print(fatjets.eta)

# print(mass_boostedtau)