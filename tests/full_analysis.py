import os
import glob
import numpy as np
import uproot
import time
from collections import OrderedDict
import argparse
import multiprocessing

import hepaccelerate
from hepaccelerate.utils import Histogram

import math
import numba
from numba import cuda

save_arrays = True

def create_datastructure(ismc):
    datastructures = {
        "Muon": [
            ("Muon_pt", "float32"),
            ("Muon_eta", "float32"),
            ("Muon_phi", "float32"),
            ("Muon_mass", "float32"),
            ("Muon_charge", "int32"),
            ("Muon_pfRelIso03_all", "float32"),
            ("Muon_tightId", "bool")
        ],
        "Electron": [
            ("Electron_pt", "float32"),
            ("Electron_eta", "float32"),
            ("Electron_phi", "float32"),
            ("Electron_mass", "float32"),
            ("Electron_charge", "int32"),
            ("Electron_pfRelIso03_all", "float32"),
            ("Electron_pfId", "bool")
        ],
        "Jet": [
            ("Jet_pt", "float32"),
            ("Jet_eta", "float32"),
            ("Jet_phi", "float32"),
            ("Jet_mass", "float32"),
            ("Jet_btag", "float32"),
            ("Jet_puId", "bool"),
        ],

        "EventVariables": [
            ("HLT_IsoMu24", "bool"),
            ('MET_pt', 'float32'),
            ('MET_phi', 'float32'),
            ('MET_sumet', 'float32'),
            ('MET_significance', 'float32'),
            ('MET_CovXX', 'float32'),
            ('MET_CovXY', 'float32'),
            ('MET_CovYY', 'float32'),
        ]
    }
    if ismc:
        datastructures["Muon"] += [("Muon_genPartIdx", "int32")]
        datastructures["Electron"] += [("Electron_genPartIdx", "int32")]
        
        datastructures["GenPart"] = [
            ('GenPart_pt', 'float32'),
            ('GenPart_eta', 'float32'),
            ('GenPart_phi', 'float32'),
            ('GenPart_mass', 'float32'),
            ('GenPart_pdgId', 'int32'),
            ('GenPart_status', 'int32'),
        ]

    return datastructures

def get_selected_muons(muons, pt_cut_leading, pt_cut_subleading, aeta_cut, iso_cut):
    passes_iso = muons.pfRelIso03_all < iso_cut
    passes_id = muons.tightId == True
    passes_subleading_pt = muons.pt > pt_cut_subleading
    passes_leading_pt = muons.pt > pt_cut_leading
    passes_aeta = NUMPY_LIB.abs(muons.eta) < aeta_cut
    
    selected_muons =  (
        passes_iso & passes_id &
        passes_subleading_pt & passes_aeta
    )
    
    selected_muons_leading = selected_muons & passes_leading_pt
    
    evs_all = NUMPY_LIB.ones(muons.numevents(), dtype=bool)
    
    selected_events = ha.sum_in_offsets(
        muons, selected_muons_leading, evs_all, muons.masks["all"]
    ) >= 1
    
    return selected_muons, selected_events

def get_selected_electrons(electrons, pt_cut_leading, pt_cut_subleading, aeta_cut, iso_cut):
    passes_iso = electrons.pfRelIso03_all < iso_cut
    passes_id = electrons.pfId == True
    passes_subleading_pt = electrons.pt > pt_cut_subleading
    passes_leading_pt = electrons.pt > pt_cut_leading
    passes_aeta = NUMPY_LIB.abs(electrons.eta) < aeta_cut
    
    evs_all = NUMPY_LIB.ones(electrons.numevents(), dtype=bool)
    els_all = NUMPY_LIB.ones(electrons.numobjects(), dtype=bool)
    
    selected_electrons =  (
        passes_iso & passes_id &
        passes_subleading_pt & passes_aeta
    )
    
    selected_electrons_leading = selected_electrons & passes_leading_pt
    
    selected_events = ha.sum_in_offsets(
        electrons, selected_electrons_leading, evs_all, electrons.masks["all"]
    ) >= 0
        
    return selected_electrons, selected_events

def apply_lepton_corrections(leptons, mask_leptons, lepton_weights):
    
    corrs = NUMPY_LIB.zeros_like(leptons.pt)
    ha.get_bin_contents(leptons.pt, lepton_weights[:, 0], lepton_weights[:-1, 1], corrs)
    
    #multiply the per-lepton weights for each event
    all_events = NUMPY_LIB.ones(leptons.numevents(), dtype=NUMPY_LIB.bool)
    corr_per_event = ha.prod_in_offsets(
        leptons, corrs, all_events, mask_leptons
    )
    
    return corr_per_event

def apply_jec(jets, jecs):
    corrs = NUMPY_LIB.zeros_like(jets.pt)
    ha.get_bin_contents(jets.pt, jecs[:, 0], jecs[:-1, 1], corrs)
    return NUMPY_LIB.abs(corrs)

def select_jets(jets, mu, el, selected_muons, selected_electrons, pt_cut, aeta_cut, jet_lepton_dr_cut, btag_cut):
    passes_id = jets.puId == True
    passes_aeta = NUMPY_LIB.abs(jets.eta) < aeta_cut
    passes_pt = jets.pt > pt_cut
    
    selected_jets = passes_id & passes_aeta & passes_pt

    jets_pass_dr_mu = ha.mask_deltar_first(
        jets, selected_jets, mu,
        selected_muons, jet_lepton_dr_cut)
        
    jets_pass_dr_el = ha.mask_deltar_first(
        jets, selected_jets, el,
        selected_electrons, jet_lepton_dr_cut)
    
    selected_jets_no_lepton = selected_jets & jets_pass_dr_mu & jets_pass_dr_el
    
    return selected_jets_no_lepton

def fill_histograms_several(hists, systematic_name, histname_prefix, variables, mask, weights, use_cuda):
    all_arrays = []
    all_bins = []
    num_histograms = len(variables)

    for array, varname, bins in variables:
        if (len(array) != len(variables[0][0]) or
            len(array) != len(mask) or
            len(array) != len(weights["nominal"])):
            raise Exception("Data array {0} is of incompatible size".format(varname))
        all_arrays += [array]
        all_bins += [bins]

    max_bins = max([b.shape[0] for b in all_bins])
    stacked_array = NUMPY_LIB.stack(all_arrays, axis=0)
    stacked_bins = np.concatenate(all_bins)
    nbins = np.array([len(b) for b in all_bins])
    nbins_sum = np.cumsum(nbins)
    nbins_sum = np.insert(nbins_sum, 0, [0])

    for weight_name, weight_array in weights.items():
        if use_cuda:
            nblocks = 32
            out_w = NUMPY_LIB.zeros((len(variables), nblocks, max_bins), dtype=NUMPY_LIB.float32)
            out_w2 = NUMPY_LIB.zeros((len(variables), nblocks, max_bins), dtype=NUMPY_LIB.float32)
            ha.fill_histogram_several[nblocks, 1024](
                stacked_array, weight_array, mask, stacked_bins,
                NUMPY_LIB.array(nbins), NUMPY_LIB.array(nbins_sum), out_w, out_w2
            )
            cuda.synchronize()

            out_w = out_w.sum(axis=1)
            out_w2 = out_w2.sum(axis=1)

            out_w = NUMPY_LIB.asnumpy(out_w)
            out_w2 = NUMPY_LIB.asnumpy(out_w2)
        else:
            out_w = NUMPY_LIB.zeros((len(variables), max_bins), dtype=NUMPY_LIB.float32)
            out_w2 = NUMPY_LIB.zeros((len(variables), max_bins), dtype=NUMPY_LIB.float32)
            ha.fill_histogram_several(
                stacked_array, weight_array, mask, stacked_bins,
                nbins, nbins_sum, out_w, out_w2
            )

        out_w_separated = [out_w[i, 0:nbins[i]-1] for i in range(num_histograms)]
        out_w2_separated = [out_w2[i, 0:nbins[i]-1] for i in range(num_histograms)]

        for ihist in range(num_histograms):
            hist_name = histname_prefix + variables[ihist][1]
            bins = variables[ihist][2]
            target_histogram = Histogram(out_w_separated[ihist], out_w2_separated[ihist], bins)
            target = {weight_name: target_histogram}
            update_histograms_systematic(hists, hist_name, systematic_name, target)

def update_histograms_systematic(hists, hist_name, systematic_name, target_histogram):

    if hist_name not in hists:
        hists[hist_name] = {}

    if systematic_name[0] == "nominal" or systematic_name == "nominal":
        hists[hist_name].update(target_histogram)
    else:
        if systematic_name[1] == "":
            syst_string = systematic_name[0]
        else:
            syst_string = systematic_name[0] + "__" + systematic_name[1]
        target_histogram = {syst_string: target_histogram["nominal"]}
        hists[hist_name].update(target_histogram)

def run_analysis(dataset, out, njec, use_cuda):
    hists = {}
    histo_bins = {
        "nmu": np.array([0,1,2,3], dtype=np.float32),
        "njet": np.array([0,1,2,3,4,5,6,7], dtype=np.float32),
        "mu_pt": np.linspace(0, 300, 100),
        "mu_eta": np.linspace(-5, 5, 100),
        "mu_phi": np.linspace(-5, 5, 100),
        "mu_iso": np.linspace(0, 1, 100),
        "mu_charge": np.array([-1, 0, 1], dtype=np.float32),
        "met_pt": np.linspace(0,200,100),
        "jet_pt": np.linspace(0,200,100),
    }

    t0 = time.time()
 
    i = 0
 
    mu = dataset.structs["Muon"][i]
    el = dataset.structs["Electron"][i]
    jets = dataset.structs["Jet"][i]
    evvars = dataset.eventvars[i]

    mu.hepaccelerate_backend = ha
    el.hepaccelerate_backend = ha
    jets.hepaccelerate_backend = ha
    
    evs_all = NUMPY_LIB.ones(dataset.numevents(), dtype=NUMPY_LIB.bool)

    sel_mu, sel_ev_mu = get_selected_muons(mu, 30, 20, 2.4, 0.3)
    mu.masks["selected"] = sel_mu
    sel_el, sel_ev_el = get_selected_electrons(el, 30, 20, 2.4, 0.3)
    el.masks["selected"] = sel_el
    
    nmu = ha.sum_in_offsets(
        mu, mu.masks["selected"], evs_all, mu.masks["all"], dtype=NUMPY_LIB.int32
    )
    nel = ha.sum_in_offsets(
        el, el.masks["selected"], evs_all, el.masks["all"], dtype=NUMPY_LIB.int32
    )
        
    #get contiguous arrays of the first two muons for all events
    mu1 = mu.select_nth(0, object_mask=sel_mu)
    mu2 = mu.select_nth(1, object_mask=sel_mu)
    el1 = el.select_nth(0, object_mask=sel_el)
    el2 = el.select_nth(1, object_mask=sel_el)
    
    weight_ev_mu = apply_lepton_corrections(mu, sel_mu, electron_weights)
    weight_ev_el = apply_lepton_corrections(el, sel_el, electron_weights)
   
    weights = {"nominal": weight_ev_mu * weight_ev_el}

    weights_jet = {}
    for k in weights.keys():
        weights_jet[k] = NUMPY_LIB.zeros_like(jets.pt)
        ha.broadcast(weights["nominal"], jets.offsets, weights_jet[k])

    all_jecs = [electron_weights for i in range(njec)]
    
    jets_pt_orig = NUMPY_LIB.copy(jets.pt)

    #per-event histograms
    fill_histograms_several(
        hists, "nominal", "hist__all__",
        [
            (evvars["MET_pt"], "met_pt", histo_bins["met_pt"]),
        ],
        evs_all,
        weights,
        use_cuda,
    )

    print(jets.pt.shape)
    fill_histograms_several(
        hists, "nominal", "hist__all__",
        [
            (jets.pt, "jets_pt", histo_bins["jet_pt"]),
        ],
        jets.masks["all"],
        weights_jet,
        use_cuda,
    )

    #loop over the jet corrections
    for ijec in range(len(all_jecs)):
        #jet_pt_corr = apply_jec(jets, all_jecs[ijec])

        #compute the corrected jet pt        
        #jets.pt = jets_pt_orig * jet_pt_corr

        #get selected jets
        sel_jet = select_jets(jets, mu, el, sel_mu, sel_el, 20, 4.0, 0.3, 0.5)
        
        #compute the number of jets per event 
        njet = ha.sum_in_offsets(
            jets, sel_jet, evs_all, jets.masks["all"], dtype=NUMPY_LIB.int32
        )

        inv_mass_3j = NUMPY_LIB.zeros(jets.numevents(), dtype=NUMPY_LIB.float32)
        best_comb_3j = NUMPY_LIB.zeros((jets.numevents(), 3), dtype=NUMPY_LIB.int32)

        if use_cuda:
            kernels.comb_3_invmass_closest[32,256](jets.pt, jets.eta, jets.phi, jets.mass, jets.offsets, 172.0, inv_mass_3j, best_comb_3j)
            cuda.synchronize()
        else:
            kernels.comb_3_invmass_closest(jets.pt, jets.eta, jets.phi, jets.mass, jets.offsets, 172.0, inv_mass_3j, best_comb_3j)

        best_btag = NUMPY_LIB.zeros(jets.numevents(), dtype=NUMPY_LIB.float32)
        if use_cuda:
            kernels.max_val_comb[32,1024](jets.btag, jets.offsets, best_comb_3j, best_btag)
            cuda.synchronize()
        else:
            kernels.max_val_comb(jets.btag, jets.offsets, best_comb_3j, best_btag)


        #get the events with at least two jets
        sel_ev_jet = (njet >= 2)
        
        selected_events = sel_ev_mu & sel_ev_el & sel_ev_jet

        #get contiguous vectors of the first two jet data
        jet1 = jets.select_nth(0, object_mask=sel_jet)
        jet2 = jets.select_nth(1, object_mask=sel_jet)
        jet3 = jets.select_nth(3, object_mask=sel_jet)
       
        #create a mask vector for the first two jets 
        first_two_jets = NUMPY_LIB.zeros_like(sel_jet)
        inds = NUMPY_LIB.zeros_like(evs_all, dtype=NUMPY_LIB.int32) 
        targets = NUMPY_LIB.ones_like(evs_all, dtype=NUMPY_LIB.int32) 
        inds[:] = 0
        ha.set_in_offsets(first_two_jets, jets.offsets, inds, targets, selected_events, sel_jet)
        inds[:] = 1
        ha.set_in_offsets(first_two_jets, jets.offsets, inds, targets, selected_events, sel_jet)

        #compute the invariant mass of the first two jets
        dijet_inv_mass, dijet_pt = compute_inv_mass(jets, selected_events, sel_jet & first_two_jets, use_cuda)

        sumpt_jets = ha.sum_in_offsets(jets, jets.pt, selected_events, sel_jet)

        #create a keras-like array
        arr = NUMPY_LIB.vstack([
            nmu, nel, njet, dijet_inv_mass, dijet_pt, 
            mu1["pt"], mu1["eta"], mu1["phi"], mu1["charge"], mu1["pfRelIso03_all"],
            mu2["pt"], mu2["eta"], mu2["phi"], mu2["charge"], mu2["pfRelIso03_all"],
            el1["pt"], el1["eta"], el1["phi"], el1["charge"], el1["pfRelIso03_all"],
            el2["pt"], el2["eta"], el2["phi"], el2["charge"], el2["pfRelIso03_all"],
            jet1["pt"], jet1["eta"], jet1["phi"], jet1["btag"],
            jet2["pt"], jet2["eta"], jet2["phi"], jet2["btag"],
            inv_mass_3j, best_btag, sumpt_jets
        ]).T
        
        fill_histograms_several(
            hists, "jec{0}".format(ijec), "hist__nmu1_njet2__",
            [
                (arr[:, 0], "nmu", histo_bins["nmu"]),
                (arr[:, 1], "nel", histo_bins["nmu"]),
                (arr[:, 2], "njet", histo_bins["njet"]),

                (arr[:, 3], "mu1_pt", histo_bins["mu_pt"]),
                (arr[:, 4], "mu1_eta", histo_bins["mu_eta"]),
                (arr[:, 5], "mu1_phi", histo_bins["mu_phi"]),
                (arr[:, 6], "mu1_charge", histo_bins["mu_charge"]),
                (arr[:, 7], "mu1_iso", histo_bins["mu_iso"]),

                (arr[:, 8], "mu2_pt", histo_bins["mu_pt"]),
                (arr[:, 9], "mu2_eta", histo_bins["mu_eta"]),
                (arr[:, 10], "mu2_phi", histo_bins["mu_phi"]),
                (arr[:, 11], "mu2_charge", histo_bins["mu_charge"]),
                (arr[:, 12], "mu2_iso", histo_bins["mu_iso"]),

                (arr[:, 13], "el1_pt", histo_bins["mu_pt"]),
                (arr[:, 14], "el1_eta", histo_bins["mu_eta"]),
                (arr[:, 15], "el1_phi", histo_bins["mu_phi"]),
                (arr[:, 17], "el1_charge", histo_bins["mu_charge"]),
                (arr[:, 18], "el1_iso", histo_bins["mu_iso"]),
                
                (arr[:, 19], "el2_pt", histo_bins["mu_pt"]),
                (arr[:, 20], "el2_eta", histo_bins["mu_eta"]),
                (arr[:, 21], "el2_phi", histo_bins["mu_phi"]),
                (arr[:, 22], "el2_charge", histo_bins["mu_charge"]),
                (arr[:, 23], "el2_iso", histo_bins["mu_iso"]),
            ],
            selected_events,
            weights,
            use_cuda
        )


        #save the array for the first jet correction scenario only
        if save_arrays and ijec == 0:
            outfile_arr = "{0}_arrs.npy".format(out)
            print("Saving array with shape {0} to {1}".format(arr.shape, outfile_arr))
            with open(outfile_arr, "wb") as fi:
                np.save(fi, NUMPY_LIB.asnumpy(arr))

    t1 = time.time()
   
    speed = dataset.numevents() / (t1 - t0) 
    print("run_analysis: {0:.2E} events in {1:.2f} seconds, speed {2:.2E} Hz".format(dataset.numevents(), t1 - t0, speed))
    return

def load_dataset(datapath, cachepath, filenames, ismc, nthreads, skip_cache, do_skim):
    ds = hepaccelerate.Dataset(
        "dataset",
        filenames,
        create_datastructure(ismc),
        datapath=datapath,
        cache_location=cachepath,
        treename="aod2nanoaod/Events",
    )
    
    cache_valid = ds.check_cache()
    
    if skip_cache or not cache_valid:
        t0 = time.time()
        
        #Load the ROOT files
        print("Loading dataset from {0} files".format(len(ds.filenames)))
        ds.preload(nthreads=nthreads, verbose=True)
        ds.make_objects(verbose=True)
        ds.cache_metadata = ds.create_cache_metadata()
        print("Loaded dataset, {0:.2f} MB, {1} files, {2} events".format(ds.memsize() / 1024 / 1024, len(ds.filenames), ds.numevents()))
    
        #Apply a skim on the trigger bit for each file
        if do_skim:
            masks = [v['HLT_IsoMu24']==True for v in ds.eventvars]
            ds.compact(masks)
            print("Applied trigger bit selection skim, {0:.2f} MB, {1} files, {2} events".format(ds.memsize() / 1024 / 1024, len(ds.filenames), ds.numevents()))
    
        print("Saving skimmed data to uncompressed cache")
        ds.to_cache(nthreads=nthreads, verbose=True)
        t1 = time.time()
        speed = ds.numevents() / (t1 - t0)
        print("load_dataset: {0:.2E} events / second".format(speed))
    else:
        t0 = time.time()
        print("Loading from existing cache")
        ds.from_cache(verbose=True)
        t1 = time.time()
        t1 = time.time()
        speed = ds.numevents() / (t1 - t0)
        print("load_dataset: {0:.2E} events / second".format(speed))
    
    #Compute the average length of the data structures
    avg_vec_length = np.mean(np.array([[v["pt"].shape[0] for v in ds.structs[ss]] for ss in ds.structs.keys()]))
    print("Average vector length before merging: {0:.0f}".format(avg_vec_length))
  
    print("Merging arrays from multiple files using awkward") 
    #Now merge all the arrays across the files to have large contiguous data
    ds.merge_inplace()
    
    print("Copying to device")
    #transfer dataset to device (GPU) if applicable
    ds.move_to_device(NUMPY_LIB)
    ds.numpy_lib = NUMPY_LIB

    avg_vec_length = np.mean(np.array([[v["pt"].shape[0] for v in ds.structs[ss]] for ss in ds.structs.keys()]))
    print("Average vector length after merging: {0:.0f}".format(avg_vec_length))

    return ds

def compute_inv_mass(objects, mask_events, mask_objects, use_cuda):
    inv_mass = NUMPY_LIB.zeros(len(mask_events), dtype=np.float32)
    pt_total = NUMPY_LIB.zeros(len(mask_events), dtype=np.float32)
    if use_cuda:
        compute_inv_mass_cudakernel[32, 1024](
            objects.offsets, objects.pt, objects.eta, objects.phi, objects.mass,
            mask_events, mask_objects, inv_mass, pt_total)
        cuda.synchronize()
    else:
        compute_inv_mass_kernel(objects.offsets,
            objects.pt, objects.eta, objects.phi, objects.mass,
            mask_events, mask_objects, inv_mass, pt_total)
    return inv_mass, pt_total

@numba.njit(parallel=True, fastmath=True)
def compute_inv_mass_kernel(offsets, pts, etas, phis, masses, mask_events, mask_objects, out_inv_mass, out_pt_total):
    for iev in numba.prange(offsets.shape[0]-1):
        if mask_events[iev]:
            start = np.uint64(offsets[iev])
            end = np.uint64(offsets[iev + 1])
            
            px_total = np.float32(0.0)
            py_total = np.float32(0.0)
            pz_total = np.float32(0.0)
            e_total = np.float32(0.0)
            
            for iobj in range(start, end):
                if mask_objects[iobj]:
                    pt = pts[iobj]
                    eta = etas[iobj]
                    phi = phis[iobj]
                    mass = masses[iobj]

                    px = pt * np.cos(phi)
                    py = pt * np.sin(phi)
                    pz = pt * np.sinh(eta)
                    e = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
                    
                    px_total += px 
                    py_total += py 
                    pz_total += pz 
                    e_total += e

            inv_mass = np.sqrt(-(px_total**2 + py_total**2 + pz_total**2 - e_total**2))
            pt_total = np.sqrt(px_total**2 + py_total**2)
            out_inv_mass[iev] = inv_mass
            out_pt_total[iev] = pt_total

@cuda.jit
def compute_inv_mass_cudakernel(offsets, pts, etas, phis, masses, mask_events, mask_objects, out_inv_mass, out_pt_total):
    xi = cuda.grid(1)
    xstride = cuda.gridsize(1)
    for iev in range(xi, offsets.shape[0]-1, xstride):
        if mask_events[iev]:
            start = np.uint64(offsets[iev])
            end = np.uint64(offsets[iev + 1])
            
            px_total = np.float32(0.0)
            py_total = np.float32(0.0)
            pz_total = np.float32(0.0)
            e_total = np.float32(0.0)
            
            for iobj in range(start, end):
                if mask_objects[iobj]:
                    pt = pts[iobj]
                    eta = etas[iobj]
                    phi = phis[iobj]
                    mass = masses[iobj]

                    px = pt * math.cos(phi)
                    py = pt * math.sin(phi)
                    pz = pt * math.sinh(eta)
                    e = math.sqrt(px**2 + py**2 + pz**2 + mass**2)
                    
                    px_total += px 
                    py_total += py 
                    pz_total += pz 
                    e_total += e

            inv_mass = math.sqrt(-(px_total**2 + py_total**2 + pz_total**2 - e_total**2))
            pt_total = math.sqrt(px_total**2 + py_total**2)
            out_inv_mass[iev] = inv_mass
            out_pt_total[iev] = pt_total

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def parse_args():
    parser = argparse.ArgumentParser(description='Caltech HiggsMuMu analysis')
    parser.add_argument('--datapath', action='store',
        help='Input file path that contains the CMS /store/... folder, e.g. /mnt/hadoop',
        required=False, default="/storage/user/jpata")
    parser.add_argument('--cachepath', action='store',
        help='Location where to store the cache',
        required=False, default="./mycache")
    parser.add_argument('--ismc', action='store_true',
        help='Flag to specify if dataset is MC')
    parser.add_argument('--skim', action='store_true',
        help='Specify if skim should be done')
    parser.add_argument('--nocache', action='store_true',
        help='Flag to specify if branch cache will be skipped')
    parser.add_argument('--cuda', action='store_true',
        help='Use the cuda backend')
    parser.add_argument('--nthreads', action='store',
        help='Number of parallel threads', default=1, type=int)
    parser.add_argument('--out', action='store',
        help='Output file name', default="out")
    parser.add_argument('--njobs', action='store',
        help='Number of multiprocessing jobs', default=1, type=int)

    parser.add_argument('filenames', nargs=argparse.REMAINDER)
 
    args = parser.parse_args()
    return args

def load_and_analyze(args_tuple):
    fn, args, njec, dataset, ismc, ichunk = args_tuple
    ds = load_dataset(args.datapath, args.cachepath, fn, ismc, args.nthreads, args.nocache, args.skim)
    run_analysis(ds, "{0}_{1}".format(dataset, ichunk), njec, args.cuda)
    return ds.numevents()

if __name__ == "__main__":
    args = parse_args()

    files_per_job = 2

    NUMPY_LIB, ha = hepaccelerate.choose_backend(args.cuda)
    if args.cuda:
        import cuda_kernels as kernels
    else:
        import cpu_kernels as kernels

    electron_weights = NUMPY_LIB.zeros((100, 2), dtype=NUMPY_LIB.float32)
    electron_weights[:, 0] = NUMPY_LIB.linspace(0, 200, electron_weights.shape[0])[:]
    electron_weights[:, 1] = NUMPY_LIB.random.normal(loc=1.0, scale=0.1, size=electron_weights.shape[0])[:]

    njec = 1
    if args.ismc:
        njec = 50

    if len(args.filenames) > 0:
        print("Processing the provided files")
        ds = load_dataset(args.datapath, args.cachepath, args.filenames, args.ismc, args.nthreads, args.nocache, args.skim)
        run_analysis(ds, args.out, njec, args.cuda)
    else:
        print("Processing all datasets")
        datasets = [
            ("DYJetsToLL", "/opendata_files/DYJetsToLL-merged/*.root", True),
            ("TTJets_FullLeptMGDecays", "/opendata_files/TTJets_FullLeptMGDecays-merged/*.root", True),
            ("TTJets_Hadronic", "/opendata_files/TTJets_Hadronic-merged/*.root", True),
            ("TTJets_SemiLeptMGDecays", "/opendata_files/TTJets_SemiLeptMGDecays-merged/*.root", True),
            ("W1JetsToLNu", "/opendata_files/W1JetsToLNu-merged/*.root", True),
            ("W2JetsToLNu", "/opendata_files/W2JetsToLNu-merged/*.root", True),
            ("W3JetsToLNu", "/opendata_files/W3JetsToLNu-merged/*.root", True),
            ("GluGluToHToMM", "/opendata_files/GluGluToHToMM-merged/*.root", True),
            ("SingleMu", "/opendata_files/SingleMu-merged/*.root", False),
        ]
        arglist = []
        for dataset, fn_pattern, ismc in datasets:
            filenames = glob.glob(args.datapath + fn_pattern)
            ichunk = 0
            for fn in chunks(filenames, files_per_job):
                arglist += [(fn, args, njec, dataset, ismc, ichunk)]
                ichunk += 1

        full_t0 = time.time()
        if args.njobs == 1:
            ret = map(load_and_analyze, arglist)
        else:
            pool = multiprocessing.Pool(args.njobs)
            ret = pool.map(load_and_analyze, arglist)
            pool.close()
        full_t1 = time.time()
        Nev = sum(ret)
        print(Nev, full_t1 - full_t0)
