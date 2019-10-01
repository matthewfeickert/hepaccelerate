import numba
import numpy as np
import math

@numba.njit
def set_array(arr, pt, eta, phi, mass, i):
    arr[0] = pt[i]
    arr[1] = eta[i]
    arr[2] = phi[i]
    arr[3] = mass[i]

@numba.njit(fastmath=True)
def spherical_to_cartesian(p4):
    pt = p4[0]
    eta = p4[1]
    phi = p4[2]
    mass = p4[3]

    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * math.sinh(eta)
    e = math.sqrt(px**2 + py**2 + pz**2 + mass**2)
    
    p4[0] = px
    p4[1] = py
    p4[2] = pz
    p4[3] = e

@numba.njit(fastmath=True)
def inv_mass_3(p1c, p2c, p3c):
    px = p1c[0] + p2c[0] + p3c[0]
    py = p1c[1] + p2c[1] + p3c[1]
    pz = p1c[2] + p2c[2] + p3c[2]
    e = p1c[3] + p2c[3] + p3c[3]
    inv_mass = math.sqrt(-(px**2 + py**2 + pz**2 - e**2))
    return inv_mass

@numba.njit(fastmath=True, parallel=True)
def comb_3_invmass_closest(pt, eta, phi, mass, offsets, candidate_mass, out_mass, out_best_comb):
    for iev in numba.prange(offsets.shape[0]-1):
        start = offsets[iev]
        end = offsets[iev + 1]
        nobj = end - start

        #create a arrays of the cartesian components
        p = np.zeros((nobj, 4), numba.float32)
        p[:] = 0
        for iobj in range(start, end):
            set_array(p[iobj - start, :], pt, eta, phi, mass, iobj)
            spherical_to_cartesian(p[iobj - start, :])

        #mass delta R to previous
        delta_previous = 1e10
        iobj_best = 0
        jobj_best = 0
        kobj_best = 0
        mass_best = 0

        #compute the invariant mass of all combinations
        for iobj in range(start, end):
            for jobj in range(iobj + 1, end):
                for kobj in range(jobj + 1, end):
                    inv_mass = inv_mass_3(p[iobj - start, :], p[jobj - start, :], p[kobj - start, :])
                    delta = abs(inv_mass - candidate_mass)
                    if delta < delta_previous:
                        mass_best = inv_mass
                        iobj_best = iobj
                        jobj_best = jobj
                        kobj_best = kobj
                        delta_previous = delta
        out_mass[iev] = mass_best
        out_best_comb[iev, 0] = iobj_best - start
        out_best_comb[iev, 1] = jobj_best - start
        out_best_comb[iev, 2] = kobj_best - start

@numba.njit
def max_arr(arr):
    m = arr[0]
    for i in range(len(arr)):
        if arr[i] > m:
            m = arr[i]
    return m

@numba.njit(fastmath=True, parallel=True)
def max_val_comb(vals, offsets, best_comb, out_vals):
    for iev in numba.prange(offsets.shape[0]-1):
        start = offsets[iev]
        end = offsets[iev + 1]

        vals_comb = np.zeros(best_comb.shape[1], dtype=numba.float32)
        for icomb in range(best_comb.shape[1]):
            idx_jet = best_comb[iev, icomb]
            if idx_jet >= 0:
                vals_comb[icomb] = vals[start + idx_jet]

        out_vals[iev] = max_arr(vals_comb)

#For events with at least three leptons and a same-flavor opposite-sign lepton pair,
#find the same-flavor opposite-sign lepton pair with the mass closest to 91.2 GeV and
#plot the transverse mass of the missing energy and the leading other lepton.
