#This file contains the public-facing kernels

def spherical_to_cartesian(backend, pt, eta, phi, mass):
    return backend.spherical_to_cartesian(pt, eta, phi, mass)

def cartesian_to_spherical(backend, px, py, pz, e):
    return backend.cartesian_to_spherical(px, py, pz, e)

def searchsorted(backend, arr, vals, side="right"):
    return backend.searchsorted(arr, vals, side=side)

def broadcast(backend, offsets, content, out):
    return backend.broadcast(offsets, content, out)

def sum_in_offsets(backend, offsets, content, mask_rows=None, mask_content=None, dtype=None):
    return backend.sum_in_offsets(offsets, content, mask_rows=mask_rows, mask_content=mask_content, dtype=dtype)

def prod_in_offsets(backend, offsets, content, mask_rows, mask_content, dtype=None):
    return backend.prod_in_offsets(offsets, content, mask_rows, mask_content, dtype=dtype)

def max_in_offsets(backend, offsets, content, mask_rows=None, mask_content=None):
    return backend.max_in_offsets(offsets, content, mask_rows=mask_rows, mask_content=mask_content)

def min_in_offsets(backend, offsets, content, mask_rows=None, mask_content=None):
    return backend.min_in_offsets(offsets, content, mask_rows=mask_rows, mask_content=mask_content)

def select_opposite_sign(backend, offsets, charges, in_mask):
    return backend.select_opposite_sign(offsets, charges, in_mask)

def get_in_offsets(backend, offsets, content, indices, mask_rows=None, mask_content=None):
    return backend.get_in_offsets(offsets, content, indices, mask_rows=mask_rows, mask_content=mask_content) 

def set_in_offsets(backend, offsets, content, indices, target, mask_rows=None, mask_content=None):
    return backend.set_in_offsets(offsets, content, indices, target, mask_rows=mask_rows, mask_content=mask_content)
 
def mask_deltar_first(backend, objs1, mask1, objs2, mask2, drcut):
    return backend.mask_deltar_first(objs1, mask1, objs2, mask2, drcut)

def histogram_from_vector(backend, data, weights, bins, mask=None):
    return backend.histogram_from_vector(data, weights, bins, mask=mask)
 
def histogram_from_vector_several(backend, variables, weights, mask):
    return backend.histogram_from_vector_several(variables, weights, mask) 

def get_bin_contents(backend, values, edges, contents, out):
    return backend.get_bin_contents(values, edges, contents, out) 

def copyto_dst_indices(backend, dst, src, inds_dst):
    return backend.copyto_dst_indices(dst, src, inds_dst) 

def compute_new_offsets(backend, offsets_old, mask_objects, offsets_new):
    return backend.compute_new_offsets(offsets_old, mask_objects, offsets_new) 
