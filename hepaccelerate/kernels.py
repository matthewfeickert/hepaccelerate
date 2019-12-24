"""This file contains the public-facing API of the kernels
"""

def spherical_to_cartesian(backend, pt, eta, phi, mass):
    """Converts an array of spherical four-momentum coordinates (pt, eta, phi, mass) to cartesian (px, py ,pz, E).
    
    Args:
        backend (library): either hepaccelerate.backend_cpu or hepaccelerate.backend_cuda
        pt (array of floats): Data array of the transverse momentum values (numpy or cupy)
        eta (array of floats): Data array of the pseudorapidity
        phi (array of floats): Data array of the azimuthal angle
        mass (array of floats): Data array of the mass
    
    Returns:
        tuple of arrays: returns the numpy or cupy arrays (px, py, pz, E) 
    """
    return backend.spherical_to_cartesian(pt, eta, phi, mass)

def cartesian_to_spherical(backend, px, py, pz, e):
    """Converts an array of cartesian four-momentum coordinates (px, py ,pz, E) to spherical (pt, eta, phi, mass).
    
    Args:
        backend (library): either hepaccelerate.backend_cpu or hepaccelerate.backend_cuda
        px (array of floats): Data array of the momentum x coordinate (numpy or cupy)
        py (array of floats): Data array of the momentum y coordinate
        pz (array of floats): Data array of the momentum z coordinate
        e (array of floats): Data array of the energy values
    
    Returns:
        tuple of arrays: returns the numpy or cupy arrays (pt, eta, phi, mass)
    """
    return backend.cartesian_to_spherical(px, py, pz, e)

def searchsorted(backend, arr, vals, side="right"):
    """Finds where to insert the values in 'vals' into a sorted array 'arr' to preserve order, as np.searchsorted.
    
    Args:
        backend (library): either hepaccelerate.backend_cpu or hepaccelerate.backend_cuda
        arr (array of floats): sorted array of bin edges
        vals (array of floats): array of values to insert into the sorted array
        side (str, optional): "left" or "right" as in np.searchsorted
    
    Returns:
        array of ints: Indices into 'arr' where the values would be inserted 
    """
    return backend.searchsorted(arr, vals, side=side)

def broadcast(backend, offsets, content, out):
    """Given the offsets from a one-dimensional jagged array, broadcasts a per-event array to a per-object array.

    >>> j = awkward.fromiter([[1.0, 2.0],[3.0, 4.0, 5.0]])
    >>> inp = numpy.array([123.0, 456.0])
    >>> out = numpy.zeros_like(j.content)
    >>> broadcast(backend_cpu, j.offsets, inp, out)
    >>> j2 = awkward.JaggedArray.fromoffsets(j.offsets, out)
    >>> r = (j2 == awkward.fromiter([[123.0, 123.0],[456.0, 456.0, 456.0]]))
    >>> assert(numpy.all(r.content))
    
    Args:
        backend (library): either hepaccelerate.backend_cpu or hepaccelerate.backend_cuda
        offsets (array of uint64): one dimensional offsets of the jagged array (depth 1)
        content (array of floats): per-event array of inputs to broadcast
        out (array of floats): per-element output
    """
    backend.broadcast(offsets, content, out)

def sum_in_offsets(backend, offsets, content, mask_rows=None, mask_content=None, dtype=None):
    """Sums the values in a depth-1 jagged array within the offsets, e.g. to compute a per-event sum
    
    >>> j = awkward.fromiter([[1.0, 2.0],[3.0, 4.0, 5.0], [6.0, 7.0], [8.0]])
    >>> mr = numpy.array([True, True, True, False]) # Disable the last event ([8.0])
    >>> mc = numpy.array([True, True, True, True, False, True, True, True]) # Disable the 5th value (5.0)
    >>> r = sum_in_offsets(backend_cpu, j.offsets, j.content, mask_rows=mr, mask_content=mc)
    >>> assert(numpy.all(r == numpy.array([3.0, 7.0, 13.0, 0.0])))

    Args:
        backend (library): either hepaccelerate.backend_cpu or hepaccelerate.backend_cuda
        offsets (array of uint64): one dimensional offsets of the jagged array (depth 1)
        content (array): data array to sum over
        mask_rows (array of bool, optional): Mask the values in the offset array that are set to False
        mask_content (array of bool, optional): Mask the values in the data array that are set to False
        dtype (data type, optional): Output data type, useful to specify e.g. int8 when summing over booleans
    
    Returns:
        array: Totals within the offsets
    """
    return backend.sum_in_offsets(offsets, content, mask_rows=mask_rows, mask_content=mask_content, dtype=dtype)

def prod_in_offsets(backend, offsets, content, mask_rows=None, mask_content=None, dtype=None):
    """Summary

    >>> j = awkward.fromiter([[1.0, 2.0],[3.0, 4.0, 5.0], [6.0, 7.0], [8.0]])
    >>> mr = numpy.array([True, True, True, False]) # Disable the last event ([8.0])
    >>> mc = numpy.array([True, True, True, True, False, True, True, True]) # Disable the 5th value (5.0)
    >>> r = prod_in_offsets(backend_cpu, j.offsets, j.content, mask_rows=mr, mask_content=mc)
    >>> assert(numpy.all(r == numpy.array([2.0, 12.0, 42.0, 1.0])))

    Args:
        backend (TYPE): Description
        offsets (TYPE): Description
        content (TYPE): Description
        mask_rows (TYPE): Description
        mask_content (TYPE): Description
        dtype (None, optional): Description
    
    Returns:
        TYPE: Description
    """
    return backend.prod_in_offsets(offsets, content, mask_rows, mask_content, dtype=dtype)

def max_in_offsets(backend, offsets, content, mask_rows=None, mask_content=None):
    """Summary

    >>> j = awkward.fromiter([[1.0, 2.0],[3.0, 4.0, 5.0], [6.0, 7.0], [8.0]])
    >>> mr = numpy.array([True, True, True, False]) # Disable the last event ([8.0])
    >>> mc = numpy.array([True, True, True, True, False, True, True, True]) # Disable the 5th value (5.0)
    >>> r = max_in_offsets(backend_cpu, j.offsets, j.content, mask_rows=mr, mask_content=mc)
    >>> assert(numpy.all(r == numpy.array([2.0, 4.0, 7.0, 0.0])))

    Args:
        backend (TYPE): Description
        offsets (TYPE): Description
        content (TYPE): Description
        mask_rows (None, optional): Description
        mask_content (None, optional): Description
    
    Returns:
        TYPE: Description
    """
    return backend.max_in_offsets(offsets, content, mask_rows=mask_rows, mask_content=mask_content)

def min_in_offsets(backend, offsets, content, mask_rows=None, mask_content=None):
    """Summary

    >>> j = awkward.fromiter([[1.0, 2.0],[3.0, 4.0, 5.0], [6.0, 7.0], [8.0]])
    >>> mr = numpy.array([True, True, True, False]) # Disable the last event ([8.0])
    >>> mc = numpy.array([True, True, True, True, False, True, True, True]) # Disable the 5th value (5.0)
    >>> r = min_in_offsets(backend_cpu, j.offsets, j.content, mask_rows=mr, mask_content=mc)
    >>> assert(numpy.all(r == numpy.array([1.0, 3.0, 6.0, 0.0])))

    Args:
        backend (TYPE): Description
        offsets (TYPE): Description
        content (TYPE): Description
        mask_rows (None, optional): Description
        mask_content (None, optional): Description
    
    Returns:
        TYPE: Description
    """
    return backend.min_in_offsets(offsets, content, mask_rows=mask_rows, mask_content=mask_content)

def select_opposite_sign(backend, offsets, charges, in_mask):
    """Summary
    
    Args:
        backend (TYPE): Description
        offsets (TYPE): Description
        charges (TYPE): Description
        in_mask (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    return backend.select_opposite_sign(offsets, charges, in_mask)

def get_in_offsets(backend, offsets, content, indices, mask_rows=None, mask_content=None):
    """Retrieves the per-event values corresponding to indices in the content array.

    >>> j = awkward.fromiter([[1.0, 2.0],[3.0, 4.0, 5.0], [6.0, 7.0], [8.0]])
    >>> #Retrieve the first non-masked value in the first and second event, and the second value in the third and fourth event
    >>> inds = numpy.array([0, 0, 1, 1])
    >>> mr = numpy.array([True, True, True, False]) # Disable the last event ([8.0])
    >>> mc = numpy.array([True, True, False, True, True, True, True, True]) # Disable the 3rd value (3.0)
    >>> r = get_in_offsets(backend_cpu, j.offsets, j.content, inds, mask_rows=mr, mask_content=mc)
    >>> assert(numpy.all(r == numpy.array([1.0, 4.0, 7.0, 0.0])))

    Args:
        backend (TYPE): Description
        offsets (TYPE): Description
        content (TYPE): Description
        indices (TYPE): Description
        mask_rows (None, optional): Description
        mask_content (None, optional): Description
    
    Returns:
        TYPE: Description
    """
    return backend.get_in_offsets(offsets, content, indices, mask_rows=mask_rows, mask_content=mask_content) 

def set_in_offsets(backend, offsets, content, indices, target, mask_rows=None, mask_content=None):
    """Sets the per-event values corresponding to indices in the content array to the values in the target array.
    
    >>> j = awkward.fromiter([[0.0, 0.0],[0.0, 0.0, 0.0], [0.0, 0.0], [0.0]])
    >>> inds = numpy.array([0, 0, 1, 1])
    >>> target = numpy.array([1, 2, 3, 4])
    >>> mr = numpy.array([True, True, True, False]) # Disable the last event
    >>> mc = numpy.array([True, True, False, True, True, True, True, True]) # Disable the 3rd value in the content array
    >>> set_in_offsets(backend_cpu, j.offsets, j.content, inds, target, mask_rows=mr, mask_content=mc)
    >>> r = j == awkward.fromiter([[1.0, 0.0],[0.0, 2.0, 0.0], [0.0, 3.0], [0.0]])
    >>> assert(numpy.all(r.content))

    Args:
        backend (TYPE): Description
        offsets (TYPE): Description
        content (TYPE): Description
        indices (TYPE): Description
        target (TYPE): Description
        mask_rows (None, optional): Description
        mask_content (None, optional): Description
    
    Returns:
        TYPE: Description
    """
    backend.set_in_offsets(offsets, content, indices, target, mask_rows=mask_rows, mask_content=mask_content)
 
def mask_deltar_first(backend, objs1, mask1, objs2, mask2, drcut):
    """Summary
    
    Args:
        backend (TYPE): Description
        objs1 (TYPE): Description
        mask1 (TYPE): Description
        objs2 (TYPE): Description
        mask2 (TYPE): Description
        drcut (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    return backend.mask_deltar_first(objs1, mask1, objs2, mask2, drcut)

def histogram_from_vector(backend, data, weights, bins, mask=None):
    """Summary
    
    Args:
        backend (TYPE): Description
        data (TYPE): Description
        weights (TYPE): Description
        bins (TYPE): Description
        mask (None, optional): Description
    
    Returns:
        TYPE: Description
    """
    return backend.histogram_from_vector(data, weights, bins, mask=mask)
 
def histogram_from_vector_several(backend, variables, weights, mask):
    """Summary
    
    Args:
        backend (TYPE): Description
        variables (TYPE): Description
        weights (TYPE): Description
        mask (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    return backend.histogram_from_vector_several(variables, weights, mask) 

def get_bin_contents(backend, values, edges, contents, out):
    """Summary
    
    Args:
        backend (TYPE): Description
        values (TYPE): Description
        edges (TYPE): Description
        contents (TYPE): Description
        out (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    return backend.get_bin_contents(values, edges, contents, out) 

def copyto_dst_indices(backend, dst, src, inds_dst):
    """Summary
    
    Args:
        backend (TYPE): Description
        dst (TYPE): Description
        src (TYPE): Description
        inds_dst (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    return backend.copyto_dst_indices(dst, src, inds_dst) 

def compute_new_offsets(backend, offsets_old, mask_objects, offsets_new):
    """Summary
    
    Args:
        backend (TYPE): Description
        offsets_old (TYPE): Description
        mask_objects (TYPE): Description
        offsets_new (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    return backend.compute_new_offsets(offsets_old, mask_objects, offsets_new)

if __name__ == "__main__":
    import doctest
    import awkward, numpy
    import hepaccelerate.backend_cpu as backend_cpu
    doctest.testmod()
