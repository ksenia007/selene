import numpy as np

cimport cython
cimport numpy as np

ctypedef np.int_t DTYPE_t
ctypedef np.float32_t FDTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False) 
def _fast_get_feature_data(int start,
                           int end,
                           int bin_size,
                           int step_size,
                           np.ndarray[FDTYPE_t, ndim=1] thresholds,
                           dict feature_index_dict,
                           rows):
    cdef int n_features = len(feature_index_dict)
    cdef int query_length = end - start
    cdef int n_bins = query_length / step_size
    cdef int feature_start, feature_end, index_start, index_end, index_feat
    cdef int threshold = -1
    cdef np.ndarray[DTYPE_t, ndim=1] int_thresholds
    cdef np.ndarray[DTYPE_t, ndim=2] encoding = np.zeros(
        (query_length, n_features), dtype=np.int)
    
    cdef np.ndarray[DTYPE_t, ndim=1] targets = np.zeros(
        n_features * n_bins, dtype=np.int)
    cdef np.ndarray[DTYPE_t, ndim=1] bin_targets = np.zeros(
        n_features, dtype=np.int)

    cdef list row
    cdef list used_feature_indices
    cdef int n_rows = 0
    cdef set feature_indices = set()

    if rows is None:
        return targets

    int_thresholds = (thresholds * query_length - 1).clip(min=0).astype(int)
    if len(np.unique(thresholds)) == 1:
        threshold = int(np.unique(int_thresholds)[0])

    for row in rows:
        feature_start = int(row[1])
        feature_end = int(row[2])
        index_start = max(0, feature_start - start)
        index_end = min(feature_end - start, query_length)
        index_feat = feature_index_dict[row[3]]
        feature_indices.add(index_feat)
        if index_start == index_end:
            index_end += 1
        encoding[index_start:index_end, index_feat] = 1
        n_rows += 1
    
    used_feature_indices = sorted(list(feature_indices))
    for ix, _ in enumerate(range(0, query_length, step_size)):
        start = ix * bin_size
        end = ix * bin_size + bin_size
        bin_encoding = encoding[start:end, used_feature_indices]
        
        if threshold > -1:
            bin_targets[used_feature_indices] = (np.sum(
                bin_encoding, axis=0) > threshold).astype(int)
        else:
            bin_targets[used_feature_indices] = (np.sum(
                bin_encoding, axis=0) > int_thresholds).astype(int)
        tgts_start = ix * n_features
        tgts_end = (ix * n_features) + n_features
        targets[tgts_start:tgts_end] = bin_targets
        bin_targets = np.zeros(n_features, dtype=np.int)
    return targets

