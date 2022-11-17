
datas
means
counts
BIC

datas_sub
assignments_sub
means_sub
counts_sub
BIC_sub


def xmeans_sub(
    data_offset: int,
    data_length: int,
    cluster_id: int,
    depth: int,
):
    if (data_length < 2):
        return
    id0 = cluster_id
    id1 = ?
    k2means  # id0, id1
    calc_BIC  # not aligned
    sub_BIC =
    if (distance(mean0, mean1) < min_distance && sub_BIC < BIC):
        align
        copy
        xmeans_sub
        xmeans_sub
