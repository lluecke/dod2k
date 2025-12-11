def identify_multiple_duplicates(pot_dup_inds, df):
    """
    Identify records that appear in multiple duplicate pairs.
    
    Returns:
    --------
    record_to_pairs : dict
        Maps record index to list of pair indices it appears in
    multiple_records : list of tuples
        [(record_idx, count, pair_indices), ...] sorted by count descending
    """
    from collections import defaultdict
    
    record_to_pairs = defaultdict(list)
    
    # Map each record to all pairs it appears in
    for pair_idx, (i, j) in enumerate(pot_dup_inds):
        record_to_pairs[i].append(pair_idx)
        record_to_pairs[j].append(pair_idx)
    
    # Find records that appear in multiple pairs
    multiple_records = []
    for record_idx, pair_indices in record_to_pairs.items():
        if len(pair_indices) > 1:
            multiple_records.append((record_idx, len(pair_indices), pair_indices))
    
    # Sort by count (highest first)
    multiple_records.sort(key=lambda x: x[1], reverse=True)
    
    return record_to_pairs, multiple_records


def create_ID_dup_dict(pot_dup_inds):
    dup_dict         = {}
    reverse_dup_dict = {}
    for id1, id2 in pot_dup_inds:
        # check if id1 appears in reverse_dup_dict: 
        if id1 in reverse_dup_dict:
            dup_id = reverse_dup_dict[id1] # if YES, it means it is already associated with a duplicate record, use this for mapping
        else:
            dup_id = id1 # if NO this is the first time id1 appears as a potential dup, then use this for mapping
        if dup_id not in dup_dict:
            dup_dict[dup_id] = []
        if dup_id!=id1
            dup_dict[dup_id]+=[id1]
        dup_dict[dup_id]+=[id2]
    return dup_dict