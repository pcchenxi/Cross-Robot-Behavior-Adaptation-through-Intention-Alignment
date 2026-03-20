import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from itertools import combinations
from collections import defaultdict
import pandas as pd
from collections import defaultdict
import numpy as np

label_to_task = {
            'water_1': 'monitor_w1', 'water_2':'monitor_w2', 'area_1':'monitor_t1', 'area_2':'monitor_t2',

            'big_wood_block':'pick_wood',
            'small_wood_block':'pick_wood',
            'blue_plastic_cup':'pick_cup',
            'red_plastic_cup':'pick_cup',
            'blue_metal_cup':'pick_cup',
            'red_metal_cup':'pick_cup',
            'red_glue_gun':'pick_gun',
            'yellow_glue_gun':'pick_gun',            
            'black_tape':'pick_tape',
            'yellow_black_tape':'pick_tape',
            'brown_spray_paint':'pick_paint',
            'blue_spray_paint':'pick_paint',

            'robot':'prepare_handover', 'put_on_table':'prepare_handover',
            'put_on_robot':'complete_handover', 'put_item':'complete_handover', 'pull_item':'complete_handover',
            'target_1':'deliver_1', 'target_2':'deliver_2',
}

label_list = ['monitor_w1', 'monitor_w2', 'monitor_t1', 'monitor_t2', 
              'pick_wood', 'pick_cup', 'pick_gun', 'pick_tape', 'pick_paint', 
              'prepare_handover', 'complete_handover',
              'deliver_1', 'deliver_2']



# label_to_task = {
#             'water_1': 'monitor_w1', 'water_2':'monitor_w2', 'area_1':'monitor_t1', 'area_2':'monitor_t2',

#             'big_wood_block':'big_wood_block',
#             'small_wood_block':'small_wood_block',
#             'blue_plastic_cup':'blue_plastic_cup',
#             'red_plastic_cup':'red_plastic_cup',
#             'blue_metal_cup':'blue_metal_cup',
#             'red_metal_cup':'red_metal_cup',
#             'red_glue_gun':'red_glue_gun',
#             'yellow_glue_gun':'yellow_glue_gun',            
#             'black_tape':'black_tape',
#             'yellow_black_tape':'yellow_black_tape',
#             'brown_spray_paint':'brown_spray_paint',
#             'blue_spray_paint':'blue_spray_paint',

#             'robot':'prepare_handover', 'put_on_table':'prepare_handover',
#             'put_on_robot':'complete_handover', 'put_item':'complete_handover', 'pull_item':'complete_handover',
#             'target_1':'deliver_1', 'target_2':'deliver_2',
# }

# label_list = [
#                 # 'monitor_w1', 'monitor_w2', 'monitor_t1', 'monitor_t2', 
#               'big_wood_block', 'small_wood_block', 'brown_spray_paint', 'blue_spray_paint', 
#               'blue_plastic_cup', 'red_plastic_cup', 'blue_metal_cup', 'red_metal_cup',
#               'red_glue_gun', 'yellow_glue_gun', 'black_tape', 'yellow_black_tape',
#             #   'prepare_handover', 'complete_handover',
#             #   'deliver_1', 'deliver_2'
#             ]

def compute_intra_inter_cosine(embeddings, labels):
    embeddings = np.array(embeddings)
    labels = np.array(labels)

    intra_dists = {}
    inter_dists = []

    # Precompute pairwise cosine distances
    pairwise_dist = cosine_distances(embeddings)

    # Mapping from label to indices
    label_to_indices = defaultdict(list)
    for i, lbl in enumerate(labels):
        # task = label_to_task[lbl]
        label_to_indices[lbl].append(i)

    # Compute intra-class distances
    for label in label_list:
        indices = label_to_indices[label]
        if len(indices) < 2:
            intra_dists[label] = np.nan  # Not enough samples
            # print('skip', label)
            continue
        sub_dist = pairwise_dist[np.ix_(indices, indices)]
        upper_triangle = sub_dist[np.triu_indices_from(sub_dist, k=1)]
        intra_dists[label] = np.mean(upper_triangle)

    # Compute inter-class distances (across all label pairs)
    for (label1, label2) in combinations(label_list, 2):
        idx1 = label_to_indices[label1]
        idx2 = label_to_indices[label2]
        if len(idx1) == 0 or len(idx2) == 0:
            continue
        dists = pairwise_dist[np.ix_(idx1, idx2)]
        inter_dists.append(np.mean(dists))

    inter_dist = np.mean(inter_dists)

    return intra_dists, inter_dist


def compute_cross_embodiment_error(embeddings, labels, robot_ids, target_task):
    """
    embeddings: (N, D) latent vectors (256-D)
    labels: (N,) task labels (e.g., 'pick', 'monitor', etc.)
    robot_ids: (N,) robot identifier (e.g., 'tello', 'spark', etc.)
    target_task: str, the task to evaluate (e.g., 'pick')
    """
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    robot_ids = np.array(robot_ids)

    # Filter for the selected task label
    task_mask = labels == target_task
    task_embeddings = embeddings[task_mask]
    task_robots = robot_ids[task_mask]

    # Group embeddings by robot
    robot_to_embeddings = defaultdict(list)
    for emb, r in zip(task_embeddings, task_robots):
        robot_to_embeddings[r].append(emb)
    
    # Compute centroids per robot
    robot_centroids = {r: np.mean(v, axis=0) for r, v in robot_to_embeddings.items() if len(v) > 0}

    # Compute pairwise cosine distances between robot centroids
    robot_list = list(robot_centroids.keys())
    if len(robot_list) < 2:
        # print('skip')
        return np.nan  # Not enough robots for this task

    dists = []
    for r1, r2 in combinations(robot_list, 2):
        dist = cosine_distances(
            robot_centroids[r1].reshape(1, -1),
            robot_centroids[r2].reshape(1, -1)
        )[0, 0]
        dists.append(dist)

    return np.mean(dists)

def compute_inter_intra_emb_error(path):
    # path = 'paper12/'
    # path = ''
    tags_p = np.load('./latent-result/'+path+'paper_tag.npy')
    marker_p = np.load('./latent-result/'+path+'paper_marker.npy')
    embed_p = np.load('./latent-result/'+path+'paper_embedding.npy')

    # print(tags_p)

    task_list = []
    for l in tags_p:
        task = label_to_task[l]
        task_list.append(task)

    # print(len(tags_p), len(marker_p), embed_p.shape)

    intra_dists, inter_dist = compute_intra_inter_cosine(embed_p, task_list)
    mean_intra = np.nanmean(list(intra_dists.values()))
    separation_ratio = inter_dist / mean_intra

    # print("Inter-class cosine distance:", inter_dist)
    # for label in label_list:
    #     print(f"Intra-class cosine distance for '{label}':", intra_dists[label])
    # # print('Seperation ratio', separation_ratio)

    emb_dists = {}
    for task in label_list:
        error = compute_cross_embodiment_error(embed_p, task_list, marker_p, task)
        # print(f"{task} → cross-embodiment error: {error:.4f}")
        emb_dists[task] = error

    mean_emb = np.nanmean(list(emb_dists.values()))
    alignment_ratio = inter_dist / mean_emb

    return intra_dists, inter_dist, emb_dists


path='seed1/'
intra_dists_1, inter_dist_1, emb_dists_1 = compute_inter_intra_emb_error(path)
path='seed2/'
intra_dists_2, inter_dist_2, emb_dists_2 = compute_inter_intra_emb_error(path)
path='seed3/'
intra_dists_3, inter_dist_3, emb_dists_3 = compute_inter_intra_emb_error(path)


seeds = [1, 2, 3]
inter_dists   = {1: inter_dist_1, 2: inter_dist_2, 3: inter_dist_3}
intra_dicts = {1: intra_dists_1, 2: intra_dists_2, 3: intra_dists_3}
emb_dicts   = {1: emb_dists_1, 2: emb_dists_2, 3: emb_dists_3}

# Collect all labels that appear in any seed
labels = {lab for d in intra_dicts.values() for lab in d}
labels = sorted(labels)

# Helper: accumulate per-label lists
data = defaultdict(lambda: defaultdict(list))   # data[label]['intra'] etc.

for s in seeds:
    inter = inter_dists[s]
    for lab in labels:
        intra = intra_dicts[s][lab]
        emb  = emb_dicts[s][lab]
        sep_ratio     = inter / intra           # separation (semantic)
        align_ratio   = inter / emb             # embodiment alignment

        data[lab]['intra'].append(intra)
        data[lab]['emb' ].append(emb)
        data[lab]['sep' ].append(sep_ratio)
        data[lab]['ali' ].append(align_ratio)

# Build a nice dataframe with mean±std strings
rows = []
for lab in labels:
    def mstd(key):
        arr = np.asarray(data[lab][key])
        return arr.mean(), arr.std(ddof=1)

    m_intra,  s_intra  = mstd('intra')
    m_emb,    s_emb    = mstd('emb')
    m_sep,    s_sep    = mstd('sep')
    m_align,  s_align  = mstd('ali')

    rows.append(dict(Subtask=lab,
                     intra_mean = m_intra,  intra_std = s_intra,
                     emb_mean   = m_emb,    emb_std   = s_emb,
                     sep_mean   = m_sep,    sep_std   = s_sep,
                     ali_mean   = m_align,  ali_std   = s_align))

df = pd.DataFrame(rows).set_index('Subtask')
print(df.to_string(float_format=lambda x: f"{x:.3f}"))

# -------- global averages --------------------------------------------
mean_inter = np.mean(list(inter_dists.values()))
mean_intra = np.mean([np.mean(v['intra']) for v in data.values()])
mean_emb   = np.mean([np.mean(v['emb'])   for v in data.values()])

# ---------- summary values ------------------------------------------------
inter_vals = np.array(list(inter_dists.values()))
g_inter_mean = inter_vals.mean()
g_inter_std  = inter_vals.std(ddof=1)

# per-seed averages of the two ratios (already in `df`)
sep_seed_avg   = df['sep_mean'].mean()
ali_seed_avg   = df['ali_mean'].mean()
sep_seed_std   = df['sep_mean'].std (ddof=1)
ali_seed_std   = df['ali_mean'].std (ddof=1)

# “global-recomputed” ratios from grand means
g_sep   = g_inter_mean / mean_intra
g_align = g_inter_mean / mean_emb

# ----------------- print exactly five lines -------------------------------
print(f"Global inter-class distance  : {g_inter_mean:.3f} ± {g_inter_std:.3f}")
print(f"Mean separation ratio       : {sep_seed_avg:.2f} ± {sep_seed_std:.2f}")
print(f"Mean alignment  ratio       : {ali_seed_avg:.2f} ± {ali_seed_std:.2f}")
print(f"Global separation ratio     : {g_sep:.3f}")
print(f"Global alignment ratio      : {g_align:.3f}")