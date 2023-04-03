import sys
import numpy as np
import eval_metrics as em
import matplotlib.pyplot as plt
from eval_metrics import compute_det_curve

# Replace CM scores with your own scores or provide score file as the first argument.
cm_score_file =  r'I:\2021\outputs\output_rawnet.txt'
# Replace ASV scores with organizers' scores or provide score file as the second argument.
asv_score_file = r'I:\2021\data\PA\ASVspoof2019_PA_asv_scores\ASVspoof2019.PA.asv.eval.gi.trl.scores.txt'


sysid_dict= {
        '-': 0,  # bonafide speech
        'SS_1': 1,  # Wavenet vocoder
        'SS_2': 2,  # Conventional vocoder WORLD
        'SS_4': 3,  # Conventional vocoder MERLIN
        'US_1': 4,  # Unit selection system MaryTTS
        'VC_1': 5,  # Voice conversion using neural networks
        'VC_4': 6,  # transform function-based voice conversion
        # For PA:
        'AA': 7,
        'AB': 8,
        'AC': 9,
        'BA': 10,
        'BB': 11,
        'BC': 12,
        'CA': 13,
        'CB': 14,
        'CC': 15
    }


asv_data = np.genfromtxt(asv_score_file, dtype=str)
asv_sources = asv_data[:, 0]
asv_keys = asv_data[:, 1]
asv_scores = asv_data[:, 2].astype(np.float)


# Load CM scores
cm_data = np.genfromtxt(cm_score_file, dtype=str)
cm_utt_id = cm_data[:, 0]
cm_sources = cm_data[:, 1]
cm_keys = cm_data[:, 2]
cm_scores = cm_data[:, 3].astype(np.float)

# Extract target, nontarget, and spoof scores from the ASV scores
tar_asv = asv_scores[asv_keys == 'target']
non_asv = asv_scores[asv_keys == 'nontarget']
spoof_asv = asv_scores[asv_keys == 'spoof']

frr, far, thresholds = compute_det_curve(tar_asv,non_asv)
print(frr)
print(far)
print(thresholds)






