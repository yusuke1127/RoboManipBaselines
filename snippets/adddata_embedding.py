from pathlib import Path
import numpy as np
import h5py as h5
from sentence_transformers import SentenceTransformer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
args = parser.parse_args()

data_dir = Path(args.data_dir)

model = SentenceTransformer('sentence-transformers/use-cmlm-multilingual')
model.eval()

R_ROBOT_TEMPLATE = "Right robot {}."

# make sample data for hdf5 (20250722)

ACTION_SENTENCES = {
    "right_robot": [
        R_ROBOT_TEMPLATE.format(action)
        for action in (
            "lifts a cable",
            "carries a cable to the first corner",
            "carries a cable to the final position",
            "put a cable down to the table",
        )
    ]
}

embeddings_list = {
    agent: model.encode(sentences)
    for agent, sentences in ACTION_SENTENCES.items()
}

joints_hdf5rawlist = list(sorted(data_dir.glob("*.rmb/main.rmb.hdf5")))
for joints_hdf5data in joints_hdf5rawlist:
    with h5.File(joints_hdf5data, 'a') as f:
        emb_idxlist = np.zeros(len(f["measured_joint_pos_rel"]), dtype=int)
        for i in range(len(emb_idxlist)):
            if f["measured_joint_pos_rel"][i, 0] > 0.004:
                emb_idxlist[i] = 1
            if f["measured_joint_pos_rel"][i, 0] < -0.004:
                emb_idxlist[i] = 2
            if f["measured_gripper_joint_pos"][i] < 150:
                emb_idxlist[i] = 3
        for i in range(len(emb_idxlist) - 1):
            if emb_idxlist[i] > emb_idxlist[i + 1]:
                emb_idxlist[i + 1] = emb_idxlist[i]
        
        embedding = np.empty((len(emb_idxlist), embeddings_list["right_robot"].shape[1]))
        for i in range(len(embedding)):
            embedding[i] = embeddings_list["right_robot"][emb_idxlist[i]]
        
        f.create_dataset("tasks", data=embedding)