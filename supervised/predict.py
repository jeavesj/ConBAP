import os
import pandas as pd
import torch
from ConBAP import downstream_docking, ConBAP, downstream_affinity
from dataset_ConBAP import GraphDataset, PLIDataLoader
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
import time

def val(model, dataloader, device):
    model.eval()

    pred_list = []
    label_list = []
    time_list = []
    for data in dataloader:
        with torch.no_grad():
            data['ligand_features'] = data['ligand_features'].to(device)
            data['atom_pocket_features'] = data['atom_pocket_features'].to(device)
            data['amino_acid_features'] = data['amino_acid_features'].to(device)
            data['complex_features'] = data['complex_features'].to(device)
            output = model(data)
            if isinstance(output, tuple):
                pred, t = output
                time_list.append(t)
            else:
                pred = output
            label = data["ligand_features"].y

            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))

    model.train()

    return pred, time_list

def predict(data_root, graph_type, batch_size, unsupervised_model, supervised_model, input_csv=None):

    data_graph = os.path.join(data_root)
    if input_csv is not None:
        valid_df = pd.read_csv(input_csv)
    else:
        csvs = [os.path.join(data_root, i) for i in os.listdir(data_root) if 'csv' in i]
        valid_df = pd.read_csv(csvs[0])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("used device:", device)
    
    model = ConBAP(35, 256).to(device)

    load_model_dict(model, unsupervised_model)
    model_pose = downstream_docking(model, 256).to(device)
    model_pose = model_pose.to(device)
    valid_set = GraphDataset(data_dir=data_graph, data_df=valid_df, graph_type=graph_type, dis_threshold=8, create=False)
    valid_loader = PLIDataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=28)
    pred_pose, _ = val(model_pose, valid_loader, device)

    model_affinity = downstream_affinity(model, 256).to(device)
    load_model_dict(model_affinity, supervised_model)

    pred_affinity, times = val(model_affinity, valid_loader, device)
    native_pose_list = valid_df['pdb'].values.tolist()
    print("pdbid", "pose_score", "affinity")
    for i in range(len(native_pose_list)):
        print(native_pose_list[i], pred_pose[i], pred_affinity[i])

    return pred_affinity, native_pose_list, times


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/toy_set/', help='Path to input data directory')
    parser.add_argument('--input_csv', type=str, default=None, required=False, help='Path to csv with columns "pdb","affinity"')
    parser.add_argument('--type', type=str, default='affinity', help='options="pose" or "affinity" for ranking or scoring')
    parser.add_argument('--out_csv', type=str, required=False, help='Path to output csv for logging preprocessing times')
    parser.add_argument('--unsupervised_model_path', type=str, default=f'./unsupervised/model/20240111_193210_ConBAP_repeat3/model/contrastive_no_filtered.pt', help='Path to model weights')
    parser.add_argument('--supervised_model_path', type=str, default=f"./supervised/model/20231007_111336_ConBAP_repeat0/model/epoch-292, train_loss-0.1220, train_rmse-0.3493, valid_rmse-1.1663, valid_pr-0.7788.pt", help='Path to model weights')
    args = parser.parse_args()

    graph_type = 'ConBAP'
    batch_size = 1
    predict_type = args.type  # 'pose' or 'affinity'
    preds, pdb_ids, times = predict(args.data_root, graph_type, batch_size, args.unsupervised_model_path, args.supervised_model_path, input_csv=args.input_csv)

    if args.out_csv is not None:
        df = pd.DataFrame({'pdb': pdb_ids, 'pred_affinity': preds, 'inference_time_s': times})
        df.to_csv(args.out_csv, index=False)









