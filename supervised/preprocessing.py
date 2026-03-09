# %%
import os
import pickle
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import pymol
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import argparse

# %%

def generate_pocket(data_dir,data_df, distance=8):
    # 设置data_df的index为pdbid
    data_df = data_df.set_index('pdb')
    complex_id = data_df.index
    for cid in complex_id:
        print(cid)
        complex_dir = os.path.join(data_dir, cid)
        if not os.path.exists(complex_dir):
            os.makedirs(complex_dir, exist_ok=True)

        # lig_native_path = f"./data/pdbbind/renumber_atom_index_same_as_smiles/{cid}.sdf"
        # protein_path = f"./data/pdbbind/protein_remove_extra_chains_10A/{cid}_protein.pdb"
        lig_native_path = os.path.join(data_dir, 'ligand', f'{cid}.sdf')
        protein_path = os.path.join(data_dir, 'protein', f'{cid}_protein.pdb')

        if os.path.exists(os.path.join(complex_dir, f'Pocket_{distance}A.pdb')):
            continue

        pymol.cmd.load(protein_path)
        pymol.cmd.remove('resn HOH')
        pymol.cmd.load(lig_native_path)
        pymol.cmd.remove('hydrogens')
        pymol.cmd.select('Pocket', f'byres {cid} around {distance}')
        pymol.cmd.save(os.path.join(complex_dir, f'Pocket_{distance}A.pdb'), 'Pocket')
        pymol.cmd.delete('all')


def generate_complex_v1(data_dir, data_df, distance=8, input_ligand_format='sdf'):
    pbar = tqdm(total=len(data_df))
    for i, row in data_df.iterrows():
        
        cid, pKa = row['pdb'], float(row['affinity'])
        complex_dir = os.path.join(data_dir, cid)
        if not os.path.exists(complex_dir):
            os.makedirs(complex_dir, exist_ok=True)
        pocket_path = os.path.join(data_dir, cid, f'Pocket_{distance}A.pdb')
        # if os.path.exists(os.path.join(complex_dir, f"{cid}_{distance}A.rdkit")):
        #     continue
        if input_ligand_format != 'pdb':
            # ligand_input_path = os.path.join(data_dir, 'renumber_atom_index_same_as_smiles', f'{cid}.{input_ligand_format}')
            # ligand_input_path =f'./data/pdbbind/renumber_atom_index_same_as_smiles/{cid}.{input_ligand_format}'
            ligand_input_path = os.path.join(data_dir, 'ligand', f'{cid}.{input_ligand_format}')

        else:
            ligand_input_path = os.path.join(data_dir, cid, f'{cid}_ligand.pdb')

        save_path = os.path.join(complex_dir, f"{cid}_{distance}A.rdkit")
        ligand = Chem.MolFromMolFile(ligand_input_path, removeHs=True)
        if ligand == None:
            print(f"Unable to process ligand of {cid}")
            continue

        pocket = Chem.MolFromPDBFile(pocket_path,sanitize=False, removeHs=True)#
        if pocket == None:
            print(f"Unable to process protein of {cid}")
            continue

        complex = (ligand, pocket)
        with open(save_path, 'wb') as f:
            pickle.dump(complex, f)

        pbar.update(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ligand_format', type=str, default='sdf', help="Input ligand file format (currently supported is only .sdf)")
    parser.add_argument('--data_root', type=str, default='./data/toy_set/', help='Path to input data directory')
    parser.add_argument('--input_csv', type=str, default=None, required=False, help='Path to csv with columns "pdb","affinity"')
    args = parser.parse_args()
    
    distance = 8
    input_ligand_format = args.ligand_format
    data_root = args.data_root
    
    if args.input_csv is None:
        input_csv = os.path.join(data_root, 'toy_set.csv')
    else:
        input_csv = args.input_csv

    data_df = pd.read_csv(input_csv)

    # generate pocket within 8 Ångström around ligand 
    generate_pocket(data_dir=data_root,data_df=data_df, distance=distance)

    generate_complex_v1(data_root, data_df, distance=distance, input_ligand_format=input_ligand_format)


# %%
