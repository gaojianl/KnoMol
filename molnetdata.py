import networkx as nx
import numpy as np
import pandas as pd
import os
import argparse
from multiprocessing import Pool
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from TFM.Dataset import MolNet
remover = SaltRemover()
smile_graph = {}
meta = ['W', 'U', 'Zr', 'He', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Rb', 'Sr', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Gd', 'Tb', 'Ho', 'W', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac']


def atom_features(atom, use_chirality=True):
    res = one_of_k_encoding_unk(atom.GetSymbol(),['C','N','O','F','P','S','Cl','Br','I','B','Si','Unknown']) + \
                    one_of_k_encoding(atom.GetDegree(), [1, 2, 3, 4, 5, 6]) + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) + [atom.GetIsAromatic(), atom.IsInRing()] + one_of_k_encoding_unk(atom.GetFormalCharge(), [-1,0,1,3]) + \
                    one_of_k_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2])
    if use_chirality:
        try:
            res = res + one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            bonds = atom.GetBonds()
            for bond in bonds:
                if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and str(bond.GetStereo()) in ["STEREOZ", "STEREOE"]:
                    res = res + one_of_k_encoding_unk(str(bond.GetStereo()), ["STEREOZ", "STEREOE"]) + [atom.HasProp('_ChiralityPossible')]
            if len(res) == 34:
                res = res + [False, False] + [atom.HasProp('_ChiralityPossible')]
    return np.array(res)          # 37


def order_gnn_features(bond):
    weight = [0.3, 0.4, 0.5, 0.36]
    bt = bond.GetBondType()
    bond_feats = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC]

    for i, m in enumerate(bond_feats):
        if m == True and i != 0:
            b = weight[i]
        elif m == True and i == 0:
            if bond.GetIsConjugated() == True:
                b = 0.32
            else:
                b = 0.3
        else:pass
    return b             


def order_tf_features(bond):
    weight = [0.8, 0.9, 1., 0.85]
    bt = bond.GetBondType()
    bond_feats = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC]
    for i, m in enumerate(bond_feats):
        if m == True:
            b = weight[i]
    return b       


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smiletopyg(smi):
    g = nx.Graph()
    mol = Chem.MolFromSmiles(smi)
    c_size = mol.GetNumAtoms()

    features = []
    hete_mask = []
    for i, atom in enumerate(mol.GetAtoms()):
        g.add_node(i)
        feature = atom_features(atom)
        features.append((feature / sum(feature)).tolist()) 

        if atom.GetSymbol() == 'C':
            hete_mask.append(0)
        else:
            hete_mask.append(1)
    hete_mask = np.asarray(hete_mask)


    ssr = Chem.GetSymmSSSR(mol) 
    ring_masm = np.zeros((c_size))
    arom_masm = np.zeros((c_size))
    alip_masm = np.zeros((c_size))
    aroml = []
    for i in range(0, len(ssr)): 
        aromring = True
        for r in list(ssr[i]):
            atom = mol.GetAtomWithIdx(r)
            if not atom.GetIsAromatic():
                aromring = False
        aroml.append(aromring)
    
    inter_arom, inter_alip = [], []
    for i in range(0, len(ssr)): 
        if aroml[i] and (i not in inter_arom):
            for r in list(ssr[i]):
                ring_masm[r] = i+1
                arom_masm[r] = i+1
            for j in range(i+1,len(ssr)):
                if aroml[j] and bool(set(ssr[i]) & set(ssr[j])):
                    inter_arom.append(j)
                    for r in list(ssr[j]):
                        ring_masm[r] = i+1
                        arom_masm[r] = i+1
        elif (not aroml[i]) and (i not in inter_alip):
            for r in list(ssr[i]):
                ring_masm[r] = i+1
                alip_masm[r] = i+1
            for j in range(i+1,len(ssr)):
                if (not aroml[j]) and bool(set(ssr[i]) & set(ssr[j])):
                    inter_alip.append(j)
                    for r in list(ssr[j]):
                        ring_masm[r] = i+1
                        alip_masm[r] = i+1
        else:pass

    c = []
    adj_order_matrix = np.eye(c_size)
    adj_order_matrix = adj_order_matrix * 0.8
    dis_order_matrix = np.zeros((c_size,c_size))
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        bfeat = order_gnn_features(bond)
        g.add_edge(a1, a2, weight=bfeat)
        tfft = order_tf_features(bond)
        adj_order_matrix[a1, a2] = tfft
        adj_order_matrix[a2, a1] = tfft
        if bond.GetIsConjugated():
            c = list(set(c).union(set([a1, a2])))

    g = g.to_directed()
    edge_index = np.array(g.edges).tolist()

    edge_attr = []
    for w in list(g.edges.data('weight')):
        edge_attr.append(w[2])

    for i in range(c_size):
        for j in range(i,c_size):
            if adj_order_matrix[i, j] == 0 and i != j:
                conj = False
                paths = list(nx.node_disjoint_paths(g, i, j))
                if len(paths) > 1: 
                    paths = sorted(paths, key=lambda i:len(i),reverse=False)
                for path in paths:
                    if set(path) < set(c):
                        conj = True
                        break
                if conj:
                    adj_order_matrix[i, j] = 0.825
                    adj_order_matrix[j, i] = 0.825
                else:
                    path = paths[0]
                    dis_order_matrix[i, j] = len(path) - 1
                    dis_order_matrix[j, i] = len(path) - 1

    g = [c_size, features, edge_index, edge_attr, ring_masm, arom_masm, alip_masm, hete_mask, adj_order_matrix, dis_order_matrix]
    return [smi, g]


def write(res):
    smi, g = res
    smile_graph[smi] = g


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TransFoxMol')
    parser.add_argument('--moldata', type=str, help='dataset name to process')
    parser.add_argument('--task', type=str, choices=['clas', 'reg'], help='Binary classification or Regression')
    parser.add_argument('--numtasks', type=int, default=1, help='Number of tasks (default: 1). ')
    parser.add_argument('--ncpu', type=int, default=4, help='number of cpus to use (default: 4)')
    args = parser.parse_args()

    moldata = args.moldata
    if moldata in ['esol', 'freesolv', 'lipo', 'qm7', 'qm8', 'qm9']:
        task = 'reg'
        if moldata == 'qm8':
            numtasks = 12
        elif moldata == 'qm9':
            numtasks = 3
        else:
            numtasks = 1
    elif moldata in ['bbbp', 'sider', 'clintox', 'tox21', 'toxcast', 'bace', 'pcba', 'muv', 'hiv']:
        task = 'clas'
        if moldata == 'sider':
            numtasks = 27
        elif moldata == 'clintox':
            numtasks = 2
        elif moldata == 'tox21':
            numtasks = 12
        elif moldata == 'toxcast':
            numtasks = 617
        elif moldata == 'pcba':
            numtasks = 128
        elif moldata == 'muv':
            numtasks = 17
        else:
            numtasks = 1
    else:
        task = args.task
        numtasks = args.numtasks

    processed_data_file = 'dataset/processed/' + moldata+task + '_pyg.pt'
    if not os.path.isfile(processed_data_file):
        try:
            df = pd.read_csv('./dataset/raw/'+moldata+'.csv')
        except:
            print('Raw data not found! Put the right raw csvfile in **/dataset/raw/')
        try:
            compound_iso_smiles = np.array(df['smiles']) 
        except:
            print('The smiles column does not exist')
        try:
            ic50s = np.array(df.iloc[:, 1:numtasks+1])
        except:
            print('Mismatch between number of tasks and .csv file')
        #ic50s = -np.log10(np.array(ic50s))
        pool = Pool(args.ncpu)
        smis = []
        y = []
        result = []

        for smi, label in zip(compound_iso_smiles, ic50s):
            record = True
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                if '.' in smi:
                    mol = remover.StripMol(mol)
            if mol is not None:
                if 80 > mol.GetNumAtoms() > 1:
                    smi = Chem.MolToSmiles(mol)
                    if '.' not in smi:
                        record = True
                    else:
                        record = False
                else:
                    record = False
                for ele in meta:
                    if ele in smi:
                        record = False
                if record:
                    smis.append(smi) 
                    y.append(label)             
                    result.append(pool.apply_async(smiletopyg, (smi,)))
                else:
                    print(smi)
        pool.close()
        pool.join()

        for res in result:
            smi, g = res.get()
            smile_graph[smi] = g

        MolNet(root='./dataset', dataset=moldata+task, xd=smis, y=y, smile_graph=smile_graph)
    