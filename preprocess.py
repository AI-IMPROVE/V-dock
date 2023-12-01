import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Descriptors
import os

data_dir = os.environ['CANDLE_DATA_DIR'].rstrip('/')


def main():
    data_file = data_dir + '/sort_vina_smi_dock_score.txt'
    smiles_df = pd.read_csv(data_file, header=None,
                            names=['SMILES', 'Docking_Score'])
    smiles_df = smiles_df.set_index('SMILES')
    feature_array = []

    for smiles_string in smiles_df.index:
        try:
            feat = get_features(smiles_string)
            feature_array.append(feat)

        except:
            print(smiles_string)
            smiles_df = smiles_df.drop(index=smiles_string)

    col_names = ['RDKitFP' + str(x) for x in range(2048)]
    col_names += ['MACCSkey' + str(x) for x in range(167)]
    col_names += ['NumHBondDonors', 'NumHBondAcceptors',
                  'NumRotatableBonds', 'NumRings', 'NumHeteroAtoms',
                  'NumHeterocycles', 'NumLipinskiHBondAcceptors',
                  'NumLipinskiHBondDonors', 'NumAromaticCarbocycles',
                  'NumAromaticHeterocycles', 'NumAmideBonds',
                  'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
                  'FracCAtomsSP3Hybridized', 'LabuteAccessibleSurfaceArea',
                  'NumRadicalElectrons', 'TopologicalPolarSurfaceArea']

    for i in range(len(feature_array[0])):
        smiles_df[col_names[i]] = [x[i] for x in feature_array]

    print(smiles_df)
    smiles_df.to_csv(data_dir + '/smiles_plus_features.csv')


def get_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    rdk_fp = Chem.RDKFingerprint(mol, fpSize=2048)
    maccs_fp = Chem.MACCSkeys.GenMACCSKeys(mol)
    rdk_fp = [x for x in rdk_fp]
    maccs_fp = [x for x in maccs_fp]

    nhbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
    nhba = Chem.rdMolDescriptors.CalcNumHBA(mol)
    rbnd = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_ri = Chem.rdMolDescriptors.CalcNumRings(mol)
    heta = Chem.rdMolDescriptors.CalcNumHeteroatoms(mol)
    hetc = Chem.rdMolDescriptors.CalcNumHeterocycles(mol)
    lhba = Chem.rdMolDescriptors.CalcNumLipinskiHBA(mol)
    lhbd = Chem.rdMolDescriptors.CalcNumLipinskiHBD(mol)
    acar = Chem.rdMolDescriptors.CalcNumAromaticCarbocycles(mol)
    ahet = Chem.rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
    amdb = Chem.rdMolDescriptors.CalcNumAmideBonds(mol)
    alcc = Chem.rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)
    alhc = Chem.rdMolDescriptors.CalcNumAliphaticHeterocycles(mol)
    csp3 = Chem.rdMolDescriptors.CalcFractionCSP3(mol)
    lasa = Chem.rdMolDescriptors.CalcLabuteASA(mol)
    nrel = Descriptors.NumRadicalElectrons(mol)
    tpsa = Descriptors.TPSA(mol)

    feat = rdk_fp + maccs_fp
    feat += [nhbd, nhba, rbnd, n_ri, heta, hetc, lhba, lhbd, acar, ahet,
             amdb, alcc, alhc, csp3, lasa, nrel, tpsa]

    return feat


if __name__=="__main__":
    main()
