#!/bin/python3
# ===================================================================================
# PBEE - Protein Binding Energy Estimator
# Authors: Roberto D. Lins, Elton J. F. Chaves, and João Sartori
# Adapted: multi-chain partners + verbose status, no input deletions
# Added: --rebuild_missing toggle (PDBFixer/OpenMM) + rebuild step before Rosetta
# HTS: multiple inputs (.pdb, .zip, directories) + global ranked CSV
# ===================================================================================

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from modules.detect_ions  import *
from modules.detect_gaps  import *
from modules.superlearner import *
from modules.rosetta_descriptors import Get_descriptors

from shutil import which
from pyrosetta import *
import pandas as pd
import numpy as np
import os, sys, math, time, glob, shutil, argparse, subprocess, zipfile
from itertools import product
from datetime import datetime

# --- optional rebuild deps (PDBFixer/OpenMM) ---
try:
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
except Exception:
    PDBFixer = None
    PDBFile  = None

# ----------------------------- small utilities ------------------------------------

def print_infos(message, type):
    if type == 'info':      print(f'            info: {message}')
    if type == 'structure': print(f'       structure: {message}')
    if type == 'protocol':  print(f'        protocol: {message}')
    if type == 'none':      print(f' {message}')

def print_dG(mol, dG_pred, affinity):
    print_infos(message=f'[{mol}] ΔG[bind] = {dG_pred:.3f} kcal/mol (KD = {affinity} M)', type='protocol')

def print_end():
    exit('\n --- End process ---\n')

def processing_time(st):
    # keep predictable progress even if env is fast
    time.sleep(1)
    elapsed_time = time.time() - st
    return elapsed_time

def ispdb(pdbfile):
    try:
        with open(pdbfile, 'r') as file:
            count = sum(1 for line in file if line.startswith("ATOM"))
        return os.path.abspath(pdbfile) if count else False
    except FileNotFoundError:
        return False

def isdir(path):
    if os.path.isdir(path):
        return os.path.abspath(path)
    else:
        print_infos(message=f'error: path not found -> {path}', type='none'); print_end()

def istool(tool):
    return which(tool) is not None

def header(version):
    print('')
    print(' =====================================================')
    print('   Protein Engineering and Structural Genomic Group  ')
    print('          Oswaldo Cruz Foundation - FIOCRUZ          ')
    print(' -----------------------------------------------------')
    print('')
    print(' ********* Protein Binding Energy Estimator **********')
    print('')
    print(' Authors: Roberto Lins, Elton Chaves, and João Sartori')
    print('     DOI: 10.1021/acs.jcim.4c01641')
    print(f' Version: {version}')
    print(' =====================================================')
    print('')

def configure_PbeePATH():
    PbeePATH = os.path.dirname(__file__)
    if not os.path.isdir(PbeePATH):
        print(' error: invalid PbeePATH'); print_end()
    return PbeePATH

def configure_mlmodels(PbeePATH, version):
    trainedmodels = [
        f'{PbeePATH}/trainedmodels/{version}/{version}__basemodel_LinearRegression.pkl',
        f'{PbeePATH}/trainedmodels/{version}/{version}__basemodel_ElasticNet.pkl',
        f'{PbeePATH}/trainedmodels/{version}/{version}__basemodel_SVR.pkl',
        f'{PbeePATH}/trainedmodels/{version}/{version}__basemodel_DecisionTreeRegressor.pkl',
        f'{PbeePATH}/trainedmodels/{version}/{version}__basemodel_KNeighborsRegressor.pkl',
        f'{PbeePATH}/trainedmodels/{version}/{version}__basemodel_AdaBoostRegressor.pkl',
        f'{PbeePATH}/trainedmodels/{version}/{version}__basemodel_BaggingRegressor.pkl',
        f'{PbeePATH}/trainedmodels/{version}/{version}__basemodel_RandomForestRegressor.pkl',
        f'{PbeePATH}/trainedmodels/{version}/{version}__basemodel_ExtraTreesRegressor.pkl',
        f'{PbeePATH}/trainedmodels/{version}/{version}__basemodel_XGBRegressor.pkl'
    ]
    for item in trainedmodels:
        if not os.path.isfile(item):
            print(f' requirement not found: {item}'); print_end()
    return trainedmodels

# ---------------------------- chain handling helpers ------------------------------

def detect_chains(pdbfile):
    chains = []
    seen = set()
    with open(pdbfile, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                c = line[21]
                if c not in seen:
                    seen.add(c); chains.append(c)
    return chains  # preserve order of appearance

def partner_letters(s):
    return [c for c in s if c.isalnum()]

def partner_checker(pdbfile, partner1, partner2):
    """Return (ok, detected_chains, present1, present2)."""
    chains = detect_chains(pdbfile)
    set_chains = set(chains)
    p1_letters = set(partner_letters(partner1))
    p2_letters = set(partner_letters(partner2))
    present1 = sorted(list(p1_letters & set_chains))
    present2 = sorted(list(p2_letters & set_chains))
    ok = (len(present1) > 0) and (len(present2) > 0)
    return ok, chains, present1, present2

def choose_on_missing(on_missing_chains, chains):
    """When requested partners are missing, decide fallback partners (single letters)."""
    if on_missing_chains == 'abort':
        return None
    if on_missing_chains == 'list':
        print_infos(message=f'detected chains: {chains}', type='info')
        return None
    # first2
    if len(chains) >= 2:
        return chains[0], chains[1]
    return None

def compute_min_ca_distance(pdbfile, group1, group2):
    """Compute minimal CA-CA distance between any chain in group1 vs any in group2."""
    ca_by_chain = {}
    with open(pdbfile, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                ch = line[21]
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                except ValueError:
                    continue
                ca_by_chain.setdefault(ch, []).append((x,y,z))
    def _pairs(chs):
        pts = []
        for ch in chs:
            pts.extend(ca_by_chain.get(ch, []))
        return pts
    g1 = _pairs(group1); g2 = _pairs(group2)
    mind = None
    for (x1,y1,z1) in g1:
        for (x2,y2,z2) in g2:
            d = ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5
            if (mind is None) or (d < mind): mind = d
    return mind

# ------------------------------ file housekeeping ---------------------------------

def remove_files(files):
    for file in files:
        if isinstance(file, list):
            for item in file:
                try: os.remove(item)
                except: pass
        else:
            if os.path.exists(file):
                try: os.remove(file)
                except: pass

# -------------------------- Rosetta support wrappers ------------------------------

def preventing_errors(pdbfile, basename, outdir):
    # replace last TER by END and remove SSBOND lines, count atoms
    pdb1 = f'{outdir}/{basename}_jd2_01.pdb'
    with open(pdbfile, "r") as input_file, open(pdb1, "w") as output_file:
        lines = input_file.readlines()
        last_ter = max([i for i,l in enumerate(lines) if l.startswith("TER")] or [-1])
        for i,l in enumerate(lines):
            if i == last_ter:
                output_file.write("END" + l[3:])
            else:
                output_file.write(l)
    atoms = 0
    pdb2  = f'{outdir}/{basename}_jd2_02.pdb'
    with open(pdb1, 'r') as inp, open(pdb2, 'w') as out:
        for line in inp:
            if not line.startswith('SSBOND'):
                out.write(line)
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    atoms += 1
    return pdb2, atoms

def scorejd2(pdbfile, basename, outdir):
    # quiet Rosetta setup to produce a renumbered pose
    print_infos(message=f'Rosetta: importing pose from {os.path.basename(basename)}_jd2.pdb', type='protocol')
    sys_stdout, sys_stderr = sys.stdout, sys.stderr
    sys.stdout = open(os.devnull, 'w'); sys.stderr = open(os.devnull, 'w')
    pyrosetta.init(extra_options=" -corrections::beta_nov16 true -ignore_unrecognized_res -output_pose_energies_table false -renumber_pdb")
    pose = rosetta.core.import_pose.pose_from_file(pdbfile)
    pose.dump_pdb(f'{outdir}/{basename}_jd2_0001.pdb')
    sys.stdout = sys_stdout; sys.stderr = sys_stderr
    print_infos(message=f'Rosetta: wrote {os.path.basename(basename)}_jd2_0001.pdb', type='protocol')
    return f'{outdir}/{basename}_jd2_0001.pdb'

# -------------------------- PBEE pipeline (clean/concat) --------------------------

def pdbcleaner(pdbfile, basename, outdir, submit_dir, partner1, partner2):
    # clean_pdb.py in PBEE supports multi-letter chain lists; pass exactly as given
    commands = [
        f'python {PbeePATH}/modules/clean_pdb.py {pdbfile} {partner1}',
        f'python {PbeePATH}/modules/clean_pdb.py {pdbfile} {partner2}',
        f'mv {submit_dir}/{basename}_{partner1}.pdb {outdir}',
        f'mv {submit_dir}/{basename}_{partner2}.pdb {outdir}',
        f'mv {submit_dir}/{basename}_*.fasta {outdir}'
    ]
    for command in commands:
        subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    return f'{outdir}/{basename}_{partner1}.pdb', f'{outdir}/{basename}_{partner2}.pdb'

def concat_pdbs(outdir, basename, partner1_path, partner2_path):
    outfile = f'{outdir}/{basename}_jd2.pdb'
    with open(partner1_path, 'r') as f1, open(partner2_path, 'r') as f2, open(outfile, 'w') as out:
        out.write(f1.read()); out.write(f2.read())
    return outfile

# ------------------------------ ML glue -------------------------------------------

def detect_outliers(x, rosetta_features, mol):
    count = 0
    for col in x.columns:
        mu = x[col].mean(); sd = x[col].std()
        for _, row in rosetta_features.iterrows():
            sup = row[col] > mu + 4*sd
            inf = row[col] < mu - 4*sd
            if sup or inf:
                print_infos(message=f'{[mol]} outlier -> {col} = {row[col]}', type='protocol')
                count += 1
    return count

def calc_affinity(dG):
    T = 298.15; R = 8.314
    dG_J = dG * 4184.0
    affinity = float(f'{math.exp(dG_J / (R * T)):.6e}')
    return affinity

def sl_predictions(X_test, models, meta_model):
    model_predictions = {}
    for name, model in models:
        yhat = model.predict(X_test)
        model_predictions[name] = yhat
    meta_X = np.column_stack(list(model_predictions.values()))
    super_learner_preds = meta_model.predict(meta_X)
    model_predictions["sl"] = super_learner_preds
    return model_predictions

def train_base_models(x, y, models):
    trained_models = []
    for model in models:
        model.fit(x, y)
        trained_models.append((model.__class__.__name__, model))
    return trained_models

def predictor(trainedmodels, mlengine, mlmodel, x, y, rosetta_features, columns_to_remove):
    with open(mlmodel, 'rb') as f:
        meta_model = joblib.load(f)
    if mlengine != 'sl':
        model_predictions = {}
        yhat = meta_model.predict(rosetta_features.values)
        model_predictions[mlengine] = yhat
    else:
        models = [joblib.load(filename) for filename in trainedmodels]
        base_models = train_base_models(x, y, models)
        model_predictions = sl_predictions(rosetta_features, base_models, meta_model)
    return model_predictions[mlengine][0]

# ------------------------------ Pre/Post stages -----------------------------------

def pre_processing(pdbfiles, partner1, partner2, on_missing_chains, strict_chains):
    bad_structures = []
    resolved_partners = {}  # map pdb -> (p1_used, p2_used)

    for mol, pdb in enumerate(pdbfiles):
        basename = os.path.basename(pdb[:-4])
        outdir = f'{args.odir[0]}/pbee_outputs/{basename}'

        condition = ispdb(pdb)
        if condition:
            print_infos(message=f'[{mol}] {pdb}', type='structure')
            if not os.path.isdir(outdir): os.makedirs(outdir)
        else:
            print_infos(message=f'invalid PDB file -> {os.path.basename(pdb)}.', type='structure')
            continue

        ok, chains, present1, present2 = partner_checker(pdb, partner1, partner2)
        print_infos(message=f'[{mol}] detected chains: {chains}', type='info')
        if not ok:
            if strict_chains:
                print_infos(message=f'[{mol}] missing requested partner chains; skipping (strict_chains=True).', type='info')
                bad_structures.append(pdb)
                shutil.rmtree(outdir, ignore_errors=True)
                continue
            choice = choose_on_missing(on_missing_chains, chains)
            if choice is None:
                print_infos(message=f'[{mol}] argument error (--partner1/--partner2): chain ID not found ({set(partner_letters(partner1)) | set(partner_letters(partner2))})', type='info')
                bad_structures.append(pdb)
                shutil.rmtree(outdir, ignore_errors=True)
                continue
            else:
                print_infos(message=f'[{mol}] auto-selected partners -> {choice[0]}/{choice[1]} (policy={on_missing_chains})', type='protocol')
                p1_use, p2_use = choice[0], choice[1]
        else:
            p1_use = ''.join(present1) if len(present1) > 1 else present1[0]
            p2_use = ''.join(present2) if len(present2) > 1 else present2[0]

        # Clean and split per resolved partners (supports multi-letter)
        partners = pdbcleaner(pdb, basename, outdir, submit_dir, p1_use, p2_use)

        # gaps
        gaps = [detect_gaps(partners[0]), detect_gaps(partners[1])]
        total_gaps = sum(1 for g in gaps if g != 0)
        if total_gaps > 0:
            print_infos(message=f'[{mol}] warning: {sum(gaps)} gap(s) found.', type='info')
            if not frcmod_struct:
                bad_structures.append(pdb)
                shutil.rmtree(outdir, ignore_errors=True)
                continue

        # report min CA–CA for the *resolved* partner groups
        g1 = list(partner_letters(p1_use))
        g2 = list(partner_letters(p2_use))
        mind = compute_min_ca_distance(pdb, g1, g2)
        if mind is not None:
            print_infos(message=f'[{mol}] min CA–CA {"".join(g1)}/{ "".join(g2)} ≈ {mind:.2f} Å', type='protocol')

        resolved_partners[pdb] = (p1_use, p2_use)

    return bad_structures, resolved_partners

def maybe_rebuild_structure(pdb_path, outdir, basename):
    """
    If --rebuild_missing was requested and PDBFixer/OpenMM are available,
    rebuild missing residues/atoms and write <basename>_rebuilt.pdb.
    Returns the path to the structure to continue with.
    """
    if not getattr(args, "rebuild_missing", False):
        return pdb_path

    if PDBFixer is None or PDBFile is None:
        print_infos(message='Rebuild skipped: PDBFixer/OpenMM not available', type='protocol')
        return pdb_path

    try:
        print_infos(message='Rebuilding missing residues/atoms with PDBFixer', type='protocol')
        fixer = PDBFixer(filename=pdb_path)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)  # neutral pH

        rebuilt = f'{outdir}/{basename}_rebuilt.pdb'
        with open(rebuilt, 'w') as fh:
            PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)

        print_infos(message=f'Wrote rebuilt structure -> {os.path.basename(rebuilt)}', type='protocol')
        return rebuilt

    except Exception as e:
        print_infos(message=f'Rebuild skipped due to error: {e}', type='protocol')
        return pdb_path

def post_processing(pdbfiles, resolved_partners, trainedmodels, mlmodel, st):
    summary_rows = []
    for mol, pdb in enumerate(pdbfiles):
        basename = os.path.basename(pdb[:-4])
        outdir = f'{args.odir[0]}/pbee_outputs/{basename}'

        p1_use, p2_use = resolved_partners.get(pdb, (None, None))
        if p1_use is None:
            # was skipped in pre
            continue

        print_infos(message=f'[{mol}] {pdb}', type='protocol')
        partners = [
            f'{outdir}/{basename}_{p1_use}.pdb',
            f'{outdir}/{basename}_{p2_use}.pdb'
        ]
        _pdb = concat_pdbs(outdir, basename, partners[0], partners[1])

        # Detect ions near ANY chain in both partner groups
        chain_list_for_ions = list(set(list(partner_letters(p1_use)) + list(partner_letters(p2_use))))
        ions = detect_ions(pdb, cutoff=ion_dist_cutoff, chains=chain_list_for_ions)
        print_infos(message=f'[{mol}] total number of ions: {len(ions)}', type='protocol')
        if len(ions) != 0:
            with open(_pdb, 'r') as f:
                lines = f.readlines()
            for ion in ions:
                for i, line in enumerate(lines):
                    if line.startswith('ATOM') and line[21] == ion[1][21]:
                        index = i
                lines.insert(index + 1, ion[1])
            with open(_pdb, 'w') as f:
                f.writelines(lines)

        # Rebuild before Rosetta if requested
        _pdb = maybe_rebuild_structure(_pdb, outdir, basename)

        _pdb = scorejd2(_pdb, basename, outdir)
        _pdb, total_atoms = preventing_errors(_pdb, basename, outdir)

        # descriptors & prediction
        train_file = f"{PbeePATH}/trainedmodels/{version}/{version}__pbee_train_file.csv"
        train_cols = pd.read_csv(train_file).drop(columns=['pdb','database','partner1','partner2','dG_exp'])
        x_train    = pd.read_csv(train_file, delimiter=',').drop(columns=['pdb','database','partner1','partner2','dG_exp'])
        y_train    = pd.read_csv(train_file, delimiter=',')['dG_exp']

        if not os.path.isfile(f'{outdir}/dG_pred.csv'):
            print_infos(message=f'[{mol}] geometry optimization and interface analysis', type='protocol')
            pose, rosetta_features = Get_descriptors(_pdb, ions, outdir, basename, p1_use, p2_use)
            selected = [c for c in train_cols if c in rosetta_features.columns]
            rosetta_features = rosetta_features[selected]

            if len(ions) != 0:
                pose.dump_pdb(f'{outdir}/{basename}_ions_rlx.pdb')
            else:
                pose.dump_pdb(f'{outdir}/{basename}_rlx.pdb')

            condition = not rosetta_features.applymap(lambda x: '-nan' in str(x)).any().any()
            if (condition is False) or ('ifa_sc_value' in rosetta_features.columns and rosetta_features['ifa_sc_value'][0] == -1):
                print_infos(message=f'[{mol}] an incorrect descriptor was found, ignoring the structure to avoid errors', type='protocol')
                continue

            if not frcmod_scores:
                outliers = detect_outliers(x_train, rosetta_features, mol)
                if outliers != 0: continue
        else:
            rosetta_features = pd.read_csv(f'{outdir}/dG_pred.csv', delimiter=',')
            selected = [c for c in train_cols if c in rosetta_features.columns]
            rosetta_features = rosetta_features[selected]

        print_infos(message=f'[{mol}] calculating ΔG[bind]', type='protocol')
        dG_pred = predictor(trainedmodels, mlengine, mlmodel, x_train, y_train, rosetta_features, [])
        affinity = calc_affinity(dG_pred)
        print_dG(mol, dG_pred, affinity)
        total_time = processing_time(st)

        rosetta_features.insert(0, 'pdb',             basename)
        rosetta_features.insert(1, 'dG_pred',         dG_pred)
        rosetta_features.insert(2, 'affinity',        affinity)
        rosetta_features.insert(3, 'mlengine',        mlengine)
        rosetta_features.insert(4, 'total_atoms',     total_atoms)
        rosetta_features.insert(5, 'processing_time', total_time)
        rosetta_features.to_csv(f'{outdir}/dG_pred.csv', index=False)

        # accumulate to summary
        summary_rows.append({
            "pdb": basename,
            "dG_pred": dG_pred,
            "KD_M": affinity,
            "mlengine": mlengine,
            "total_atoms": total_atoms,
            "processing_time": total_time
        })

        # remove only temporary JD2 files
        remove_files(files=[
            glob.glob(f'{outdir}/*fasta'),
            f'{outdir}/{basename}_jd2_01.pdb',
            f'{outdir}/{basename}_jd2_02.pdb'
        ])

    return summary_rows

# ------------------------------ Input expansion (HTS) -----------------------------

def collect_pdbs_from_dir(root):
    pdbs = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".pdb"):
                full = os.path.join(dirpath, fn)
                if ispdb(full):
                    pdbs.append(os.path.abspath(full))
    return pdbs

def expand_inputs(inputs, odir):
    """
    Accepts files (.pdb or .zip) and/or directories.
    - If .zip is provided, extracts to odir/hts_inputs_<ts>/zip_<i> and collects .pdb files only.
    - If directory, collects .pdb recursively.
    - If .pdb, adds directly.
    Returns: list of absolute paths to valid PDB files.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    hts_root = os.path.join(odir, f"hts_inputs_{ts}")
    os.makedirs(hts_root, exist_ok=True)

    all_pdbs = []

    for i, item in enumerate(inputs):
        item = os.path.abspath(item)
        if os.path.isdir(item):
            # directory of pdbs
            pdbs = collect_pdbs_from_dir(item)
            print_infos(message=f'found {len(pdbs)} PDB(s) in dir -> {item}', type='info')
            all_pdbs.extend(pdbs)
        elif item.lower().endswith(".zip"):
            # extract zip
            dest = os.path.join(hts_root, f"zip_{i}")
            os.makedirs(dest, exist_ok=True)
            try:
                with zipfile.ZipFile(item, 'r') as zf:
                    zf.extractall(dest)
                pdbs = collect_pdbs_from_dir(dest)
                print_infos(message=f'extracted {len(pdbs)} PDB(s) from zip -> {item}', type='info')
                all_pdbs.extend(pdbs)
            except zipfile.BadZipFile:
                print_infos(message=f'bad zip file (skipped) -> {item}', type='info')
        elif item.lower().endswith(".pdb"):
            if ispdb(item):
                all_pdbs.append(item)
            else:
                print_infos(message=f'non-PDB or empty ATOM records (skipped) -> {item}', type='info')
        else:
            print_infos(message=f'unsupported input (skipped) -> {item}', type='info')

    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for p in all_pdbs:
        if p not in seen:
            seen.add(p); ordered.append(p)
    return ordered

# ----------------------------------- main -----------------------------------------

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    version   = 'v1.1'
    st        = time.time()
    submit_dir= os.getcwd()
    header(version)

    PbeePATH = configure_PbeePATH()
    trainedmodels = configure_mlmodels(PbeePATH, version)
    mlmodels = {
        'sl': f'{PbeePATH}/trainedmodels/{version}/{version}__SuperLearner.pkl',
        'lr': trainedmodels[0],
        'en': trainedmodels[1],
        'sv': trainedmodels[2],
        'dt': trainedmodels[3],
        'kn': trainedmodels[4],
        'ad': trainedmodels[5],
        'bg': trainedmodels[6],
        'rf': trainedmodels[7],
        'et': trainedmodels[8],
        'xb': trainedmodels[9]
    }

    parser = argparse.ArgumentParser(add_help=True)
    mandatory = parser.add_argument_group('mandatory arguments')
    # Backward compatible flag (still supported)
    mandatory.add_argument('--ipdb', nargs='*', type=str, metavar='',
                           help='[deprecated] input file(s) in PDB format (supports globs). Prefer --inputs.')
    # New unified inputs: file(s)/dir(s)/zip(s)
    mandatory.add_argument('--inputs', nargs='*', type=str, metavar='',
                           help='path(s) to .pdb, .zip, or directory containing PDBs (recursive).')

    mandatory.add_argument('--partner1', nargs=1, type=str, required=True, metavar='',
                           help='str | chain ID(s) for partner1 (e.g., H or HL)')
    mandatory.add_argument('--partner2', nargs=1, type=str, required=True, metavar='',
                           help='str | chain ID(s) for partner2 (e.g., L or AB)')

    parser.add_argument('--odir', nargs=1, type=isdir, default=[submit_dir], metavar='',
                        help=f'str | output directory (default={submit_dir})')
    parser.add_argument('--mlengine', nargs=1, type=str, default=['sl'],
                        choices=['sl','lr','en','sv','dt','kn','ad','bg','rf','et','xb'], metavar='',
                        help='str | choose ML engine')
    parser.add_argument('--ion_dist_cutoff', nargs=1, type=float, default=[2], metavar='',
                        help='float | cutoff distance (Å) to detect ions (default=2)')
    parser.add_argument('--frcmod_struct', action='store_true', help='ignore warnings about gaps')
    parser.add_argument('--frcmod_scores', action='store_true', help='ignore warnings about low-quality descriptors')
    parser.add_argument('--on_missing_chains', nargs=1, type=str, default=['abort'],
                        choices=['abort','list','first2'],
                        help='what to do if requested partner chains are not found')
    parser.add_argument('--list_models', action='store_true', help='list available ML models and exit if no inputs')

    # NEW FLAGS
    parser.add_argument('--rebuild_missing', action='store_true',
                        help='Rebuild missing residues/atoms with PDBFixer/OpenMM before Rosetta (requires openmm & pdbfixer).')
    parser.add_argument('--strict_chains', action='store_true',
                        help='Require both partner1 and partner2 chains to be present; otherwise skip. Recommended for HTS.')

    args            = parser.parse_args()
    partner1        = args.partner1[0].upper().strip()
    partner2        = args.partner2[0].upper().strip()
    odir            = args.odir[0]
    mlengine        = args.mlengine[0]
    mlmodel         = mlmodels[mlengine]
    ion_dist_cutoff = float(args.ion_dist_cutoff[0])
    frcmod_struct   = args.frcmod_struct
    frcmod_scores   = args.frcmod_scores
    on_missing_chains = args.on_missing_chains[0]
    strict_chains   = args.strict_chains

    # Resolve inputs (new --inputs preferred; fallback to --ipdb)
    input_args = []
    if args.inputs: input_args.extend(args.inputs)
    if args.ipdb:   input_args.extend(args.ipdb)

    if len(input_args) == 0:
        print_infos(message='no inputs provided. Use --inputs with .pdb/.zip/dir, or --ipdb.', type='info'); print_end()

    # Expand globs and normalize file/dir paths
    expanded_args = []
    for pat in input_args:
        if any(ch in pat for ch in ['*','?','[']):
            expanded_args.extend(glob.glob(pat))
        else:
            expanded_args.append(pat)

    # Build final pdb list from all supported inputs
    inputs_expanded = expand_inputs(expanded_args, odir)
    pdbfiles = [p for p in inputs_expanded if ispdb(p)]
    if len(pdbfiles) == 0:
        print_infos(message=f'no valid PDB files found from inputs: {expanded_args}', type='info'); print_end()

    # Show chosen config
    print(f'        mlengine: {mlmodel}')
    print(f'      output_dir: {odir}')
    print(f'        partner1: {partner1}')
    print(f'        partner2: {partner2}')
    print(f' ion_dist_cutoff: {ion_dist_cutoff}')
    if frcmod_struct: print(f'   frcmod_struct: {frcmod_struct}')
    if frcmod_scores: print(f'   frcmod_scores: {frcmod_scores}')
    if args.rebuild_missing: print(f'  rebuild_missing: {args.rebuild_missing}')
    if strict_chains: print(f'    strict_chains: {strict_chains}')
    print('')

    if args.list_models:
        print('\n Available ML engines and model files:')
        for key, path in mlmodels.items():
            print(f'  - {key:2s} : {path}  (exists: {os.path.isfile(path)} )')
        print('')

    # Enumerate to user
    print_infos(message=f'will process {len(pdbfiles)} file(s):', type='info')
    for i, p in enumerate(pdbfiles[:50]):  # avoid flooding
        print_infos(message=f'  [{i}] {p}', type='info')
    if len(pdbfiles) > 50:
        print_infos(message=f'  ... and {len(pdbfiles)-50} more', type='info')

    # Pre
    bad_structures, resolved = pre_processing(pdbfiles, partner1, partner2, on_missing_chains, strict_chains)
    worklist = [p for p in pdbfiles if p not in bad_structures]

    print_infos(message=f'total structures: {len(worklist)}', type='info')
    if len(worklist) != 0:
        summary_rows = post_processing(worklist, resolved, trainedmodels, mlmodel, st)

        # Write global ranked CSV
        summary_dir = os.path.join(odir, "pbee_outputs", "_summary")
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, "pbee_ranked.csv")

        if len(summary_rows) == 0:
            print_infos(message='no successful predictions to summarize.', type='info')
        else:
            df = pd.DataFrame(summary_rows)
            # rank best (most negative ΔG) on top
            df = df.sort_values(by="dG_pred", ascending=True)
            df.to_csv(summary_path, index=False)
            print_infos(message=f'Wrote ranked summary CSV -> {summary_path}', type='protocol')

            # Print CSV contents (full)
            try:
                # Avoid scientific notation for KD when printing
                with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.float_format", "{:,.6f}".format):
                    print(df.to_string(index=False))
            except Exception as e:
                print_infos(message=f'could not print summary table: {e}', type='info')

        elapsed_time = processing_time(st)
        print(' processing time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        print_end()
    else:
        print_infos(message='nothing to do', type='info'); print_end()
