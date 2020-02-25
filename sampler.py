#!/usr/bin/env python

import numpy as np
from cspy.linalg import kabsch
import random
from scipy.spatial import cKDTree as KDTree
from scipy.spatial.transform import Rotation as R
import copy
from numpy import sin as sn, cos as cs, radians as rad
import meshzoo
from os import path, mkdir
from cspy.crystal import Crystal, AsymmetricUnit, UnitCell, SpaceGroup, symmetry_operation
from copy import deepcopy
from tqdm import tqdm
import sys
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed


number_of_cores = 1
pool = ProcessPoolExecutor(number_of_cores)

atoms_dict = {1: 'H',
              6: 'C',
              7: 'N',
              8: 'O',
              'H': 1,
              'C': 6,
              'N': 7,
              'O': 8}

bonds_dictionary = {('C', 'H'): 1.28,
                     ('C', 'C'): 1.65,
                     ('C', 'N'): 1.55,
                     ('C', 'O'): 1.55,
                     ('C', 'F'): 1.45,
                     ('C', 'S'): 2.0,
                     ('C', 'Cl'): 1.85,
                     ('C', 'Br'): 1.95,
                     ('N', 'H'): 1.2,
                     ('N', 'N'): 1.55,
                     ('N', 'O'): 1.55,
                     ('N', 'S'): 1.8,
                     ('N', 'Hg'): 2.8,
                     ('O', 'H'): 1.3,
                     ('O', 'O'): 1.7,
                     ('O', 'S'): 1.6,
                     ('B', 'F'): 1.45,
                     ('B', 'C'): 1.65,
                     ('I', 'Hg'): 2.8,
                     ('Br', 'Hg'): 2.5,
                     ('S', 'S'): 2.5,
                     ('H', 'H'): 0.85,
                     ('C', 'I'): 2.2,
                     ('S', 'Cl'): 1.7,
                     ('S', 'F'): 1.6,
                     ('Cl', 'H'): 1.35,
                     ('Br', 'Br'): 2.35,
                     ('Cl', 'Br'): 2.2,
                     ('F', 'Br'): 2.6,
                     ('H', 'Br'): 1.5,
                     ('X', 'H'): 0.85,
                     ('X', 'C'): 1.28,
                     ('X', 'N'): 1.2,
                     ('X', 'O'): 1.3,
                     ('X', 'Cl'): 1.35,
                     ('X', 'Br'): 1.5,
                     ('X', 'X'): 0.85,
                     ('H', 'C'): 1.28,
                     ('N', 'C'): 1.55,
                     ('O', 'C'): 1.55,
                     ('F', 'C'): 1.45,
                     ('S', 'C'): 2.0,
                     ('Cl', 'C'): 1.85,
                     ('Br', 'C'): 1.95,
                     ('H', 'N'): 1.2,
                     ('O', 'N'): 1.55,
                     ('S', 'N'): 1.8,
                     ('Hg', 'N'): 2.8,
                     ('H', 'O'): 1.3,
                     ('S', 'O'): 1.6,
                     ('F', 'B'): 1.45,
                     ('C', 'B'): 1.65,
                     ('Hg', 'I'): 2.8,
                     ('Hg', 'Br'): 2.5,
                     ('I', 'C'): 2.2,
                     ('Cl', 'S'): 1.7,
                     ('F', 'S'): 1.6,
                     ('H', 'Cl'): 1.35,
                     ('Br', 'Cl'): 2.2,
                     ('Br', 'F'): 2.6,
                     ('Br', 'H'): 1.5,
                     ('H', 'X'): 0.85,
                     ('C', 'X'): 1.28,
                     ('N', 'X'): 1.2,
                     ('O', 'X'): 1.3,
                     ('Cl', 'X'): 1.35,
                     ('Br', 'X'): 1.5}


def tqdm_enumerate(iter):
    i = 0
    for y in tqdm(iter):
        yield i, y
        i += 1


def rotation(crys, x, y, z, angle):
    cos_a_2 = np.cos(angle / 2)
    sin_a_2 = np.sin(angle / 2)
    r = R.from_quat([sin_a_2 * x,
                     sin_a_2 * y,
                     sin_a_2 * z,
                     cos_a_2])
    mol = crys.asymmetric_unit
    pos_cart = crys.to_cartesian(mol.positions)
    centroid = np.mean(pos_cart, axis = 0)
    
    crys.asymmetric_unit.positions = crys.to_fractional(r.apply(pos_cart - centroid) + centroid)
    for attribute in ["_unit_cell_atom_dict",
                      "_uc_graph",
                      "_unit_cell_molecules",
                      "_symmetry_unique_molecules"]:
        if hasattr(crys, attribute):
            crys.__delattr__(attribute)
    return crys


def volume_change(crys, P):
    """
    Expand the whole crystal, including unit cell lengths and positions of
    molecules by a fractional.
    *** Note that this implementation is for Z=1 cases
    """
    lengths = crys.unit_cell.parameters[:3]
    angles = crys.unit_cell.parameters[-3:]
    mol = crys.asymmetric_unit
    cen_frac = np.mean(mol.positions, axis = 0)
    mol_cart = crys.to_cartesian(mol.positions - cen_frac)
    new_unit_cell = UnitCell.from_lengths_and_angles(lengths * (1 + P),
                                                 np.radians(angles))

    crys_new = Crystal(new_unit_cell, 
                        deepcopy(crys.space_group), 
                        deepcopy(crys.asymmetric_unit))
    crys_new.asymmetric_unit.positions = crys_new.to_fractional(mol_cart) - \
            np.mean(crys_new.to_fractional(mol_cart), axis = 0) + cen_frac
    return crys_new


def rotation_finder(crysA, crysB):
    """
    Finds the rotation that connects the asymmetric unit of crysA and crysB,
    when centroid of the corresponding asymmetric units are shifted to (0,0,0)
    """
    molA_frac = crysA.asymmetric_unit.positions
    molA_cart = crysA.to_cartesian(molA_frac)
    asymm_u_A = molA_cart - np.mean(molA_cart, axis = 0)
    molB_frac = crysB.asymmetric_unit.positions
    molB_cart = crysB.to_cartesian(molB_frac) 
    asymm_u_B = molB_cart - np.mean(molB_cart, axis = 0)
    
    # Rotation Matrix
    RM = kabsch.kabsch_rotation_matrix(asymm_u_A,asymm_u_B)
    
    # Converting Rotation Matrix to quaternion
#     w = np.sqrt(1.0 + RM[0,0] + RM[1,1] + RM[2,2]) / 2.0
#     w4 = (4.0 * w)
#     x = (RM[2,1] - RM[1,2]) / w4
#     y = (RM[0,2] - RM[2,0]) / w4
#     z = (RM[1,0] - RM[0,1]) / w4
    x, y, z, w = R.from_dcm(RM).as_quat()
    angle = np.arccos(w) * 2
    x, y, z = np.array([x,y,z]) / (np.sin(angle/2) + 1e-8)
    return x, y, z, angle


def interpolate_structure(A, B, mixture_x):
    """
    Takes crystal A and B as the input an interpolates them as
    (1-mixture_x)*A + mixture_x*B
    """
    crysA = deepcopy(A)
    crysB = deepcopy(B)
    
    lengthsA = crysA.unit_cell.parameters[:3]
    anglesA = crysA.unit_cell.parameters[-3:]
    lengthsB = crysB.unit_cell.parameters[:3]
    anglesB = crysB.unit_cell.parameters[-3:]
    new_unit_cell = UnitCell.from_lengths_and_angles(lengthsA * (1 - mixture_x) + lengthsB * mixture_x,
                                                 np.radians(anglesA * (1 - mixture_x) + anglesB * mixture_x))
    
    centroidA_frac = np.mean(crysA.asymmetric_unit.positions, axis = 0)
#    centroidA_cart = crysA.to_cartesian(centroidA_frac)
    centroidB_frac = np.mean(crysB.asymmetric_unit.positions, axis = 0)
#    centroidB_cart = crysB.to_cartesian(centroidB_frac)
#    centroidC_cart = centroidA_cart * (1 - mixture_x) + centroidB_cart * mixture_x
    
#    trans_vecA = centroidC_cart - centroidA_cart
    
    # Rotation axis and angle
    x, y, z, angle = rotation_finder(crysA, crysB)

    crysA_rotated = rotation(crysA, x, y, z, -angle * mixture_x)
    molA_rotated = crysA_rotated.asymmetric_unit
    molA_rotated_cart = crysA_rotated.to_cartesian(molA_rotated.positions)
#    translated_molA_rotated_cart = molA_rotated_cart + trans_vecA
    
    
    crysC = Crystal(new_unit_cell, 
                    A.space_group, 
                    A.asymmetric_unit)
    translated_molA_rotated_frac = crysC.to_fractional(molA_rotated_cart) - \
            np.mean(crysC.to_fractional(molA_rotated_cart), axis = 0) + \
            centroidA_frac * (1 - mixture_x) + centroidB_frac * mixture_x
    crysC.asymmetric_unit = AsymmetricUnit(molA_rotated.elements,
                          translated_molA_rotated_frac,
                          molA_rotated.labels)

    return crysC


class NearestAtoms:
    def __init__(self, atomic_numbers, atomic_positions):
        self.nums = atomic_numbers
        self.symbols = np.array([atoms_dict[i] for i in atomic_numbers])
        self.positions = atomic_positions
        self.tree = KDTree(self.positions)

    def neighbours(self, pt, r):
        idx = self.tree.query_ball_point(pt, r, n_jobs=number_of_cores)
        return self.symbols[idx], self.positions[idx]
    

def hkl_finder(c, radius):
    """
    Samples the surface of a sphere around the centroid, and returns maximum
    and minimum of h,k, and l in a way that all the sampled points are inside
    the cluster (supercell).

    """
    centroid_frac = np.mean(c.asymmetric_unit.positions, axis = 0)
    centroid_xyz = c.to_cartesian(centroid_frac)
    thetas = np.linspace(0, np.pi, 18, endpoint=True)
    phis = np.linspace(-np.pi, np.pi, 36, endpoint=False)
    surface_points = []
    for t in thetas:
        for ph in phis:
            surface_points.append([radius * sn(t) * cs (ph) + centroid_xyz[0],
                                  radius * sn(t) * sn(ph) + centroid_xyz[1],
                                  radius * cs(t) + centroid_xyz[2]])
    surface_points = np.array(surface_points)

    #Surface by lattice vectors = sblv
    sblv = c.to_fractional(surface_points)
    h_max, k_max, l_max = np.array([np.ceil(np.max(sblv[:,0])),
                                   np.ceil(np.max(sblv[:,1])),
                                   np.ceil(np.max(sblv[:,2]))])
    h_min, k_min, l_min = np.array([np.floor(np.min(sblv[:,0])),
                                   np.floor(np.min(sblv[:,1])),
                                   np.floor(np.min(sblv[:,2]))])
    return h_max, k_max, l_max, h_min, k_min, l_min


def atoms_in_radius(c, radius):
#     nums = c.asymmetric_unit.atomic_numbers
    mol_frac = c.asymmetric_unit.positions
    centroid = np.mean(mol_frac, axis=0)
    farthest_atom_xyz = np.max(np.linalg.norm(c.to_cartesian(mol_frac - centroid), axis=1))
#     mol_cart = c.to_cartesian(mol_frac)

    h_max, k_max, l_max, h_min, k_min, l_min = hkl_finder(c, radius + farthest_atom_xyz)

    cc = c.as_P1()
    nums_in_full_unit_cell = cc.asymmetric_unit.atomic_numbers
    full_unit_cell = cc.asymmetric_unit.positions
    atomic_positions = full_unit_cell
    for h in np.arange(h_min, h_max + 1):
        for k in np.arange(k_min, k_max + 1):
            for l in np.arange(l_min, l_max + 1):
                if h == k == l == 0:
                    continue
                else:
                    atomic_positions = np.concatenate((atomic_positions,
                                                       full_unit_cell + np.array([h,k,l])),
                                                       axis=0)
    atomic_positions = cc.to_cartesian(atomic_positions)

    ncells = int(((h_max-h_min)+1)*((k_max-k_min)+1)*((l_max-l_min)+1))
    atomic_numbers = np.tile(nums_in_full_unit_cell, ncells)
    return NearestAtoms(atomic_numbers, atomic_positions)


def close_checker(crys, radius = 3, expansion_factor = 1.5):
#     crys = CrystalStructure(res_file)

    nearest_atoms = atoms_in_radius(crys, radius)
    cm_frac = crys.asymmetric_unit.positions
    cm_xyz = crys.to_cartesian(cm_frac)
    cm_symbols = np.array(crys.asymmetric_unit.elements).astype('str')
    
    # Initializing the close_dict
    close_dict = {} #type1, type2, close?, required_expansion along 3 lattice vectors
    for s1 in np.unique(cm_symbols):
        for s2 in np.unique(cm_symbols):
            if s1 <= s2:
                close_dict[s1, s2] = [False, None]


    # Finding CLOSE atom pairs considering their type
    for idx1 in range(cm_symbols.shape[0]):
        cluster_symb, cluster_pos = nearest_atoms.neighbours(cm_xyz[idx1], radius)
        
        # Removing atoms in the parent molecule
        for atom in cm_xyz:
            idx_r = np.where(np.linalg.norm(cluster_pos - atom, axis = 1) < 0.0001)
            cluster_symb = np.delete(cluster_symb, [idx_r])
            cluster_pos = np.delete(cluster_pos, idx_r, 0)

        for idx2 in range(cluster_symb.shape[0]):
            connecting_vector = cm_xyz[idx1] - cluster_pos[idx2]
            dist = np.linalg.norm(connecting_vector)
            d_ref = expansion_factor * bonds_dictionary[cm_symbols[idx1], cluster_symb[idx2]]
            if abs(dist - d_ref) > 0.1 and dist < d_ref:
                direction = connecting_vector / np.linalg.norm(connecting_vector)
                required_expansion = (d_ref - dist) * direction
                if cm_symbols[idx1] <= cluster_symb[idx2]:
                    close_dict[cm_symbols[idx1], cluster_symb[idx2]] = \
                            [True, np.abs(crys.to_fractional(required_expansion))]
                else:
                    close_dict[cluster_symb[idx2], cm_symbols[idx1]] = \
                            [True, np.abs(crys.to_fractional(required_expansion))]
                    
    # Determining required expansion in along different lattice vectors
    P = np.zeros(3)
    for key in close_dict.keys():
        if close_dict[key][0]:
            if close_dict[key][1][0] > P[0]:
                P[0] = close_dict[key][1][0]
            if close_dict[key][1][1] > P[1]:
                P[1] = close_dict[key][1][1]
            if close_dict[key][1][2] > P[2]:
                P[2] = close_dict[key][1][2]
    for key in close_dict.keys():
        if close_dict[key][0]:
            return True, P
    return False, P


def direct_path_finder(crysA, crysB):
    x, y, z, angle = rotation_finder(crysA, crysB)
    no_lenths_steps = int(np.max(np.ceil(np.abs((crysA.unit_cell.parameters[:3] \
            - crysB.unit_cell.parameters[:3]) / 0.3))))
    no_angels_steps = int(angle / np.deg2rad(10)) + 1
    no_of_images = max(no_lenths_steps, no_angels_steps)
    direct_path_crysts = []
    for i, mixture_x in enumerate(np.linspace(0,1, no_of_images, endpoint=True)):
        direct_path_crysts.append(interpolate_structure(crysA, crysB, mixture_x))
    return direct_path_crysts


def check_and_expand(ID, crys, x, y, z, angle, expansion_factor = 1.5):
    crystal = deepcopy(crys)
    crystal = rotation(crystal, x, y, z, angle)
    is_close , P = close_checker(crystal, expansion_factor = expansion_factor)
    while is_close:
        crystal = volume_change(crystal, P)
        is_close, P = close_checker(crystal)
    return ID, crystal


def sampler(crysA, crysB, expansion_factor = 1.5, cwd='.'):
    x, y, z, angle = rotation_finder(crysA, crysB)
    direct_path_crysts = direct_path_finder(crysA, crysB)

    if not path.exists('%s/direct_path/' %cwd):
        mkdir('%s/direct_path/' %cwd)
    for i, c in enumerate(direct_path_crysts):
        c.save('%s/direct_path/%02i.res' %(cwd, i))
    
    rot_axes = meshzoo.icosa_sphere(1)[0] 
    rot_axes = rot_axes[np.where(rot_axes.dot(np.array([x,y,z])) > 0)]
    no_fo_angles = 10
    rot_angles = np.linspace(-np.pi *(1 - 0.5/no_fo_angles), 
                             np.pi*(1 + 0.5/no_fo_angles),
                             no_fo_angles, 
                             endpoint=True)
    
    print("Sampling points around 'direct' connecting images...")
    db_connect = sqlite3.connect('%s/per_structures.db' %cwd)
    db_cursor = db_connect.cursor()
    db_cursor.execute('CREATE TABLE IF NOT EXISTS per_struc (id, per_res)')
    stmt = 'insert into per_struc ({}) VALUES ({})'.format('id, per_res', '?, ?')
    
    futures = []
    for i, dpcrys in tqdm_enumerate(direct_path_crysts):
        for j, rot_axis in enumerate(rot_axes):
            for k, angle in enumerate(rot_angles):
                futures.append(pool.submit(check_and_expand, '%02i_%03i_%03i'
                %(i, j, k), dpcrys, x, y, z, angle, expansion_factor = expansion_factor))

    ntotal = len(direct_path_crysts) * rot_axes.shape[0] * rot_angles.shape[0]
    with tqdm(total=ntotal, desc="sampling", unit="points") as pbar:
        for x in as_completed(futures):
            ID_temp, crys_temp = x.result()
            db_cursor.execute(stmt, (ID_temp, crys_temp.to_shelx_string())) 
            pbar.update(1)
    db_connect.commit()
    db_connect.close()

resfileA, resfileB = sys.argv[1:3]

if __name__ == "__main__":
    crys_A = Crystal.load(str(resfileA))
    crys_B = Crystal.load(str(resfileB))
    sampler(crys_A, crys_B, expansion_factor = 1.5)
