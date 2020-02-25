#!/usr/bin/env python

from cspy.crystal import Crystal, AsymmetricUnit, UnitCell, SpaceGroup, symmetry_operation
from copy import deepcopy
import numpy as np
import sys

def permute_lattice_vectors(crys, i, j, k):
    """
    Permutes lattice vectors of a crystal and generates a "rotated version".
    """
    # Transformation Matrix
    tm = np.zeros((3,3))
    tm[0,i] = 1
    tm[1,j] = 1
    tm[2,k] = 1
    
    lengths = crys.unit_cell.parameters[:3]
    angles = crys.unit_cell.parameters[-3:]
    mol = crys.asymmetric_unit
    sg = SpaceGroup(international_tables_number=crys.space_group.international_tables_number)
    
    # Update Symmetry Operations
    sym_ops = []
    for sym in crys.space_group.symmetry_operations:
        sym_ops.append(symmetry_operation.SymmetryOperation(
                                        np.dot(tm.dot(sym.rotation), tm.T), 
                                        tm.dot(sym.translation)))
    
    new_unit_cell = UnitCell.from_lengths_and_angles(tm.dot(lengths),
                                                 np.radians(tm.dot(angles)))
    
    new_asym_unit = AsymmetricUnit(mol.elements,
                          tm.dot(mol.positions.T).T,
                          mol.labels)
    new_sg = sg.from_symmetry_operations(sym_ops)
    crys_new = Crystal(new_unit_cell, new_sg, new_asym_unit)
    return crys_new


def equivalent_crystals(crys):
    """
    Generates all the crystals that are equivalent to the input.
    """
    lt = crys.space_group.lattice_type
    equivalent_lattice_vectors = [[0,1,2]]
    if lt == 'monoclinic':
        equivalent_lattice_vectors = [[0,1,2], [2,1,0]]
    if lt == 'orthorhombic' or lt == 'triclinic':
        equivalent_lattice_vectors = [[0,1,2], [0,2,1],
                                      [2,1,0], [1,0,2],
                                      [2,0,1], [1,2,0]]

    crys_by_other_mols = []
    for i, mol in enumerate(crys.unit_cell_molecules()):
        asym_temp = AsymmetricUnit(mol.elements,
                                   crys.to_fractional(mol.positions),
                                   mol.labels)
        crys_by_other_mols.append(Crystal(deepcopy(crys.unit_cell), 
                                          deepcopy(crys.space_group), 
                                          asym_temp))
    eq_crysts = []
    for elv in equivalent_lattice_vectors:
        i, j, k = elv
        for cbom in crys_by_other_mols:
            eq_crysts.append(permute_lattice_vectors(cbom, i, j, k))
    return eq_crysts

resfile = sys.argv[1]

if __name__ == "__main__":
    crys = Crystal.load(str(resfile))
    eq_crysts = equivalent_crystals(crys)
    for i, c in enumerate(eq_crysts):
        c.save('%s_eq_%03i.res' %(res_file[:,-4], i))



