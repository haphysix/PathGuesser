#!/usr/bin/env python

import sys, os
from shutil import copyfile
import equivalent_lattices as el
import sampler as s
from cspy.crystal import Crystal
import gc
from pathlib import Path
from shmolecule.crystal import Crystal as proCrystal
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
import numpy as np
import min_span_tree
import sqlite3


LOG = logging.getLogger("test_describe")

def describe(args, l_max=3):
    name, crystal = args
    name = Path(name).stem
    return name, crystal.asymmetric_unit.atomic_numbers, crystal.atomic_shape_descriptors(l_max=l_max, radius = 6)

def descriptor(db_file, lmax, outfile):
    crystals = []
    db_connect = sqlite3.connect(db_file)
    c = db_connect.cursor()
    c.execute('SELECT id, per_res FROM per_struc')
    for row in tqdm(c.fetchall(), desc="Loading crystals"):
        crystals.append((str(row[0]),
        proCrystal.from_shelx_string(row[1])))
    db_connect.close()

    natoms = sum(len(x[1].asymmetric_unit) for x in crystals)
    print("Total atoms in all crystals: ", natoms)
    l_max = lmax
    with ProcessPoolExecutor(4) as e:
            descriptors = {}
            futures = [
                e.submit(describe, crystal, l_max=l_max)
                for crystal in crystals
            ]
            with tqdm(total=natoms,
                      desc=f"l_max={l_max}", unit="atom") as pbar:
                for f in as_completed(futures):
                    name, nums, desc = f.result()
                    descriptors[name] = desc
#                    descriptors[name + "desc"] = desc
                    descriptors[name + "element"] = nums
                    pbar.update(len(nums))

    print("Saving to ", outfile)
    np.savez_compressed(outfile, **descriptors)


resfileA, resfileB = sys.argv[1:3]

def main():
    crys_A = Crystal.load(str(resfileA))
    crys_B = Crystal.load(str(resfileB))
    if crys_A.space_group.international_tables_number != \
            crys_B.space_group.international_tables_number:
        raise Exception("""
        Crystal A and crystal B are not in the same space group.
        For now, only crystal pairs in the same space group are allowed.
        """)
    cwd = os.getcwd()
    
    eq_crysts_of_A = el.equivalent_crystals(crys_A)
    for i, crys in enumerate(eq_crysts_of_A):
        if not os.path.exists('%s/path_%02i/' %(cwd, i)):
            os.mkdir('%s/path_%02i/' %(cwd, i))
            copyfile('path.py', '%s/path_%02i/path.py' %(cwd, i))
            
        s.sampler(crys, crys_B, cwd='%s/path_%02i/' %(cwd, i))
        db_connect = sqlite3.connect('%s/path_%02i/per_structures.db' %(cwd, i))
        db_cursor = db_connect.cursor()
        db_cursor.execute('CREATE TABLE IF NOT EXISTS per_struc (id, per_res)')
        stmt = 'insert into per_struc ({}) VALUES ({})'.format('id, per_res', '?, ?')
        db_cursor.execute(stmt, ('A', crys.to_shelx_string()))
        db_cursor.execute(stmt, ('B', crys_B.to_shelx_string()))
        db_connect.commit()
        db_connect.close()
        gc.collect()
       
        descriptor('%s/path_%02i/per_structures.db' %(cwd, i), 
                   lmax = 7, 
                   outfile = '%s/path_%02i/descriptors.npz' %(cwd, i))
        gc.collect()
        
        min_span_tree.connecting_path('%s/path_%02i/descriptors.npz' %(cwd, i), 
                        outdir = '%s/path_%02i/' %(cwd, i))
        gc.collect()
        break

if __name__ == "__main__":
    main()
