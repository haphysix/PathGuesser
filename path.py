import sqlite3
from cspy import Crystal
import os

cwd = os.getcwd()
if not os.path.exists('%s/path_res_files/' %cwd):
        os.mkdir('%s/path_res_files/' %cwd)

db_connect = sqlite3.connect('per_structures.db')
db_cursor = db_connect.cursor()

with open('path.txt', 'r') as f:
    with open('path.res', 'a') as rf:
        for i, line in enumerate(f.readlines()):
            ID = line.split()[0]
            db_cursor.execute("SELECT per_res FROM per_struc WHERE id=?",
            (str(ID),))
            shelx_temp = db_cursor.fetchall()[0][0]
            rf.write(shelx_temp)
            rf.write('\nEND\n\n')
            c = Crystal.from_shelx_string(shelx_temp)
            c.save('%s/path_res_files/%02i.res' %(cwd, i))
            
rf.close()
f.close()
db_connect.close()
