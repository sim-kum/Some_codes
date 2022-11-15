from heapq import nsmallest

from ase import atoms, Atom
from ase.build import molecule, add_adsorbate
from ase.data.pubchem import pubchem_atoms_search, pubchem_atoms_conformer_search
import numpy as np
from pygcga.checkatoms import CheckAtoms
from pygcga.utilities import NoReasonableStructureFound
from ase.io import *
import random
import os
import math
tmp_cwd = '/Users/simrankumari/Google Drive/Shared drives/ZrO2_Cu111/Zr2O3_HCOO_4096/3_HCOO/3_layer'
os.chdir(tmp_cwd)
folder_list = [j for j in next(os.walk('.'))[1]]
#folder_list = ['Zr3O2_OH3']#, 'Zr3O4_OH4', 'Zr3O4_OH3', 'Zr3O5_OH2', 'Zr3O3_OH5', 'Zr3O3_OH4', 'Zr3O2_OH4', 'Zr3O4_OH2', 'Zr3O1_OH5', 'Zr3O5_OH1', 'Zr3O3_OH1', 'Zr3O3_OH3','Zr3O1_OH2','Zr3O3_OH2','Zr3O3_OH6', 'Zr3O4_OH1', 'Zr3O4_OH6', 'Zr3O5_OH4','Zr3O2_OH2','Zr3O2_OH3', 'Zr3O0_OH6', 'Zr3O1_OH6', 'Zr3O5_OH3', 'Zr3O1_OH4', 'Zr3O4_OH5', 'Zr3O1_OH3','Zr3O6_OH1', 'Zr3O0_OH5', 'Zr3O2_OH1', 'Zr3O6_OH2', 'Zr3O1_OH7', 'Zr3O2_OH6', 'Zr3O0_OH4']
print(folder_list)
def find_normal_vector(coord1,coord2):
    distance = [coord1[0] - coord2[0], coord1[1] - coord2[1], coord1[2] - coord2[2]]
    norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2 + distance[2] ** 2)
    direction = [-distance[0] / norm, -distance[1] / norm, -distance[2]/norm]
    return direction
#HCOO = pubchem_atoms_search(smiles='C(=O)[O-]')
HCOO = read('/Users/simrankumari/PycharmProjects/ML_with_python_book/chapter-2/CONTCAR_HCOO')
def O_atom_index(surface, dr_Zr = 0.5):
    t = surface.copy()
    rx = t.get_positions()

    HCOO = read('/Users/simrankumari/PycharmProjects/ML_with_python_book/chapter-2/CONTCAR_HCOO')
    H_indx = [atom.index for atom in t if atom.symbol == 'H']
    # if O and H distance is less than 1 , remove those coordinates from the list and assign an indx, select an index randomly and the
    # remove that indx
    # get O coordinates which is above a certain height.
    O_indx = [atom.index for atom in t if atom.symbol == 'O']
    OH_indx, ZrH_indx, H_new = [] , [] , []
    for j in O_indx:
        O_new = []
        for i in H_indx:
            if t.get_distance(j, i) < 1.1:
                O_new.append(j)
                O_new.append(i)
            else:
                continue
        if len(O_new) == 0:
            continue
        else: OH_indx.append(O_new)
    print(OH_indx)
    del_list = random.choice(OH_indx)
    print(del_list)
    normal = find_normal_vector(rx[del_list[0]], rx[del_list[1]])
    HCOO.rotate([0, 1, 0], normal, center=[0,0,0])
    HCOO.translate([rx[del_list[0]][0]+normal[0]*1.5,rx[del_list[0]][1]+normal[1]*1.5, rx[del_list[0]][2]])
    del t[[del_list]]
    t.extend(HCOO)
    return t

def get_OH_list(surface):
    H_indx = [atom.index for atom in surface if atom.symbol == 'H']
    O_indx = [atom.index for atom in surface if atom.symbol == 'O']
    OH_indx, ZrH_indx, H_new = [], [], []
    for j in O_indx:
        O_new = []
        for i in H_indx:
            print('OH_dis:', surface.get_distance(j, i))
            if surface.get_distance(j, i, mic=True) < 1.1:
                O_new.append(j)
                O_new.append(i)
            else:
                continue
        if len(O_new) == 0 or len(O_new) == 3:
            continue
        else:
            OH_indx.append(O_new)
    return OH_indx

def get_COOH_list(surface):
    H_indx = [atom.index for atom in surface if atom.symbol == 'H']
    O_indx = [atom.index for atom in surface if atom.symbol == 'O']
    C_indx = [atom.index for atom in surface if atom.symbol == 'C']
    COH_indx, ZrH_indx, H_new = [], [], []
    for j in C_indx:
        COOH_indx = []
        for i in H_indx:
            print('COOH_dis:',surface.get_distance(j, i))
            if surface.get_distance(j, i , mic=True) < 1.3:
                COOH_indx.append(j)
                COOH_indx.append(i)
            else:
                continue
        for k in O_indx:
            print('COOH_dis:',surface.get_distance(j, k))
            surface.get_distance(j, k)
            if surface.get_distance(j, k, mic=True) < 1.4 :
                COOH_indx.append(k)
        if len(COOH_indx) == 0:
            continue
        else:
            COH_indx.append(COOH_indx)

    return COH_indx

def replace_OH_With_COOH(del_list, t):
    rx = t.get_positions()
    normal = find_normal_vector(rx[del_list[0]], rx[del_list[1]])
    HCOO = read('/Users/simrankumari/PycharmProjects/ML_with_python_book/chapter-2/CONTCAR_HCOO')
    HCOO.rotate([0, 1, 0], normal, center=[0, 0, 0])
    HCOO.translate([rx[del_list[0]][0] + normal[0] * 1.5, rx[del_list[0]][1] + normal[1] * 1.5, rx[del_list[0]][2]])
    return HCOO

def replace_COOH_With_OH(COOH_list, t):
    rx = t.get_positions()
    normal = find_normal_vector(rx[COOH_list[0]], rx[COOH_list[1]])
    OH = read('/Users/simrankumari/PycharmProjects/ML_with_python_book/chapter-2/CONTCAR_OH')
    OH.rotate([0, 1, 0], normal, center=[0, 0, 0])
    OH.translate([rx[COOH_list[0]][0] + normal[0] * -0.5, rx[COOH_list[0]][1] + normal[1] * -0.5, rx[COOH_list[0]][2]])
    OH.write('CONTCAR_OH')
    return OH

def get_key(val, my_dict):
    for key, value in my_dict.items():
        # print(val,value)
        if val == value:
            # print(val)
            return key
        else:
            continue
def lowest_energy_structure(folder):
    #print(folder)
    ########### If reading from unique direc###################
    #os.chdir(folder + '/unique_dir/')
    #folder_list_new = [j for j in os.listdir()]
    #print(folder_list_new)
    os.chdir(folder)
    current_dir = os.getcwd()
    folder_list_new = [j for j in next(os.walk('.'))[1]]
    #print(folder_list_new)
    total_En = {}
    #print(folder)
    #try: os.chdir(folder + '/unique_dir/')
    #except FileNotFoundError: print('run the analyze command in ', folder)
    #outcar_list = [j for j in os.listdir()]
    #print(len(outcar_list))

    for i in folder_list_new:
        os.chdir(current_dir + '/' + i)
        #print(os.getcwd())
        try:
            slab = read('OUTCAR')
            total_En[i] = slab.get_potential_energy()
            #print(total_En)
        except FileNotFoundError :
            continue
    dir_we_need = []
    for i in nsmallest(3, total_En.values()):
        dir_we_need.append(current_dir + '/' + get_key(i , total_En)+'/OUTCAR')
    ##return a list of 10 best energy structure files from each directory
    return dir_we_need
for p in folder_list:
    os.chdir(tmp_cwd)
    list_of_contcar = lowest_energy_structure(p)
    for k in list_of_contcar:
        print(k)
        starting = read(k)
        os.chdir('/Users/simrankumari/Google Drive/Shared drives/ZrO2_Cu111/Zr2O3_HCOO_4096/3_HCOO/3_layer_to_submit_6/')
        try :
            t = starting.copy()
            OH_indx = get_OH_list(t)
            COOH_new_indx = get_COOH_list(t)
            print(OH_indx,COOH_new_indx)
            for del_list in OH_indx:
                os.chdir('/Users/simrankumari/Google Drive/Shared drives/ZrO2_Cu111/Zr2O3_HCOO_4096/3_HCOO/3_layer_to_submit_6/')
                COOH = replace_OH_With_COOH(del_list, t)
                for COOH_list in COOH_new_indx:
                    final = t.copy()
                    OH = replace_COOH_With_OH(COOH_list, t)
                    for i in del_list:
                        COOH_list.append(i)
                    del final[[COOH_list]]
                    final.extend(OH)
                    final.extend(COOH)
                    os.chdir('/Users/simrankumari/Google Drive/Shared drives/ZrO2_Cu111/Zr2O3_HCOO_4096/3_HCOO/3_layer_to_submit_6')
                    os.mkdir(str(p) + '_' + str(list_of_contcar.index(k)) + 'new_b_' + str(del_list[0])+'_'+str(COOH_list[0]))
                    os.chdir(str(p)+'_'+str(list_of_contcar.index(k))+'new_b_'+str(del_list[0])+'_'+str(COOH_list[0]))
                    final.write('input.traj')
        except FileExistsError: continue
