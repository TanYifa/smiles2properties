import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import random
import pandas as pd
import pickle

def get_element_xy_bond(molblock):
    ''' molblock is the output str of Chem.MolToBlock(),
    returns the xy coordinate, bond connection, atomic number for
    each atom in the molecule''' 
    x = molblock.splitlines()
    line4th = x[3].split()
    natoms = int(line4th[0])
    nbonds = int(line4th[1])
    #print(natoms,nbonds)
    R,composition,bond,atomic_numbers = [],[],[],[]
    element=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl',\
    'Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr',\
    'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn']
    atomic = list(range(1,51)) # a list [1,2,...,50]

    periodic_table=dict(zip(element,atomic))
    #periodic_table = {'H':1,'C':6,'N':7,'O':8,'F':9,'P':15,'S':16,'Cl':17}
    
    for i in range(natoms):
        xyz_line = x[4+i].split()
        R.append(xyz_line[0])
        R.append(xyz_line[1])
        composition.append(xyz_line[3])
    r = np.asarray(R,dtype=float)
    r = np.reshape(r,[-1,2])

    for i in range(nbonds):
        bond_line = x[4+natoms+i].split()
        bond.append(bond_line[0])
        bond.append(bond_line[1])    

    bond = np.reshape(bond,[-1,2])
    bond = bond.astype(int)
    #print(bond)

    for i in composition:
        if i not in periodic_table.keys():
            raise ValueError("The input element must be between H (No.1) and Sn (No.50).")
        else:
            atomic_numbers.append(periodic_table[i])

    return r, bond, atomic_numbers



def interpolation_bond(xy_normal,bond,mapsize,resolution):
    '''Get the interpolated dots between two atoms which are connected by a bond'''
    angstrom_per_pixel = mapsize/resolution
    interpolated = []
    for i in range(bond.shape[0]):
        atom_a = xy_normal[bond[i,0]-1]
        atom_b = xy_normal[bond[i,1]-1]
        len_i = 0.5*mapsize*np.linalg.norm(atom_a-atom_b)
        segments = int(len_i/angstrom_per_pixel)+1
        #print(len_i,segments)
        # segments: how many points interpolated between atom_a and atom_b
        for j in range(1,segments):
            interpolated.append(atom_a[0]+(atom_b[0]-atom_a[0])*j/segments)
            interpolated.append(atom_a[1]+(atom_b[1]-atom_a[1])*j/segments)
    interpolated = np.asarray(interpolated,dtype=float)
    interpolated = np.reshape(interpolated,[-1,2])
    return interpolated


def pixelation(xy,bond,atomic_numbers,mapsize=20.0,resolution=80):
    '''Get a 2D molecular graph that the pixels of atoms are replaced by atomic numbers,
    and the pixels of bonds are replaced by interpolated dots of the same color
    mapsize is the height or width of the graph, angstrom, resolution tells how many
    pixels in each dimension'''
    xy_center = xy-np.sum(xy,axis=0)
    xy_normal = xy_center/(0.5*mapsize)
    xy_normal += [1.0,1.0]
    xy_normal = xy_normal*0.500    # now the center is (0.5,0.5)
    if resolution is not int:
        resolution = int(resolution) # how many pixels for height, weight                
    pixelated = np.zeros([resolution,resolution])  # the graph is now blank
    pixelated = pixelated.astype(np.int8)

    interpolated = interpolation_bond(xy_normal,bond,mapsize,resolution)

    pixel_xy = resolution*xy_normal
    pixel_xy = np.asarray(pixel_xy,dtype=int)
    #if pixel_xy[:,0].max() > resolution or pixel_xy[:,1].max() > resolution:
    #	raise ValueError("The molecule is too big, input a smaller one (main chain length <= 16 atoms).")
    pixel_bond = resolution*interpolated
    pixel_bond = np.asarray(pixel_bond,dtype=int)
    for k in range(np.shape(pixel_xy)[0]):
        pixelated[pixel_xy[k,0],pixel_xy[k,1]]= 2*atomic_numbers[k]
    for l in range(np.shape(pixel_bond)[0]):
        pixelated[pixel_bond[l,0],pixel_bond[l,1]] = 3

    pixelated = pixelated.astype(np.int8)
    pixelated = pixelated.reshape([resolution,resolution,1])
    return pixelated


def from_smiles_to_pixelated_2d_graph(smile):  
    '''Transfer a smile code to a numpy.ndarray with shape (resolution,resolution,1). 
    The ndarray is a pixelated molecular graph.'''
    mol = Chem.MolFromSmiles(smile)
    AllChem.Compute2DCoords(mol)
    z = Chem.MolToMolBlock(mol) #z is str
    r, bond, atomic_numbers = get_element_xy_bond(z)
    mol_pixelated = pixelation(xy=r,bond=bond,atomic_numbers=atomic_numbers)
    return mol_pixelated


def show_2d_graph(smile):
    '''Show the 2d grey-scale graph of a molecule given a smile code
    '''
    mol_pixelated = from_smiles_to_pixelated_2d_graph(smile)
    plt.imshow(mol_pixelated[:,:,0],cmap='Greys_r')
    plt.show()




def split_train_and_test_molecules(properties_array,
	                               total_number=1000,
	                               split_ratio=0.7,
	                               property='Cv'):
    '''
    '''
    import random

    if int(total_number) >= 133885:
        total_number = 133885
    else:
        total_number = int(total_number)

    train_number = int(split_ratio*total_number)
    index_mol =  list(range(total_number))
    #random.shuffle(index_mol)

    names = ['A','B','C','mu','alpha','homo','lumo','gap',\
                        'r2','zpve','U0','U','H','G','Cv']
    k = zip(names,list(range(0,len(names))))
    properties_dict = dict(k)

    if  property not in properties_dict.keys():
    	raise ValueError("Choose a property from A, B, C, mu, alpha, homo, lumo, gap,\
                         r2, zpve, U0, U, H, G, Cv")
    else:
        y       =  properties_array[:,properties_dict[property]][index_mol]
        smiles  =  properties_array[:,16][index_mol]

    x      =  []
    for i in range(total_number):
        mol_i_pixelated = from_smiles_to_pixelated_2d_graph(smiles[i])
        x.append(mol_i_pixelated)

    x_train = x[:train_number]
    x_test  = x[train_number:]

    x_train = np.stack(x_train,axis=0)
    x_test  = np.stack(x_test,axis=0)

    y_train = y[0:train_number]
    y_test  = y[train_number:]
    return x_train,y_train,x_test,y_test


'''
xlsx             = pd.ExcelFile('BasicProperties.xlsx')
basic_properties = pd.read_excel(xlsx,'basic_properties_molecules')
p                = np.asarray(basic_properties)
x_train,y_train,x_test,y_test = split_train_and_test_molecules(properties_array=p,total_number=1000,split_ratio=0.7)

'''
def store_dataset(dataset,filename):
	import pickle
	fw = open(filename,'w')
	pickle.dump(dataset,fw)
	fw.close()
def grab_dataset(filename):
	import pickle
	fr = open(filename)
	return pickle.load(fr)

if __name__ == '__main__':
    xlsx             = pd.ExcelFile('BasicProperties.xlsx')
    basic_properties = pd.read_excel(xlsx,'basic_properties_molecules')
    p                = np.asarray(basic_properties)
    x_train,y_train,x_test,y_test = split_train_and_test_molecules(properties_array = p,
    	                                                           total_number = 9600,
    	                                                           split_ratio = 0.7,
    	                                                           property = 'U0')
    print("Now we have x_train,y_train,x_test,y_test.")
    #store_dataset(x_train,'x_train.dat');
    #store_dataset(x_train,'y_train.dat');
    #store_dataset(x_train,'x_test.dat');
    #store_dataset(x_train,'y_test.dat');
    np.save("x_train",x_train)
    np.save("y_train",y_train)
    np.save("x_test",x_test)
    np.save("y_test",y_test)


'''
np.savez("train_and_test_data.npz",x_train,y_train,x_test,y_test)
r = np.load("train_and_test_data.npz") # type r is numpy.lib.npyio.NpzFile
x_train = r["arr_0"]
y_train = r["arr_1"]
'''
'''
# if you input np.savez("train_and_test_data.npz",x_train,y_train,x_test,y_test=y_test)
# you can retrieve y_test by calling r["y_test"]
'''