# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 13:35:05 2020

@author: chris
"""

import torch
import pyro

import numpy


import numpy as np
from operator import itemgetter 
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader , Subset



""" The output of this file is a dictionary containing 3 dictionaries: training data, testing data, validation data.
The data is protein sequences that have been padded to the same length. The original intend was to set up this data to be used for time series modelling.
Data shape is as so: sequences, sequence, datapoint. So as an example with imaginary numbers take the 
data for dihedral angles: size [120, 800, 2], this indicates there are 120 sequences which are 800 long(time steps), and each timestep contains two datapoints (phi and psi). 

DSSP and AminoAcid is done as a categorical number, so even though the data only has 1 datapoint for each timestep, this 1 data point will in the case of DSSP
be a categorical of 0,1 or 2. and in the case of AminoAcid be a categorical of 0 to 19.

The length of each sequence is included as well because the sequences are padded, so the original length is needed for masking

"""


def load_500(filename):
    # Amino Acids labels list
    AA_list_outside = []
    # DSSP symbol labels list
    DSSP_list_outside = []
    # Angles list
    angles_list_outside = []
    
    AA_list = []
    # DSSP symbol labels list
    DSSP_list = []
    # Angles list
    angles_list = []
    
    for line in filename.readlines():
        
        if "#" in line: #emptying/clearing the lists for each new peptide
            # list --> numpy  
            angles_array = np.array(angles_list)       
            # reshape to numpy:[ phi, psi ]
            angles_array = np.reshape(angles_array,((int(len(angles_array)/2)), 2)) 
            
            AA_list_outside.append(AA_list)
            DSSP_list_outside.append(DSSP_list)
            angles_list_outside.append(angles_array)
            
            
            # Amino Acids labels list
            AA_list = []
            # DSSP symbol labels list
            DSSP_list = []
            # Angles list
            angles_list = []
        elif "NT" in line:
            continue
        elif "CT" in line:
            continue
        else:
            elements_of_line = line.split()
            
            AA_list.append(elements_of_line[0])                 # append AA   label
            DSSP_list.append(elements_of_line[1])               # append DSSP label
            angles_list.append(float(elements_of_line[2]))   # append phi
            angles_list.append(float(elements_of_line[3]))   # append psi
         

        
    return(angles_list_outside,AA_list_outside,DSSP_list_outside)

def AA_to_num(list_of_peptides):
    '''A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V'''
    # Check for imbalance in data labels                         
    # counts the elements' frequency
#    print(Counter(list_of_AA).keys())
#    print("AA freqs in dataset:", Counter(list_of_AA).values())
    #keys_AA = Counter(list_of_AA).keys()      
    #freqs_AA =  Counter(list_of_AA).values() 
#    print("Check correspondance of freq with spesific AA", list_of_AA.count('C') )    
    
    big_list = []
    
    for list_of_AA in list_of_peptides:
        for i in range(0, len(list_of_AA)):
            if list_of_AA[i].startswith('A'):
                list_of_AA[i] = int(0)
            elif list_of_AA[i].startswith('R'):
                list_of_AA[i] = int(1)
            elif list_of_AA[i].startswith('N'):
                list_of_AA[i] = int(2)
            elif list_of_AA[i].startswith('D'):
                list_of_AA[i] = int(3)
            elif list_of_AA[i].startswith('C'):
                list_of_AA[i] = int(4)
            elif list_of_AA[i].startswith('Q'):
                list_of_AA[i] = int(5)       
            elif list_of_AA[i].startswith('E'):
                list_of_AA[i] = int(6)
            elif list_of_AA[i].startswith('G'):
                list_of_AA[i] = int(7)
            elif list_of_AA[i].startswith('H'):
                list_of_AA[i] = int(8)       
            elif list_of_AA[i].startswith('I'):
                list_of_AA[i] = int(9)
            elif list_of_AA[i].startswith('L'):
                list_of_AA[i] = int(10)
            elif list_of_AA[i].startswith('K'):
                list_of_AA[i] = int(11)       
            elif list_of_AA[i].startswith('M'):
                list_of_AA[i] = int(12)
            elif list_of_AA[i].startswith('F'):
                list_of_AA[i] = int(13)
            elif list_of_AA[i].startswith('P'):
                list_of_AA[i] = int(14)
            elif list_of_AA[i].startswith('S'):
                list_of_AA[i] = int(15)
            elif list_of_AA[i].startswith('T'):
                list_of_AA[i] = int(16)
            elif list_of_AA[i].startswith('W'):
                list_of_AA[i] = int(17)
            elif list_of_AA[i].startswith('Y'):
                list_of_AA[i] = int(18)
            elif list_of_AA[i].startswith('V'):
                list_of_AA[i] = int(19)
        
        #new_list_of_AA = np.array(list_of_AA)
        #new_list_of_AA = new_list_of_AA.astype(int)
        #big_list.append(new_list_of_AA)
        
        big_list.append(list_of_AA)

    
    #y_pos=list(range(0,20))
    # Create bars
    #plt.bar(y_pos, freqs_AA, color = 'purple')
    # Create names on the x-axis
    #plt.xticks(y_pos, keys_AA)
    #plt.title("Frequencies of AAs in dataset")
    # Show graphic
    #fig=plt.figure()   
    return(big_list)


def sensor_or_not(outer_list_of_DSSP, Num_of_classes = 3):
    from collections import Counter
    '''Sensoring and plotting'''
    new_outer_list_of_DSSP = []
    if  Num_of_classes == 3:       
        '''Sensoring METHOD A'''
        for list_of_DSSP in outer_list_of_DSSP:
            # B to E
            list_of_DSSP = [dssp.replace('B', 'E') for dssp in list_of_DSSP]
            # G to E
            list_of_DSSP = [dssp.replace('G', 'H') for dssp in list_of_DSSP]
            
            # Rest to coil
            list_of_DSSP = [dssp.replace('S', 'C') for dssp in list_of_DSSP]
            list_of_DSSP = [dssp.replace('T', 'C') for dssp in list_of_DSSP]
            list_of_DSSP = [dssp.replace('-', 'C') for dssp in list_of_DSSP]
            list_of_DSSP = [dssp.replace('I', 'C') for dssp in list_of_DSSP]
            
            #keys_DSSP = Counter(list_of_DSSP).keys()      
            #freqs_DSSP =  Counter(list_of_DSSP).values() 
            
            # Turn to np.array
            list_of_DSSP = np.array(list_of_DSSP)
            list_of_DSSP[list_of_DSSP=='E'] = 0           # E=0
            list_of_DSSP[list_of_DSSP=='H'] = 1           # H=1
            list_of_DSSP[list_of_DSSP=='C'] = 2           # C=2
              
            list_of_DSSP = list_of_DSSP.astype(int)
            new_outer_list_of_DSSP.append(list_of_DSSP)
            ###############################################################################
            #y_pos=list(range(0,Num_of_classes))
            # Create bars
            #plt.bar(y_pos, freqs_DSSP , width = 0.5, color = 'b')
            # Create names on the x-axis
            #plt.xticks(y_pos,  keys_DSSP)
            #plt.title("Frequencies of DSSP symbols in dataset")
            #restart Kernel in Error: TypError:'str' object is not callable'
            # Show graphic
            #fig = plt.figure()
            ###############################################################################
        
    elif Num_of_classes == 8:
        for list_of_DSSP in outer_list_of_DSSP:
            '''Keep the 8 classes'''
            #keys = Counter(list_of_DSSP).keys()      
            #freqs =  Counter(list_of_DSSP).values()
            # Turn to np.array
            list_of_DSSP = np.array(list_of_DSSP)
            list_of_DSSP[list_of_DSSP=='B'] = 0
            list_of_DSSP[list_of_DSSP=='E'] = 1
            list_of_DSSP[list_of_DSSP=='G'] = 2
            list_of_DSSP[list_of_DSSP=='H'] = 3
            list_of_DSSP[list_of_DSSP=='S'] = 4
            list_of_DSSP[list_of_DSSP=='T'] = 5
            list_of_DSSP[list_of_DSSP=='-'] = 6
            list_of_DSSP[list_of_DSSP=='I'] = 7
            
            list_of_DSSP = list_of_DSSP.astype(int)
            new_outer_list_of_DSSP.append(list_of_DSSP)
            ###############################################################################
            #y_pos=list(range(0,Num_of_classes))
            # Create bars
            #plt.bar(y_pos, freqs, width = 0.5, color = 'b')
            # Create names on the x-axis
            #plt.xticks(y_pos,  keys)
            #plt.title("Frequencies of DSSP symbols in dataset")
            # Show graphic
            #fig = plt.figure()
            #print(" 'I' has only 20 occurencies in whole dataset")
            ###############################################################################
        

    else:
        print('Error: 3 or 8 classes of labels')
        
    return(new_outer_list_of_DSSP)







def padding_and_mask(AAs_numbers, DSSP_numbers, Angles):
    """ input is data as categorical (and angles) as list, where sequences are of different lengths.
    output is data in tensors that have been padded with zeros.""" 
    
    # set up data matrix
    length_of_sequences = torch.zeros(len(AAs_numbers)) # this is used for masking in the model
    
    
    
    for i in range(len(AAs_numbers)):
        length_of_sequences[i] = len(AAs_numbers[i])
        #print(len(AAs_numbers[i]))
    
    max_length = int(max(length_of_sequences))
    
    # AAs and DSSP only has size 1 in the rightmost dimension because we are using a categorical to indicate.
    AAs_tensor_outer = torch.zeros(len(AAs_numbers), max_length, 1) # shape = number of sequences, length of sequence (max length), data
    DSSP_tensor_outer = torch.zeros(len(AAs_numbers), max_length, 1) # shape = number of sequences, length of sequence (max length), data
    
    # Angles tensor has size 2 in the rightmost dimension because each data point concists of two angles: phi and psi
    Angles_tensor_outer = torch.zeros(len(AAs_numbers), max_length, 2) # shape = number of sequences, length of sequence (max length), data
    
    
    print('max length: ', max_length)
    for sequence_i in range(len(AAs_numbers)):
        
        AA_tensor_sequence = (torch.tensor(AAs_numbers[sequence_i])).reshape(len(AAs_numbers[sequence_i]), 1)
        AAs_tensor_outer[sequence_i, 0:len(AA_tensor_sequence)] = AA_tensor_sequence
        
        DSSP_tensor_sequence = (torch.tensor(DSSP_numbers[sequence_i])).reshape(len(AAs_numbers[sequence_i]), 1)
        DSSP_tensor_outer[sequence_i, 0:len(AA_tensor_sequence)] = DSSP_tensor_sequence
        
        Angles_tensor_sequence = (torch.tensor(Angles[sequence_i])).reshape(len(AAs_numbers[sequence_i]), 2)
        Angles_tensor_outer[sequence_i, 0:len(AA_tensor_sequence)] = Angles_tensor_sequence
        
        
        
        if len(AA_tensor_sequence) != len(DSSP_tensor_sequence):
            print(' PROBLEM!! '*40)
            print('sequence nr {} has different lengths of AA and DSSP sequences!!!'.format(sequence_i))
    #print(DSSP_tensor_outer)
    return AAs_tensor_outer[1:], DSSP_tensor_outer[1:], Angles_tensor_outer[1:], length_of_sequences[1:] ## not returning the first sequence as this is empty



def set_up_data():
    # don't know why, but first protein has len 0....
    angles, AAs, DSSP = load_500(open(r'top500.txt'))
    
    AAs_num = AA_to_num(AAs) #categorical numbers
    
    DSSP_num = sensor_or_not(DSSP) #categorical numbers
    
    
    
    AA_padded, DSSP_padded, angles_padded, sequence_lengths_mask = padding_and_mask(AAs_num, DSSP_num, angles) #padding with zeros so all have same length
    
    
    return AA_padded, DSSP_padded, angles_padded, sequence_lengths_mask

#AA_pad, DSSP_pad, angles_pad, mask = set_up_data()





def data_dictionary():
    """"No Omega data"""
    
    AA_pad, DSSP_pad, angles_pad, mask = set_up_data()
    
    test_seq_AA = AA_pad[0:28]
    test_seq_DSSP = DSSP_pad[0:28]
    test_seq_Dihedral = angles_pad[0:28]
    test_lengths = mask[0:28]
    
    valid_seq_AA = AA_pad[28:30]
    valid_seq_DSSP = DSSP_pad[28:30]
    valid_seq_Dihedral = angles_pad[28:30]
    valid_lengths = mask[28:30]
    
    
    
    train_seq_AA = AA_pad[30:]
    train_seq_DSSP = DSSP_pad[30:]
    train_seq_Dihedral = angles_pad[30:]
    train_lengths = mask[30:]
    
    
    outer_dic = {'test': {'sequence_lengths': test_lengths, 'sequences_AA': test_seq_AA, 'sequences_DSSP': test_seq_DSSP, 'sequences_Dihedral': test_seq_Dihedral},
                 'train': {'sequence_lengths': train_lengths, 'sequences_AA': train_seq_AA, 'sequences_DSSP': train_seq_DSSP, 'sequences_Dihedral': train_seq_Dihedral},
                 'valid': {'sequence_lengths': valid_lengths, 'sequences_AA': valid_seq_AA, 'sequences_DSSP': valid_seq_DSSP, 'sequences_Dihedral': valid_seq_Dihedral}}
    
    return outer_dic
    
"""
my_dic = data_dictionary()

my_seq = my_dic['test']['sequences']

print('first sequence in test data:')
print(my_seq[0])
"""



