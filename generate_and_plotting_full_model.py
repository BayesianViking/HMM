# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 17:35:58 2020

@author: chris
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pyro

import pyro.distributions as dist

from matplotlib.colors import LogNorm

"""
import bvm_class_unimodal_NoValErr as bvm
"""

def generate_sequence(mean1_param, mean2_param, probs_x_param, k1_param, k2_param, DSSP_param, AA_param, length = 20000):

    # In this first model we'll sequentially iterate over sequences in a
    # minibatch; this will make it easy to reason about tensor shapes.
    
    dihedral_torch = torch.empty((length, 2))
    AA_torch = torch.empty((length, 1))
    DSSP_torch = torch.empty((length, 1))
    
    
    probs_x = probs_x_param
    
    probs_y_means1 = mean1_param
    probs_y_means2 = mean2_param
    
    #probs_y_var1 = pyro.param('AutoDelta.probs_y_var1')
    #probs_y_var2 = pyro.param('AutoDelta.probs_y_var2')
    #probs_y_lam = pyro.param('AutoDelta.probs_y_lam')

    
    state_x = 0

    for t in range(length):
        #if t%150 == 0: #equivilant of each protein being of length 150
            #state_x = 0
        # On the next line, we'll overwrite the value of x with an updated
        # value. If we wanted to record all x values, we could instead
        # write x[t] = pyro.sample(...x[t-1]...).
        state_x = pyro.sample("x_{}".format(t), dist.Categorical(probs_x[state_x]))
        
        #print(state_x)
        phi = pyro.sample("y_phi_{}".format(t), dist.VonMises(
                probs_y_means1[ state_x], k1_param[state_x]))
            
        psi = pyro.sample("y_psi_{}".format(t), dist.VonMises(
                probs_y_means2[state_x], k2_param[state_x]))
        
        AA = pyro.sample("y_AA_{}".format(t), dist.Categorical(AA_param[state_x]))
        
        DSSP = pyro.sample("y_DSSP_{}".format(t), dist.Categorical(DSSP_param[state_x]))
        
        dihedral_torch[t,0] = phi.item()
        dihedral_torch[t,1] = psi.item()
        
        AA_torch[t,:] = AA.item()
        DSSP_torch[t,:] = DSSP.item()
        
        
        #print(dihedral)
        #print(dihedral_torch)
    return dihedral_torch.numpy(), AA_torch.numpy(), DSSP_torch.numpy()
"""
def generate_sequence_BIVARIATE(probs_mean1, probs_mean2, probs_x, probs_k1,probs_k2,probs_lam, length = 20000, batch_size = 100): # remember to change data_dim to 20 
    if (length%batch_size)!= 0:
        print('Length / batch_size has to result in an integer!')
        return(False)
    # In this first model we'll sequentially iterate over sequences in a
    # minibatch; this will make it easy to reason about tensor shapes.
    
    state_x = 0
    
    dihedral_torch = torch.zeros((length, 2))
    
    b = 0
    
    for t in range(int(length/batch_size)):
        #print(t)
        # On the next line, we'll overwrite the value of x with an updated
        # value. If we wanted to record all x values, we could instead
        # write x[t] = pyro.sample(...x[t-1]...).
        state_x = pyro.sample("x_{}".format(t), dist.Categorical(probs_x[state_x]))

        
        bvms = bvm.BivariateVonMises(torch.tensor([probs_mean1[state_x].item()]),
                                     torch.tensor([probs_mean2[state_x].item()]), 
                                     torch.tensor([probs_k1[state_x].item()]), 
                                     torch.tensor([probs_k2[state_x].item()]), 
                                     torch.tensor([probs_lam[state_x].item()]))
        
        phi_psi = bvms.sample(batch_size)
        
        #print(phi_psi)
        phi = phi_psi[:,0]
        psi = phi_psi[:,1]
        
        
        dihedral_torch[(b):(b+batch_size),0] = phi
        dihedral_torch[(b):(b+batch_size),1] = psi
        b += batch_size
    return dihedral_torch.numpy()


"""

    
def plot_rama( data_angles,plot_title = 'forgot to add title', subtitle = ' ', textstr = '',save_title = 'forgot_to_add_name.png', colorbar = True):
    '''makes Ramachandran plot '''
    #convert radians to degrees
    
    
    
    Degrees = np.rad2deg(data_angles) 
    
    #Degrees = data_angles
    
    # Get phi, psi angles from dataset
    phi = Degrees[:,0] 
    psi = Degrees[:,1] 
    plt.figure(figsize=(7, 6))
    plt.hist2d( phi, psi, bins = 200, norm = LogNorm(), cmap = plt.cm.plasma )
    plt.suptitle(plot_title, fontsize = 14, x = 0.45)
    plt.title(subtitle, fontsize = 10)
    
    plt.figtext(0.1, -0.01, textstr, fontsize=14)
    
    plt.xlabel('φ')
    plt.ylabel('ψ')
    # set axes range
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    if colorbar == True:
        plt.colorbar()
        
    plt.savefig(save_title, bbox_inches='tight')
    plt.close('all')


def plot_data_provide_clusters(mask, DSSP_train, AA_train, nr_clusters, hidden_dims, colorbar = True):
    
    
    
    import numpy as np
    from sklearn.cluster import KMeans
    import torch
    import pyro
    import matplotlib.pyplot as plt
    import dataloader_top500
    from matplotlib.colors import LogNorm
    import math
    
    all_angles, AA, DSSP = dataloader_top500.load_500(open(r'top500.txt'))
    
    kmeans = KMeans(n_clusters=nr_clusters, random_state=0).fit(all_angles)
    kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    
    plt.figure(figsize=(7, 6))
    plt.hist2d( all_angles[:,0], all_angles[:,1], bins = 200, norm = LogNorm(), cmap = plt.cm.plasma )
    plt.suptitle('Ramachandran plot', fontsize = 14, x = 0.45)
    plt.title('Showing data overlayed with cluster centers in radians', fontsize = 10)
    
    plt.figtext(0.1, -0.01, 'Number of clusters: {}'.format(nr_clusters), fontsize=14)
    
    plt.xlabel('φ')
    plt.ylabel('ψ')
    # set axes range
    #plt.xlim(-180, 180)
    #plt.ylim(-180, 180)
    if colorbar == True:
        plt.colorbar()
    plt.scatter(cluster_centers[:,0], cluster_centers[:,1], alpha=0.9, c='red', edgecolors='black', s=90, label='Cluster centers')

    plt.savefig('cluster_map.png', bbox_inches='tight')
    
    torch_cluster_centers = torch.repeat_interleave(torch.from_numpy(cluster_centers), int(hidden_dims/nr_clusters), 0)
    
    torch_cluster_centers = torch.cat((torch_cluster_centers, torch_cluster_centers))
    
    torch_cluster_centers = torch_cluster_centers[0:hidden_dims, :] 
    
    priors_mu_nu = torch_cluster_centers + ((torch.rand((hidden_dims, 2))*(2)-1)*0.3)
    
    
    
    plt.figure(figsize=(7, 6))
    plt.hist2d( all_angles[:,0], all_angles[:,1], bins = 200, norm = LogNorm(), cmap = plt.cm.plasma )
    plt.suptitle('Ramachandran plot', fontsize = 14, x = 0.45)
    plt.title('Showing location of priors overlayed the data in radians', fontsize = 10)
    plt.xlabel('φ')
    plt.ylabel('ψ')
    plt.figtext(0.1, -0.01, 'Number of hidden states and thus priors: {}'.format(hidden_dims), fontsize=14)
    
    if colorbar == True:
        plt.colorbar()
    plt.scatter(priors_mu_nu[:,0], priors_mu_nu[:,1], alpha=0.9, c='red', edgecolors='black', s=90, label='Cluster centers')
    plt.savefig('priors_map.png', bbox_inches='tight')
    
    
    AA_train = AA_train.int()
    DSSP_train = DSSP_train.int()
    # set up the data for hist plots of DSSP and AA
    AA = []
    DSSP = []
    for sequence in range(len(DSSP_train)):
        
        AA_1 = ((AA_train[sequence, :mask[sequence]]).flatten()).tolist()
        AA += AA_1
        
        DSSP_1 = ((DSSP_train[sequence, :mask[sequence]]).flatten()).tolist()
        DSSP += DSSP_1
        
    
    
    
    DSSP = (np.array(DSSP).flatten())
    AA = (np.array(AA).flatten())
    
    x_DSSP, DSSP_counts = np.unique(DSSP, return_counts=True)
    x_AA, AA_counts = np.unique(AA, return_counts=True)
    
    #print(x_DSSP)
    #print(DSSP_counts)
    
    plt.figure(figsize=(7, 6))
    bar1 = plt.bar(x_DSSP, DSSP_counts/sum(DSSP_counts), color = 'gray', edgecolor = 'black')
    plt.suptitle('Histogram', fontsize = 14, x = 0.45)
    plt.title('Secondary structure labels: training data', fontsize = 10)
    plt.xticks((0, 1, 2), ('0: E', '1: H', '2: C'), rotation = 60)
    plt.ylabel('Density', rotation = 'vertical')
    for rect in bar1:
        height = round(rect.get_height(), 2)
      
        plt.text(rect.get_x() + rect.get_width()/2.0, height+0.0005, height, ha='center', va='bottom')

    #plt.show()
    plt.savefig('DSSP_train.png', bbox_inches='tight')
    
    
    
    AA_names = ('0: A', '1: R','2: N', '3: D','4: C','5: Q','6: E','7: G','8: H','9: I','10: L','11: K','12: M','13: F','14: P','15: S','16: T','17: W','18: Y','19: V')
    

    plt.figure()
    plt.figure(figsize=(7, 6))
    bar2 = plt.bar(x_AA, AA_counts/sum(AA_counts), color = 'gray', edgecolor = 'black')
    plt.suptitle('Histogram', fontsize = 14, x = 0.45)
    plt.title('Amino acid labels: training data', fontsize = 10)
    plt.xticks(range(0,20,1), AA_names, rotation = 60)
    plt.ylabel('Density', rotation = 'vertical')
    for rect in bar2:
        height = round(rect.get_height(), 3)
        
        plt.text(rect.get_x() + rect.get_width()/2.0, height+0.0005, height, ha='center', va='bottom')
    
    plt.savefig('AA_train.png', bbox_inches='tight')
    
    plt.close('all')
    return priors_mu_nu.float()





def plot_ELBO(total_steps, test_loss, train_loss, test_frequency, hidden_dims, batch_size, savetitle_end = 'last'):
    
    
    plt.figure(figsize=(7, 6))
    
    plt.plot(range(0,total_steps, test_frequency), test_loss, label = "Testing loss")
    
    plt.plot(range(0,total_steps, test_frequency), train_loss, label = "Training loss")
    
    plt.xlabel('Step')
    plt.ylabel('Avg. loss')
    
    #plt.suptitle('ELBO', fontsize = 14, x = 0.45)
    plt.suptitle('ELBO plot', fontsize = 14)
    plt.title('Data: Avg. loss from evaluation of testing and training data', fontsize = 10)
    
    plt.figtext(0.1, -0.01, 'Number of hidden states: {}, batch size: {} '.format(hidden_dims, batch_size), fontsize=14)
    
    plt.xlim(0, total_steps, 1)
    plt.legend()
    
    plt.savefig('ELBO_test_loss_{}'.format(savetitle_end), bbox_inches='tight')
    
    
    
    
    
    
    plt.close('all')





def plot_compare(mask,AA_train, DSSP_train, AA_sequence, DSSP_sequence, Dihedral_sequence, nr_clusters, hidden_dims, batch_size, current_step, total_step, savetitle_end = '',colorbar = True):
    """ xxx_train is the training data.... 
        xxx_sequence is the generated data"""
    
    
    
    import numpy as np
    from sklearn.cluster import KMeans
    import torch
    import matplotlib.pyplot as plt
    import dataloader_top500
    from matplotlib.colors import LogNorm
    import math
    
    
    
    kmeans = KMeans(n_clusters=nr_clusters, random_state=0).fit(Dihedral_sequence)
    kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    
    plt.figure(figsize=(7, 6))
    plt.hist2d( Dihedral_sequence[:,0], Dihedral_sequence[:,1], bins = 200, norm = LogNorm(), cmap = plt.cm.plasma )
    plt.suptitle('Ramachandran plot', fontsize = 14, x = 0.45)
    plt.title('Showing generated angles overlayed with cluster centers in radians', fontsize = 10)
    
    plt.figtext(0.1, -0.01, 'Number of clusters: {}, Number of hidden states: {}\nbatch size: {}, steps: {} / {} '.format(nr_clusters, hidden_dims, batch_size, current_step, total_step), fontsize=14)
    
    plt.xlabel('φ')
    plt.ylabel('ψ')
    # set axes range
    #plt.xlim(-180, 180)
    #plt.ylim(-180, 180)
    if colorbar == True:
        plt.colorbar()
    plt.scatter(cluster_centers[:,0], cluster_centers[:,1], alpha=0.9, c='red', edgecolors='black', s=90, label='Cluster centers')

    plt.savefig('cluster_map_{}.png'.format(savetitle_end), bbox_inches='tight')
    
    
    AA_train = AA_train.int()
    DSSP_train = DSSP_train.int()
    # set up the data for hist plots of DSSP and AA
    AA = [] #train data
    DSSP = [] #train data
    for sequence in range(len(DSSP_train)): # training data
        
        AA_1 = ((AA_train[sequence, :mask[sequence]]).flatten()).tolist()
        AA += AA_1
        
        DSSP_1 = ((DSSP_train[sequence, :mask[sequence]]).flatten()).tolist()
        DSSP += DSSP_1
        
    
    
    
    DSSP = (np.array(DSSP).flatten())#train
    AA = (np.array(AA).flatten()) #train
    
    x_DSSP, DSSP_counts = np.unique(DSSP, return_counts=True) #train
    x_AA, AA_counts = np.unique(AA, return_counts=True) #train data
    
    DSSP_generated = DSSP_sequence.flatten()
    AA_generated = AA_sequence.flatten()
    generated_x, DSSP_generated = np.unique(DSSP_generated, return_counts=True)
    AA_generated_x, AA_generated = np.unique(AA_generated, return_counts=True)
    #print(x_DSSP)
    #print(DSSP_counts)
    
    plt.figure(figsize=(7, 6))
    width1 = 0.45
    bar1a = plt.bar(x_DSSP-width1/2, DSSP_counts/sum(DSSP_counts), color = 'gray', edgecolor = 'black', width = width1)
    bar1b = plt.bar(generated_x+width1/2, DSSP_generated/sum(DSSP_generated), color = 'blue', edgecolor = 'black', width = width1)
    
    
    plt.legend((bar1a, bar1b), ('Training data', 'Generated data'))
    
    plt.suptitle('Histogram', fontsize = 14, x = 0.45)
    plt.title('Secondary structure labels', fontsize = 10)
    
    plt.figtext(0.1, -0.01, 'Number of hidden states: {}, batch size: {}, steps: {} / {} '.format(hidden_dims, batch_size, current_step, total_step), fontsize=14)
    
    plt.ylabel('Density', rotation = 'vertical')
    plt.ylim(0,0.7)
    
    plt.xticks((0, 1, 2), ('0: E', '1: H', '2: C'), rotation = 60)
    
    for rect in bar1a:
        height = round(rect.get_height(), 2)
      
        plt.text(rect.get_x() + rect.get_width()/2.0, height+0.0005, height, ha='center', va='bottom')
    
    for rect in bar1b:
        height = round(rect.get_height(), 2)
      
        plt.text(rect.get_x() + rect.get_width()/2.0, height+0.0005, height, ha='center', va='bottom')

    #plt.show()
    plt.savefig('DSSP_comparison_{}.png'.format(savetitle_end), bbox_inches='tight')
    
    
    
    AA_names = ('0: A', '1: R','2: N', '3: D','4: C','5: Q','6: E','7: G','8: H','9: I','10: L','11: K','12: M','13: F','14: P','15: S','16: T','17: W','18: Y','19: V')
    

    plt.figure()
    plt.figure(figsize=(7, 6))
    width2 = 0.4
    bar2a = plt.bar(x_AA-width2/2., AA_counts/sum(AA_counts), color = 'gray', edgecolor = 'black', width=width2)
    bar2b = plt.bar(AA_generated_x +width2/2, AA_generated/sum(AA_generated), color = 'blue', edgecolor = 'black', width = width2)
    
    plt.suptitle('Histogram', fontsize = 14, x = 0.45)
    plt.title('Amino acid labels', fontsize = 10)
    
    plt.figtext(0.1, -0.01, 'Number of hidden states: {}, batch size: {}, steps: {} / {} '.format(hidden_dims, batch_size, current_step, total_step), fontsize=14)
    
    plt.ylabel('Density', rotation = 'vertical')
    plt.ylim(0,0.26)
    
    plt.xticks(range(0,20,1), AA_names, rotation = 60)
    for rect in bar2a:
        height = round(rect.get_height(), 3)
        plt.text(rect.get_x() + rect.get_width()/2.0+0.09, height+0.0005, height, ha='center', va='bottom', rotation = 'vertical', fontsize = 8)
    
    for rect in bar2b:
        height = round(rect.get_height(), 3)
        plt.text(rect.get_x() + rect.get_width()/2.0+0.11, height+0.0005, height, ha='center', va='bottom', rotation = 'vertical', fontsize = 8)
    
    plt.legend((bar2a, bar2b), ('Training data', 'Generated data'))
    plt.savefig('AA_comparison_{}.png'.format(savetitle_end), bbox_inches='tight')
    
    plt.close('all')
    




def plot_compare_histograms(mask,AA_train, DSSP_train, AA_sequence, DSSP_sequence, Dihedral_sequence, nr_clusters, hidden_dims, batch_size, current_step, total_step, savetitle_end = '',colorbar = True):
    """ xxx_train is the training data.... 
        xxx_sequence is the generated data"""
    
    
    
    import numpy as np
    from sklearn.cluster import KMeans
    import torch
    import matplotlib.pyplot as plt
    import dataloader_top500
    from matplotlib.colors import LogNorm
    import math
    
    AA_train = AA_train.int()
    DSSP_train = DSSP_train.int()
    # set up the data for hist plots of DSSP and AA
    AA = [] #train data
    DSSP = [] #train data
    for sequence in range(len(DSSP_train)): # training data
        
        AA_1 = ((AA_train[sequence, :mask[sequence]]).flatten()).tolist()
        AA += AA_1
        
        DSSP_1 = ((DSSP_train[sequence, :mask[sequence]]).flatten()).tolist()
        DSSP += DSSP_1
        
    
    
    
    DSSP = (np.array(DSSP).flatten())#train
    AA = (np.array(AA).flatten()) #train
    
    x_DSSP, DSSP_counts = np.unique(DSSP, return_counts=True) #train
    x_AA, AA_counts = np.unique(AA, return_counts=True) #train data
    
    DSSP_generated = DSSP_sequence.flatten()
    AA_generated = AA_sequence.flatten()
    generated_x, DSSP_generated = np.unique(DSSP_generated, return_counts=True)
    AA_generated_x, AA_generated = np.unique(AA_generated, return_counts=True)
    #print(x_DSSP)
    #print(DSSP_counts)
    
    plt.figure(figsize=(7, 6))
    width1 = 0.45
    bar1a = plt.bar(x_DSSP-width1/2, DSSP_counts/sum(DSSP_counts), color = 'gray', edgecolor = 'black', width = width1)
    bar1b = plt.bar(generated_x+width1/2, DSSP_generated/sum(DSSP_generated), color = 'blue', edgecolor = 'black', width = width1)
    
    
    plt.legend((bar1a, bar1b), ('Training data', 'Generated data'))
    
    plt.suptitle('Histogram', fontsize = 14, x = 0.45)
    plt.title('Secondary structure labels', fontsize = 10)
    
    plt.figtext(0.1, -0.01, 'Number of hidden states: {}, batch size: {}, steps: {} / {} '.format(hidden_dims, batch_size, current_step, total_step), fontsize=14)
    
    plt.ylabel('Density', rotation = 'vertical')
    plt.ylim(0,0.7)
    
    plt.xticks((0, 1, 2), ('0: E', '1: H', '2: C'), rotation = 60)
    
    for rect in bar1a:
        height = round(rect.get_height(), 2)
      
        plt.text(rect.get_x() + rect.get_width()/2.0, height+0.0005, height, ha='center', va='bottom')
    
    for rect in bar1b:
        height = round(rect.get_height(), 2)
      
        plt.text(rect.get_x() + rect.get_width()/2.0, height+0.0005, height, ha='center', va='bottom')

    #plt.show()
    plt.savefig('DSSP_comparison_{}.png'.format(savetitle_end), bbox_inches='tight')
    
    
    
    AA_names = ('0: A', '1: R','2: N', '3: D','4: C','5: Q','6: E','7: G','8: H','9: I','10: L','11: K','12: M','13: F','14: P','15: S','16: T','17: W','18: Y','19: V')
    

    plt.figure()
    plt.figure(figsize=(7, 6))
    width2 = 0.4
    bar2a = plt.bar(x_AA-width2/2., AA_counts/sum(AA_counts), color = 'gray', edgecolor = 'black', width=width2)
    bar2b = plt.bar(AA_generated_x +width2/2, AA_generated/sum(AA_generated), color = 'blue', edgecolor = 'black', width = width2)
    
    plt.suptitle('Histogram', fontsize = 14, x = 0.45)
    plt.title('Amino acid labels', fontsize = 10)
    
    plt.figtext(0.1, -0.01, 'Number of hidden states: {}, batch size: {}, steps: {} / {} '.format(hidden_dims, batch_size, current_step, total_step), fontsize=14)
    
    plt.ylabel('Density', rotation = 'vertical')
    plt.ylim(0.,0.26)
    plt.xticks(range(0,20,1), AA_names, rotation = 60)
    for rect in bar2a:
        height = round(rect.get_height(), 3)
        plt.text(rect.get_x() + rect.get_width()/2.0+0.09, height+0.0005, height, ha='center', va='bottom', rotation = 'vertical', fontsize = 8)
    
    for rect in bar2b:
        height = round(rect.get_height(), 3)
        plt.text(rect.get_x() + rect.get_width()/2.0+0.11, height+0.0005, height, ha='center', va='bottom', rotation = 'vertical', fontsize = 8)
    
    plt.legend((bar2a, bar2b), ('Training data', 'Generated data'))
    plt.savefig('AA_comparison_{}.png'.format(savetitle_end), bbox_inches='tight')
    
    plt.close('all')
    