# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:03:35 2020

@author: chris
"""

# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
This example shows how to marginalize out discrete model variables in Pyro.
This combines Stochastic Variational Inference (SVI) with a
variable elimination algorithm, where we use enumeration to exactly
marginalize out some variables from the ELBO computation. We might
call the resulting algorithm collapsed SVI or collapsed SGVB (i.e
collapsed Stochastic Gradient Variational Bayes). In the case where
we exactly sum out all the latent variables (as is the case here),
this algorithm reduces to a form of gradient-based Maximum
Likelihood Estimation.
To marginalize out discrete variables ``x`` in Pyro's SVI:
1. Verify that the variable dependency structure in your model
    admits tractable inference, i.e. the dependency graph among
    enumerated variables should have narrow treewidth.
2. Annotate each target each such sample site in the model
    with ``infer={"enumerate": "parallel"}``
3. Ensure your model can handle broadcasting of the sample values
    of those variables
4. Use the ``TraceEnum_ELBO`` loss inside Pyro's ``SVI``.
Note that empirical results for the models defined here can be found in
reference [1]. This paper also includes a description of the "tensor
variable elimination" algorithm that Pyro uses under the hood to
marginalize out discrete latent variables.
References
1. "Tensor Variable Elimination for Plated Factor Graphs",
Fritz Obermeyer, Eli Bingham, Martin Jankowiak, Justin Chiu,
Neeraj Pradhan, Alexander Rush, Noah Goodman. https://arxiv.org/abs/1902.03210
"""
import argparse
import logging
import sys

import torch
import torch.nn as nn
from torch.distributions import constraints

import numpy as np

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO, TraceTMC_ELBO
from pyro.ops.indexing import Vindex
from pyro.optim import Adam
from pyro.util import ignore_jit_warnings


import dataloader_top500_sequences_D1 as dataloader_top500_sequences
import bvm_class_unimodal_FINAL_29_03_2020 as BVM
from Load_utils_final import process_dataset, extract_elements_of_test_set,Rama,plot_ELBOs
import matplotlib.pyplot as plt

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.DEBUG)

# Add another handler for logging debugging events (e.g. for profiling)
# in a separate stream that can be captured.
log = logging.getLogger()
debug_handler = logging.StreamHandler(sys.stdout)
debug_handler.setLevel(logging.DEBUG)
debug_handler.addFilter(filter=lambda record: record.levelno <= logging.DEBUG)
log.addHandler(debug_handler)


# Let's start with a simple Hidden Markov Model.
#
#     x[t-1] --> x[t] --> x[t+1]
#        |        |         |
#        V        V         V
#     y[t-1]     y[t]     y[t+1]
#
# This model includes a plate for the data_dim = 88 keys on the piano. This
# model has two "style" parameters probs_x and probs_y that we'll draw from a
# prior. The latent state is x, and the observed state is y. We'll drive
# probs_* with the guide, enumerate over x, and condition on y.
#
# Importantly, the dependency structure of the enumerated variables has
# narrow treewidth, therefore admitting efficient inference by message passing.
# Pyro's TraceEnum_ELBO will find an efficient message passing scheme if one
# exists.
def model_0(seq_AA, seq_DSSP, seq_dihedral, lengths, args, batch_size=None, include_prior=True):
    assert not torch._C._get_tracing_state()
    num_sequences, max_length, data_dim = seq_AA.shape
    
    data_dim_dihedral = 2
    
    
    with poutine.mask(mask=include_prior):
        # Our prior on transition probabilities will be:
        # stay in the same state with 90% probability; uniformly jump to another
        # state with 10% probability.
        probs_x = pyro.sample("probs_x",
                              dist.Dirichlet(0.9 * torch.eye(args.hidden_dim) + 0.1)
                                  .to_event(1))
        # We put a weak prior on the conditional probability of a tone sounding.
        # We know that on average about 4 of 88 tones are active, so we'll set a
        # rough weak prior of 10% of the notes being active at any one time.
        # NOTE: there isn't actually 88 tones... only 51 as we have removed the ones that don't occur very often when preparing the data further down...
        
        
        probs_y_means = pyro.sample("probs_y_means",
                              dist.Normal(torch.tensor([0.]), torch.tensor(1.))
                                  .expand([args.hidden_dim, data_dim_dihedral])
                                  .to_event(2))
        

        # gamma is always positive
        probs_y_var = pyro.sample("probs_y_var",
                              dist.Gamma(torch.tensor([2.]), torch.tensor(2.))
                                  .expand([args.hidden_dim, 2])
                                  .to_event(2))
        
        
    # In this first model we'll sequentially iterate over sequences in a
    # minibatch; this will make it easy to reason about tensor shapes.
    
    for i in pyro.plate("sequences", len(lengths), batch_size):
        length = int(lengths[i].item())
        
        
        sequence_dihedral = seq_dihedral[i,:length]
        state_x = 0
        for t in pyro.markov(range(length)):
            # On the next line, we'll overwrite the value of x with an updated
            # value. If we wanted to record all x values, we could instead
            # write x[t] = pyro.sample(...x[t-1]...).
            state_x = pyro.sample("state_x_{}_{}".format(i, t), dist.Categorical(Vindex(probs_x)[..., state_x, :]),
                            infer={"enumerate": "sequential"})
            
            pyro.sample("y_phi_{}_{}".format(i,t), dist.Normal(
                Vindex(probs_y_means)[..., state_x.squeeze(-1), 0], 
                Vindex(probs_y_var)[..., state_x.squeeze(-1), 0]),
                obs=sequence_dihedral[t][0])
            
            pyro.sample("y_psi_{}_{}".format(i,t), dist.Normal(
                Vindex(probs_y_means)[..., state_x.squeeze(-1), 1], 
                Vindex(probs_y_var)[..., state_x.squeeze(-1), 1]),
                obs=sequence_dihedral[t][1])
            
            
            
            
            """
            pyro.sample("y_phi_{}_{}".format(i,t), dist.VonMises(
                loc = Vindex(probs_y_means)[..., state_x.squeeze(-1), 0], 
                concentration = Vindex(probs_y_k1k2)[..., state_x.squeeze(-1), 0]),
                obs=sequence_dihedral[t][0])
            
            pyro.sample("y_psi_{}_{}".format(i,t), dist.VonMises(
                loc = Vindex(probs_y_means)[..., state_x.squeeze(-1), 1], 
                concentration = Vindex(probs_y_k1k2)[..., state_x.squeeze(-1), 1]),
                obs=sequence_dihedral[t][1])
            """
                
            
            
# To see how enumeration changes the shapes of these sample sites, we can use
# the Trace.format_shapes() to print shapes at each site:
# $ python examples/hmm.py -m 0 -n 1 -b 1 -t 5 --print-shapes
# ...
#  Sample Sites:
#   probs_x dist          | 16 16
#          value          | 16 16
#   probs_y dist          | 16 88
#          value          | 16 88
#     tones dist          |
#          value       88 |
# sequences dist          |
#          value        1 |
#   x_178_0 dist          |
#          value    16  1 |
#   y_178_0 dist    16 88 |
#          value       88 |
#   x_178_1 dist    16  1 |
#          value 16  1  1 |
#   y_178_1 dist 16  1 88 |
#          value       88 |
#   x_178_2 dist 16  1  1 |
#          value    16  1 |
#   y_178_2 dist    16 88 |
#          value       88 |
#   x_178_3 dist    16  1 |
#          value 16  1  1 |
#   y_178_3 dist 16  1 88 |
#          value       88 |
#   x_178_4 dist 16  1  1 |
#          value    16  1 |
#   y_178_4 dist    16 88 |
#          value       88 |
#
# Notice that enumeration (over 16 states) alternates between two dimensions:
# -2 and -3.  If we had not used pyro.markov above, each enumerated variable
# would need its own enumeration dimension.


# Next let's make our simple model faster in two ways: first we'll support
# vectorized minibatches of data, and second we'll support the PyTorch jit
# compiler.  To add batch support, we'll introduce a second plate "sequences"
# and randomly subsample data to size batch_size.  To add jit support we
# silence some warnings and try to avoid dynamic program structure.

# Note that this is the "HMM" model in reference [1] (with the difference that
# in [1] the probabilities probs_x and probs_y are not MAP-regularized with
# Dirichlet and Beta distributions for any of the models)
def model_1(seq_AA, seq_DSSP, seq_dihedral, lengths, args, batch_size=None, include_prior=True):
    # Sometimes it is safe to ignore jit warnings. Here we use the
    # pyro.util.ignore_jit_warnings context manager to silence warnings about
    # conversion to integer, since we know all three numbers will be the same
    # across all invocations to the model.
    with ignore_jit_warnings():
        num_sequences, max_length, data_dim = map(int, seq_AA.shape) # map just takes the tensor.shape and splits it...
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
        assert not torch._C._get_tracing_state()
    #num_sequences, max_length, data_dim = seq_AA.shape
    
    #data_dim_AA = 20 #for aminoacids
    data_dim_dihedral = 2
        #print('num_seq', num_sequences)
        #print('max_length', max_length)
        #print('data_dim', data_dim)
    with poutine.mask(mask=include_prior):
        # Our prior on transition probabilities will be:
        # stay in the same state with 90% probability; uniformly jump to another
        # state with 10% probability.
        probs_x = pyro.sample("probs_x",
                              dist.Dirichlet(0.9 * torch.eye(args.hidden_dim) + 0.1)
                                  .to_event(1))
        # We put a weak prior on the conditional probability of a tone sounding.
        # We know that on average about 4 of 88 tones are active, so we'll set a
        # rough weak prior of 10% of the notes being active at any one time.
        # NOTE: there isn't actually 88 tones... only 51 as we have removed the ones that don't occur very often when preparing the data further down...
        """probs_y_AA = pyro.sample("probs_y_AA",
                              dist.Beta(0.1, 0.9)
                                  .expand([args.hidden_dim, data_dim_AA])
                                  .to_event(2))"""
        
        probs_y_means = pyro.sample("probs_y_means",
                              dist.VonMises(torch.tensor([0.]), torch.tensor(90.))
                                  .expand([args.hidden_dim, data_dim_dihedral])
                                  .to_event(2))
        

        
        probs_y_k1k2 = pyro.sample("probs_y_k1k2",
                              dist.Gamma(torch.tensor([70.]), torch.tensor(1.))
                                  .expand([args.hidden_dim, 2])
                                  .to_event(2))
        
        
    tones_plate = pyro.plate("tones", data_dim, dim=-1)
    # We subsample batch_size items out of num_sequences items. Note that since
    # we're using dim=-1 for the notes plate, we need to batch over a different
    # dimension, here dim=-2.
    with pyro.plate("sequences", num_sequences, batch_size, dim=-1) as batch:
        #length = int(lengths[i].item())
        lengths = lengths[batch]
        
        #sequence_AA = seq_AA[i, :length]
        #sequence_dihedral = seq_dihedral[i,:length]
        state_x = 0
        # If we are not using the jit, then we can vary the program structure
        # each call by running for a dynamically determined number of time
        # steps, lengths.max(). However if we are using the jit, then we try to
        # keep a single program structure for all minibatches; the fixed
        # structure ends up being faster since each program structure would
        # need to trigger a new jit compile stage.
        #print('max length', max_length)
        #print('lengths.max()', lengths.max())
        for t in pyro.markov(range(max_length if args.jit else int(lengths.max()))):
            #print('t: ', t)
            #print('x: ', x)
            with poutine.mask(mask=(t < lengths).unsqueeze(-2)): # this expression sets the mask to either True or False
                
                # On the next line, we'll overwrite the value of x with an updated
                # value. If we wanted to record all x values, we could instead
                # write x[t] = pyro.sample(...x[t-1]...).
                
                #print('probs_x[x]: ',probs_x[x].shape)
                state_x = pyro.sample("state_x_{}".format(t), dist.Categorical(Vindex(probs_x)[..., state_x, :]),
                            infer={"enumerate": "sequential"})
            
            
                #a = pyro.sample("y_AAb_{}_{}".format(i, t), dist.Categorical(Vindex(probs_y_AA)[...,state_x.squeeze(-1), :]),)
                #print('sample: ', a)
                """
                pyro.sample("y_AA_{}".format(t), dist.Categorical(Vindex(probs_y_AA)[...,state_x.squeeze(-1), :]),
                            obs=sequence_AA[t])
                """
                
                #print('obs phi: ', (seq_dihedral[batch,t, 0].reshape((16,1))).shape)
                
                pyro.sample("y_phi_{}".format(t), dist.VonMises(
                    loc = Vindex(probs_y_means)[..., state_x.squeeze(-1), 0], 
                    concentration = Vindex(probs_y_k1k2)[..., state_x.squeeze(-1), 0]),
                    obs=seq_dihedral[batch,t,0].reshape((args.batch_size,1)))
                
                pyro.sample("y_psi_{}".format(t), dist.VonMises(
                    loc = Vindex(probs_y_means)[..., state_x.squeeze(-1), 1], 
                    concentration = Vindex(probs_y_k1k2)[..., state_x.squeeze(-1), 1]),
                    obs=seq_dihedral[batch,t,1].reshape((args.batch_size,1)))
# $ python examples/hmm.py -m 1 -n 1 -t 5 --batch-size=10 --print-shapes
# ...
#  Sample Sites:
#   probs_x dist             | 16 16
#          value             | 16 16
#   probs_y dist             | 16 88
#          value             | 16 88
#     tones dist             |
#          value          88 |
# sequences dist             |
#          value          10 |
#       x_0 dist       10  1 |
#          value    16  1  1 |
#       y_0 dist    16 10 88 |
#          value       10 88 |
#       x_1 dist    16 10  1 |
#          value 16  1  1  1 |
#       y_1 dist 16  1 10 88 |
#          value       10 88 |
#       x_2 dist 16  1 10  1 |
#          value    16  1  1 |
#       y_2 dist    16 10 88 |
#          value       10 88 |
#       x_3 dist    16 10  1 |
#          value 16  1  1  1 |
#       y_3 dist 16  1 10 88 |
#          value       10 88 |
#       x_4 dist 16  1 10  1 |
#          value    16  1  1 |
#       y_4 dist    16 10 88 |
#          value       10 88 |
#
# Notice that we're now using dim=-2 as a batch dimension (of size 10),
# and that the enumeration dimensions are now dims -3 and -4.


def model_7():
    """here because name is needed  to be present for code check later. Historic reasons"""
    pass


models = {name[len('model_'):]: model
          for name, model in globals().items()
          if name.startswith('model_')}


def main(args):
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    logging.info('Loading data')
    
    """ setting up data """

    dic = dataloader_top500_sequences.data_dictionary()
    
    train_dic = dic['train']
    train_AA = train_dic['sequences_AA']
    train_DSSP = train_dic['sequences_DSSP']
    train_Dihedral = train_dic['sequences_Dihedral']
    train_mask = train_dic['sequence_lengths']
    
    test_dic = dic['test']
    test_AA = test_dic['sequences_AA']
    test_DSSP = test_dic['sequences_DSSP']
    test_Dihedral = test_dic['sequences_Dihedral']
    test_mask = test_dic['sequence_lengths']
    
    validate_dic = dic['valid']
    validate_AA = validate_dic['sequences_AA']
    validate_DSSP = validate_dic['sequences_DSSP']
    validate_Dihedral = validate_dic['sequences_Dihedral']
    validate_mask = validate_dic['sequence_lengths']
    
    
    
    
    logging.info('-' * 40)
    model = models[args.model]
    logging.info('Training {} on {} sequences'.format(
        model.__name__, len(dic['train']['sequences_AA'])))


    num_observations = float(train_mask.sum())
    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()
    pyro.enable_validation(__debug__)

    # We'll train using MAP Baum-Welch, i.e. MAP estimation while marginalizing
    # out the hidden state x. This is accomplished via an automatic guide that
    # learns point estimates of all of our conditional probability tables,
    # named probs_*.
    guide = AutoDelta(poutine.block(model, expose_fn=lambda msg: msg["name"].startswith("probs_")))

    # To help debug our tensor shapes, let's print the shape of each site's
    # distribution, value, and log_prob tensor. Note this information is
    # automatically printed on most errors inside SVI.
    
    
    
    
    #removed this for now as I couldn't be bothered changing sequences and lengths to train_AA, train_DSP, and so on.....
    """
    if args.print_shapes:
        first_available_dim = -2 if model is model_0 else -3
        guide_trace = poutine.trace(guide).get_trace(
            sequences, lengths, args=args, batch_size=args.batch_size)
        model_trace = poutine.trace(
            poutine.replay(poutine.enum(model, first_available_dim), guide_trace)).get_trace(
            sequences, lengths, args=args, batch_size=args.batch_size)
        logging.info(model_trace.format_shapes())
        """
        
        
    # Enumeration requires a TraceEnum elbo and declaring the max_plate_nesting.
    # All of our models have two plates: "data" and "tones".
    optim = Adam({'lr': args.learning_rate})
    if args.tmc:
        if args.jit:
            raise NotImplementedError("jit support not yet added for TraceTMC_ELBO")
        #elbo = TraceTMC_ELBO(max_plate_nesting=1 if model is model_0 else 2)
        elbo = TraceTMC_ELBO(max_plate_nesting=1)
        tmc_model = poutine.infer_config(
            model,
            lambda msg: {"num_samples": args.tmc_num_samples, "expand": False} if msg["infer"].get("enumerate", None) == "parallel" else {})  # noqa: E501
        svi = SVI(tmc_model, guide, optim, elbo)
    else:
        Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO
        #elbo = Elbo(max_plate_nesting=1 if model is model_0 else 2,
        elbo = Elbo(max_plate_nesting=1,
                    strict_enumeration_warning=(model is not model_7),
                    jit_options={"time_compilation": args.time_compilation})
        svi = SVI(model, guide, optim, elbo)

    # We'll train on small minibatches.
    logging.info('Step\tLoss')
    for step in range(args.num_steps):
        loss = svi.step(train_AA, train_DSSP, train_Dihedral, train_mask, args=args, batch_size=args.batch_size)
        logging.info('{: >5d}\t{}'.format(step, loss / num_observations))

    if args.jit and args.time_compilation:
        logging.debug('time to compile: {} s.'.format(elbo._differentiable_loss.compile_time))

    # We evaluate on the entire training dataset,
    # excluding the prior term so our results are comparable across models.
    train_loss = elbo.loss(model, guide, train_AA, train_DSSP, train_Dihedral, train_mask, args, include_prior=False)
    logging.info('training loss = {}'.format(train_loss / num_observations))

    # Finally we evaluate on the test dataset.
    logging.info('-' * 40)
    logging.info('Evaluating on {} test sequences'.format(len(dic['test']['sequences_AA'])))
    
    
    """
    if args.truncate:
        lengths = lengths.clamp(max=args.truncate)"""
    num_observations = float(test_mask.sum())

    # note that since we removed unseen notes above (to make the problem a bit easier and for
    # numerical stability) this test loss may not be directly comparable to numbers
    # reported on this dataset elsewhere.
    test_loss = elbo.loss(model, guide, test_AA, test_DSSP, test_Dihedral, test_mask, args=args, include_prior=False)
    logging.info('test loss = {}'.format(test_loss / num_observations))

    # We expect models with higher capacity to perform better,
    # but eventually overfit to the training set.
    capacity = sum(value.reshape(-1).size(0)
                   for value in pyro.get_param_store().values())
    logging.info('{} capacity = {} parameters'.format(model.__name__, capacity))


if __name__ == '__main__':
    assert pyro.__version__.startswith('1.3.1')
    parser = argparse.ArgumentParser(description="MAP Baum-Welch learning Bach Chorales")
    parser.add_argument("-m", "--model", default="0", type=str,
                        help="one of: {}".format(", ".join(sorted(models.keys()))))
    parser.add_argument("-n", "--num-steps", default=50, type=int)
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("-d", "--hidden-dim", default=16, type=int)
    parser.add_argument("-nn", "--nn-dim", default=48, type=int)
    parser.add_argument("-nc", "--nn-channels", default=2, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("-t", "--truncate", type=int)
    parser.add_argument("-p", "--print-shapes", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('--time-compilation', action='store_true')
    parser.add_argument('-rp', '--raftery-parameterization', action='store_true')
    parser.add_argument('--tmc', action='store_true',
                        help="Use Tensor Monte Carlo instead of exact enumeration "
                             "to estimate the marginal likelihood. You probably don't want to do this, "
                             "except to see that TMC makes Monte Carlo gradient estimation feasible "
                             "even with very large numbers of non-reparametrized variables.")
    parser.add_argument('--tmc-num-samples', default=10, type=int)
    args = parser.parse_args()
    main(args)
    
    
    
    for key, value in pyro.get_param_store().items():    
        print(f"{key}:\n{value}\n")
        





def RamachandranP( data_angles,plot_title = 'forgot to add title', subtitle = ' ', colorbar = True):
    '''makes Ramachandran plot '''
    import numpy as np
    from matplotlib.colors import LogNorm
    #convert radians to degrees
    Degrees = np.rad2deg(data_angles) 
    # Get phi, psi angles from dataset
    phi = Degrees[:,0] 
    psi = Degrees[:,1] 
    plt.figure(figsize=(7, 6))
    plt.hist2d( phi, psi, bins = 200, norm = LogNorm(), cmap = plt.cm.jet )
    plt.suptitle(plot_title, fontsize = 14)
    plt.title(subtitle, fontsize = 10)

    plt.xlabel('φ')
    plt.ylabel('ψ')
    # set axes range
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    if colorbar == True:
        plt.colorbar()
    plt.show()
    #fig = plt.figure()
    #return(fig)    

def generate_random_sequence(length = 2000): # remember to change data_dim to 20 

    # In this first model we'll sequentially iterate over sequences in a
    # minibatch; this will make it easy to reason about tensor shapes.
    
    dihedral_torch = torch.empty((length, 2))
    
    #probs_x = pyro.param('AutoDelta.probs_x')
    
    #probs_y_AA = pyro.param('AutoDelta.probs_y_AA')
    
    probs_y_means = pyro.param('AutoDelta.probs_y_means')
    
    probs_y_k1k2 = pyro.param('AutoDelta.probs_y_var')
    
    #probs_y_lam = pyro.param('AutoDelta.probs_y_lam')
    probs_equal = torch.ones((args.hidden_dim))
    
    state_x = 0

    for t in range(length):
        # On the next line, we'll overwrite the value of x with an updated
        # value. If we wanted to record all x values, we could instead
        # write x[t] = pyro.sample(...x[t-1]...).
        state_x = pyro.sample("x_{}".format(t), dist.Categorical(probs_equal))
        
        phi = pyro.sample("y_phi_{}".format(t), dist.VonMises(
                loc = probs_y_means[ state_x, 0], 
                concentration = probs_y_k1k2[state_x, 0]))
            
        psi = pyro.sample("y_psi_{}".format(t), dist.VonMises(
                loc = probs_y_means[state_x, 1], 
                concentration = probs_y_k1k2[state_x, 1]))
        
        
        
        dihedral_torch[t,0] = phi.item()
        dihedral_torch[t,1] = psi.item()
        #print(dihedral)
        #print(dihedral_torch)
    return dihedral_torch.numpy()




def generate_sequence(length = 100): # remember to change data_dim to 20 

    # In this first model we'll sequentially iterate over sequences in a
    # minibatch; this will make it easy to reason about tensor shapes.
    
    dihedral_torch = torch.empty((length, 2))
    
    probs_x = pyro.param('AutoDelta.probs_x')
    
    #probs_y_AA = pyro.param('AutoDelta.probs_y_AA')
    
    probs_y_means = pyro.param('AutoDelta.probs_y_means')
    
    probs_y_k1k2 = pyro.param('AutoDelta.probs_y_var')
    
    #probs_y_lam = pyro.param('AutoDelta.probs_y_lam')

    
    state_x = 0

    for t in range(length):
        # On the next line, we'll overwrite the value of x with an updated
        # value. If we wanted to record all x values, we could instead
        # write x[t] = pyro.sample(...x[t-1]...).
        state_x = pyro.sample("x_{}_{}".format(t), dist.Categorical(probs_x[state_x]))
        
        phi = pyro.sample("y_phi_{}".format(t), dist.VonMises(
                loc = probs_y_means[ state_x, 0], 
                concentration = probs_y_k1k2[state_x, 0]))
            
        psi = pyro.sample("y_psi_{}".format(t), dist.VonMises(
                loc = probs_y_means[state_x, 1], 
                concentration = probs_y_k1k2[state_x, 1]))
        
        
        
        dihedral_torch[t,0] = phi.item()
        dihedral_torch[t,1] = psi.item()
        #print(dihedral)
        #print(dihedral_torch)
    return dihedral_torch.numpy()
        
dihedral_angles = generate_random_sequence()
RamachandranP( dihedral_angles, 'Ramachandran plot: (Data: Generated from trained TorusDBN)', colorbar = False )
"""
for key, value in pyro.get_param_store().items():    
        print(key)


for name in pyro.get_param_store().get_all_param_names():
    print(name)
    

"""

