#!/usr/bin/env python

import argparse
import cPickle
import traceback
import logging
import time
import sys

import os
import numpy
import codecs

from dialog_encdec import DialogEncoderDecoder
from numpy_compat import argpartition
from state import prototype_state
logger = logging.getLogger(__name__)

def sample_wrapper(sample_logic):
    def sample_apply(*args, **kwargs):
        sampler = args[0]
        contexts = args[1]

        verbose = kwargs.get('verbose', False)

        if verbose:
            logger.info("Starting {} : {} start sequences in total".format(sampler.name, len(contexts)))
         
        context_samples = []
        context_costs = []
        context_weights = []
        context_context = []

        # Start loop for each utterance
        for context_id, context_utterances in enumerate(contexts):
            if verbose:
                logger.info("Searching for {}".format(context_utterances))

            # Convert contextes into list of ids
            joined_context = []
            if len(context_utterances) == 0:
                joined_context = [sampler.model.eos_sym]
            else:
                utterance_ids = sampler.model.words_to_indices(context_utterances.split())
                # Add eos tokens
                if len(utterance_ids) > 0:
                    if not utterance_ids[0] == sampler.model.eos_sym:
                        utterance_ids = [sampler.model.eos_sym] + utterance_ids
                    if not utterance_ids[-1] == sampler.model.eos_sym:
                        utterance_ids += [sampler.model.eos_sym]
                
                else:
                    utterance_ids = [sampler.model.eos_sym]

                joined_context += utterance_ids

            samples, costs, att_weight, context2 = sample_logic(sampler, joined_context, **kwargs) 
            #print att_weight
            # Convert back indices to list of words
            converted_samples = map(lambda sample : sampler.model.indices_to_words(sample, exclude_end_sym=kwargs.get('n_turns', 1) == 1), samples)
            # Join the list of words
            converted_samples = map(' '.join, converted_samples)

            if verbose:
                for i in range(len(converted_samples)):
                    print "{}: {}".format(costs[i], converted_samples[i].encode('utf-8'))

            context_samples.append(converted_samples)
            context_costs.append(costs)
            context_weights.append(att_weight)
            context_context.append(context2)
        return context_samples, context_costs,context_weights,context_context
    return sample_apply

class Sampler(object):
    """
    An abstract sampler class 
    """
    def __init__(self, model):
        # Compile beam search
        self.name = 'Sampler'
        self.model = model
        self.compiled = False
        self.max_len = 160

    def compile(self):
        self.next_probs_predictor_return_att_weight = self.model.build_next_probs_function_return_att_weight()
        self.compute_encoding = self.model.build_encoder_function()

        if not self.model.reset_utterance_decoder_at_end_of_utterance:
            self.compute_decoder_encoding = self.model.build_decoder_encoding()

        self.compiled = True
    
    def select_next_words(self, next_probs, step_num, how_many):
        pass

    def count_n_turns(self, utterance):
        return len([w for w in utterance \
                    if w == self.model.eos_sym])

    @sample_wrapper
    def sample(self, *args, **kwargs):
        context = args[0]
        
        print context
        n_samples = kwargs.get('n_samples', 1)
        ignore_unk = kwargs.get('ignore_unk', True)
        min_length = kwargs.get('min_length', 1)
        max_length = kwargs.get('max_length', 100)
        beam_diversity = kwargs.get('beam_diversity', 1)
        normalize_by_length = kwargs.get('normalize_by_length', True)
        verbose = kwargs.get('verbose', False)
        n_turns = kwargs.get('n_turns', 1)

        if not self.compiled:
            self.compile()
        
        
        #yawa add following##############
        #xd_raw1 = context
        context_orig = numpy.array(context, dtype='int32')
        #################
        
        # Convert to matrix, each column is a context 
        # [[1,1,1],[4,4,4],[2,2,2]]
        context = numpy.repeat(numpy.array(context, dtype='int32')[:,None], 
                               n_samples, axis=1)
        #print "context:" + str(context.shape)
        #print context
        if context[-1, 0] != self.model.eos_sym:
            raise Exception('Last token of context, when present,'
                            'should be the end of utterance: %d' % self.model.eos_sym)

        # Generate the reversed context
        reversed_context = self.model.reverse_utterances(context)

        if self.model.direct_connection_between_encoders_and_decoder:
            if self.model.bidirectional_utterance_encoder:
                dialog_enc_size = self.model.sdim+self.model.qdim_encoder*2
            else:
                dialog_enc_size = self.model.sdim+self.model.qdim_encoder
        else:
            dialog_enc_size = self.model.sdim

        prev_hs = numpy.zeros((n_samples, dialog_enc_size), dtype='float32')

        prev_hd = numpy.zeros((n_samples, self.model.utterance_decoder.complete_hidden_state_size), dtype='float32')

        if not self.model.reset_utterance_decoder_at_end_of_utterance:
            assert self.model.bs >= context.shape[1]
            enlarged_context = numpy.zeros((context.shape[0], self.model.bs), dtype='int32')
            enlarged_context[:, 0:context.shape[1]] = context[:]
            enlarged_reversed_context = numpy.zeros((context.shape[0], self.model.bs), dtype='int32')
            enlarged_reversed_context[:, 0:context.shape[1]] = reversed_context[:]

            ran_vector = self.model.rng.normal(size=(context.shape[0],n_samples,self.model.latent_gaussian_per_utterance_dim)).astype('float32')
            zero_mask = numpy.zeros((context.shape[0], self.model.bs), dtype='float32')
            ones_mask = numpy.zeros((context.shape[0], self.model.bs), dtype='float32')

            # Computes new utterance decoder hidden states (including intermediate utterance encoder and dialogue encoder hidden states)
            new_hd = self.compute_decoder_encoding(enlarged_context, enlarged_reversed_context, self.max_len, zero_mask, zero_mask, numpy.zeros((self.model.bs), dtype='float32'), ran_vector, ones_mask)
            prev_hd[:] = new_hd[0][-1][0:context.shape[1], :]


        fin_gen = []
        fin_costs = []
        ### yawa add fin_weights
        fin_att_weights =[]
        fin_context =[]
        gen = [[] for i in range(n_samples)]
        costs = [0. for i in range(n_samples)]
        
        ### yawa add att_weights
        att_weights =[[] for i in range(n_samples)]
        
        
        beam_empty = False

        h_raw = self.compute_encoding(context,reversed_context,context.shape[0])#
        h_raw1 = numpy.asarray([item[0] for item in h_raw[0]])
        hs_raw1 = numpy.asarray([item[0] for item in h_raw[1]])
        #numpy.asarray([item[0] for item in context])
        print "context:" + str(context.shape)
        
        # Compute random vector as additional input
        ran_vectors = self.model.rng.normal(size=(n_samples,self.model.latent_gaussian_per_utterance_dim)).astype('float32')

        for k in range(max_length):
            if len(fin_gen) >= n_samples or beam_empty:
                break
             
            if verbose:
                logger.info("{} : sampling step {}, beams alive {}".format(self.name, k, len(gen)))
            '''
            ####yawa add viable length search
            #h_raw = self.compute_encoding(context,reversed_context,context.shape[0])#
            #h_raw1 = numpy.asarray([item[0] for item in h_raw[0]])
            #hs_raw1 = numpy.asarray([item[0] for item in h_raw[1]])
            ####meng add######################################
            #h_raw2 = numpy.asarray([h_raw1 for i in range(len(gen))])
            #h_raw3 = numpy.transpose(h_raw2,(1,0,2))
            ####yawa add##################################################
            #hs_raw2 = numpy.asarray([hs_raw1 for i in range(len(gen))])
            #hs_raw3 = numpy.transpose(hs_raw2,(1,0,2))
            ##
            #xd_raw1 = context
            #xd_raw3 = numpy.repeat(numpy.array(xd_raw1, dtype='int32')[:,None], 
                               len(gen), axis=1)
            #######################################################
            '''
            #if len(gen)!=5:
            #continue
            # Here we aggregate the context and recompute the hidden state
            # at both session level and query level.
            # Stack only when we sampled something
            if k > 0:
                context = numpy.vstack([context, \
                                        numpy.array(map(lambda g: g[-1], gen))]).astype('int32')
                reversed_context = numpy.copy(context)
                for idx in range(context.shape[1]):
                    eos_indices = numpy.where(context[:, idx] == self.model.eos_sym)[0]
                    prev_eos_index = -1
                    for eos_index in eos_indices:
                        reversed_context[(prev_eos_index+2):eos_index, idx] = (reversed_context[(prev_eos_index+2):eos_index, idx])[::-1]
                        prev_eos_index = eos_index

            prev_words = context[-1, :]
            #h_raw1 = numpy.asarray([item[0] for item in h_raw[0]])
            #hs_raw1 = numpy.asarray([item[0] for item in h_raw[1]])
            #xd_raw1 = numpy.asarray([item[0] for item in context])
            ####meng add######################################
            h_raw2 = numpy.asarray([h_raw1 for i in range(len(gen))])
            h_raw3 = numpy.transpose(h_raw2,(1,0,2))
            ####yawa add##################################################
            hs_raw2 = numpy.asarray([hs_raw1 for i in range(len(gen))])
            hs_raw3 = numpy.transpose(hs_raw2,(1,0,2))
            
            ##
            #xd_raw2 = numpy.asarray([xd_raw1 for i in range(len(gen))])
            #xd_raw3 = numpy.transpose(xd_raw2,(1,0))
            
            #h_raw3 = h_raw[0]
            
            #hs_raw3 = h_raw[1]
            #xd_raw3 = context
            xd_raw2 = numpy.asarray([context_orig for i in range(len(gen))])
            xd_raw3 = numpy.transpose(xd_raw2,(1,0))
            
            # Recompute encoder states, hs and random variables 
            # only for those particular utterances that meet the end-of-utterance token
            indx_update_hs = [num for num, prev_word in enumerate(prev_words)
                                if prev_word == self.model.eos_sym]
            if len(indx_update_hs):
                encoder_states = self.compute_encoding(context[:, indx_update_hs], reversed_context[:, indx_update_hs], self.max_len)
                prev_hs[indx_update_hs] = encoder_states[1][-1]
                ran_vectors[indx_update_hs,:] = self.model.rng.normal(size=(len(indx_update_hs),self.model.latent_gaussian_per_utterance_dim)).astype('float32')


            # ... done
            #next_probs, new_hd = self.next_probs_predictor(prev_hs, prev_hd, prev_words, context, ran_vectors, h_raw[0])
            next_probs, new_hd, newout_att_weights= self.next_probs_predictor_return_att_weight(prev_hs, prev_hd, prev_words, context, ran_vectors, h_raw3, hs_raw3, xd_raw3)
            #print newout_att_weights.shape
            assert next_probs.shape[1] == self.model.idim
            
            # Adjust log probs according to search restrictions
            if ignore_unk:
                next_probs[:, self.model.unk_sym] = 0
            if k <= min_length:
                next_probs[:, self.model.eos_sym] = 0
                next_probs[:, self.model.eod_sym] = 0
             
            # Update costs 
            next_costs = numpy.array(costs)[:, None] - numpy.log(next_probs)

            # Select next words here
            (beam_indx, word_indx), costs = self.select_next_words(next_costs, next_probs, k, n_samples)
            
            # Update the stacks
            new_gen = [] 
            new_costs = []
            new_sources = []
            ##yawa add new_att_weights =[]
            new_att_weights =[]

            for num, (beam_ind, word_ind, cost) in enumerate(zip(beam_indx, word_indx, costs)):
                if len(new_gen) > n_samples:
                    break

                hypothesis = gen[beam_ind] + [word_ind]
                print "hypothesis:"+str(numpy.array(hypothesis).shape)
                #####yawa add following##############
                print "beam_id:" +str(beam_ind)
                hypothesis_att_weight = list(att_weights[beam_ind])
                print "att_weights[beam_ind]:"+str(numpy.array(att_weights[beam_ind]).shape)
                #print numpy.copy(newout_att_weights[beam_ind])
                #print "newout_att_weights[beam_ind]:"+str(newout_att_weights[beam_ind].shape)
                hypothesis_att_weight.append(newout_att_weights[beam_ind].tolist())
                #for xx in newout_att_weights[beam_ind].tolist():
                    #print xx
                print "hypothesis_att_weight2:"+str(numpy.array(hypothesis_att_weight).shape)
                
                 
                # End of utterance has been detected
                n_turns_hypothesis = self.count_n_turns(hypothesis)
                if n_turns_hypothesis == n_turns:
                    print "n_turns_hypothesis == n_turns"
                    if verbose:
                        logger.debug("adding utterance {} from beam {}".format(hypothesis, beam_ind))

                    # We finished sampling
                    fin_gen.append(hypothesis)
                    fin_costs.append(cost)
                    ##yawa
                    fin_att_weights.append(hypothesis_att_weight)
                    print "fin_att_weights:"+str(numpy.array(fin_att_weights).shape)
                    fin_context.append(context[:,beam_ind])
                elif self.model.eod_sym in hypothesis: # End of dialogue detected
                    print "self.model.eod_sym in hypothesis"
                    new_hypothesis = []
                    new_hypothesis_att_weights =[]
                    for wrd in hypothesis:
                        new_hypothesis += [wrd]
                        if wrd == self.model.eod_sym:
                            break
                    hypothesis = new_hypothesis
                    
                    
                    ###yawa
                    for (wrd, hypothesis_att_weight1)in zip(hypothesis, hypothesis_att_weight):
                        new_hypothesis_att_weights.append(hypothesis_att_weight1)
                        if wrd == self.model.eod_sym:
                            break
                    hypothesis_att_weight = new_hypothesis_att_weights
                    #####
                    
                    if verbose:
                        logger.debug("adding utterance {} from beam {}".format(hypothesis, beam_ind))
                        

                    # We finished sampling
                    fin_gen.append(hypothesis)
                    fin_costs.append(cost)
                    #yawa
                    fin_att_weights.append(hypothesis_att_weight)
                    print "fin_att_weights:"+str(numpy.array(fin_att_weights).shape)
                else:
                    print "else"
                    # Hypothesis recombination
                    # TODO: pick the one with lowest cost 
                    has_similar = False
                    if self.hyp_rec > 0:
                        has_similar = len([g for g in new_gen if \
                            g[-self.hyp_rec:] == hypothesis[-self.hyp_rec:]]) != 0
                    
                    if not has_similar:
                        new_sources.append(beam_ind)
                        new_gen.append(hypothesis)
                        new_costs.append(cost)
                        #print "hypothesis_att_weight3:"+str(numpy.array(hypothesis_att_weight).shape)
                        new_att_weights.append(hypothesis_att_weight)
                        #print "new_att_weights:"+str(numpy.array(new_att_weights).shape)
            
            if verbose:
                for gen in new_gen:
                    logger.debug("partial -> {}".format(' '.join(self.model.indices_to_words(gen))))

            prev_hd = new_hd[new_sources]
            prev_hs = prev_hs[new_sources]
            ran_vectors = ran_vectors[new_sources,:]
            context = context[:, new_sources]
            reversed_context = reversed_context[:, new_sources]
            gen = new_gen
            costs = new_costs
            att_weights=list(new_att_weights)
            print "att_weights:"+str(numpy.array(att_weights).shape)
            beam_empty = len(gen) == 0

        # If we have not sampled anything
        # then force include stuff
        if len(fin_gen) == 0:
            fin_gen = gen 
            fin_costs = costs 
         
        # Normalize costs
        if normalize_by_length:
            fin_costs = [(fin_costs[num]/len(fin_gen[num])) \
                         for num in range(len(fin_gen))]

        fin_gen = numpy.array(fin_gen)[numpy.argsort(fin_costs)]
        fin_att_weights = numpy.array(fin_att_weights)[numpy.argsort(fin_costs)]
        fin_context = numpy.array(fin_context)[numpy.argsort(fin_costs)]
        print numpy.array(fin_costs)
        print numpy.argsort(fin_costs)
        fin_costs = numpy.array(sorted(fin_costs))
        
        
        for item in fin_att_weights:
            print numpy.array(item).shape
        return fin_gen[:n_samples], fin_costs[:n_samples], fin_att_weights[:n_samples], fin_context[:n_samples] 

class RandomSampler(Sampler):
    def __init__(self, model):
        Sampler.__init__(self, model)
        self.name = 'RandomSampler'
        self.hyp_rec = 0

    def select_next_words(self, next_costs, next_probs, step_num, how_many):
        # Choice is complaining
        next_probs = next_probs.astype("float64") 
        word_indx = numpy.array([self.model.rng.choice(self.model.idim, p = x/numpy.sum(x))
                                    for x in next_probs], dtype='int32')
        beam_indx = range(next_probs.shape[0])

        args = numpy.ravel_multi_index(numpy.array([beam_indx, word_indx]), next_costs.shape)
        return (beam_indx, word_indx), next_costs.flatten()[args]

class BeamSampler(Sampler):
    def __init__(self, model):
        Sampler.__init__(self, model)
        self.name = 'BeamSampler'
        self.hyp_rec = 3

    def select_next_words(self, next_costs, next_probs, step_num, how_many):
        # Pick only on the first line (for the beginning of sampling)
        # This will avoid duplicate <q> token.
        if step_num == 0:
            flat_next_costs = next_costs[:1, :].flatten()
        else:
            # Set the next cost to infinite for finished utterances (they will be replaced)
            # by other utterances in the beam
            flat_next_costs = next_costs.flatten()
         
        voc_size = next_costs.shape[1]
         
        args = numpy.argpartition(flat_next_costs, how_many)[:how_many]
        args = args[numpy.argsort(flat_next_costs[args])]
        
        return numpy.unravel_index(args, next_costs.shape), flat_next_costs[args]
        

