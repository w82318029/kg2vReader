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
import search
import utils

from dialog_encdec import DialogEncoderDecoder
from numpy_compat import argpartition
from state import prototype_state

logger = logging.getLogger(__name__)

class Timer(object):
    def __init__(self):
        self.total = 0

    def start(self):
        self.start_time = time.time()

    def finish(self):
        self.total += time.time() - self.start_time

def parse_args():
    parser = argparse.ArgumentParser("Sample (with beam-search) from the session model")

    parser.add_argument("--ignore-unk",
            action="store_false",
            help="Allows generation procedure to output unknown words (<unk> tokens)")

    parser.add_argument("model_prefix",
            help="Path to the model prefix (without _model.npz or _state.pkl)")

    parser.add_argument("context",
            help="File of input contexts")

    parser.add_argument("output",
            help="Output file")
    
    parser.add_argument("--beam_search",
                        action="store_true",
                        help="Use beam search instead of random search")

    parser.add_argument("--n-samples",
            default="1", type=int,
            help="Number of samples")

    parser.add_argument("--n-turns",
                        default=1, type=int,
                        help="Number of dialog turns to generate")

    parser.add_argument("--verbose",
            action="store_true", default=False,
            help="Be verbose")

    parser.add_argument("changes", nargs="?", default="", help="Changes to state")
    return parser.parse_args()

def main():
    ####yawa add
    raw_dict = cPickle.load(open('./Data/Dataset.dict.pkl', 'r'))
    str_to_idx = dict([(tok, tok_id) for tok, tok_id, _, _ in raw_dict])
    idx_to_str = dict([(tok_id, tok) for tok, tok_id, _, _ in raw_dict])
    #########
    
    args = parse_args()
    state = prototype_state()

    state_path = args.model_prefix + "_state.pkl"
    model_path = args.model_prefix + "_model.npz"

    with open(state_path) as src:
        state.update(cPickle.load(src))

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    model = DialogEncoderDecoder(state) 
    
    sampler = search.RandomSampler(model)
    if args.beam_search:
        sampler = search.BeamSampler(model)

    if os.path.isfile(model_path):
        logger.debug("Loading previous model")
        model.load(model_path)
    else:
        raise Exception("Must specify a valid model path")
    
    contexts = [[]]
    lines = open(args.context, "r").readlines()
    if len(lines):
        contexts = [x.strip() for x in lines]
    #contexts = cPickle.load(open('./Data/Test.dialogues.pkl', 'r'))
    print('Sampling started...')
    context_samples, context_costs, att_weights, att_context= sampler.sample(contexts,
                                            n_samples=args.n_samples,
                                            n_turns=args.n_turns,
                                            ignore_unk=args.ignore_unk,
                                            verbose=args.verbose)
    print('Sampling finished.')
    print('Saving to file...')
     
    # Write to output file
    output_handle = open(args.output, "w")
    for context_sample in context_samples:
        print >> output_handle, '\t'.join(context_sample)
    outline = ''
    #for att_weight in att_weights:
        #for att_in in att_weight:
            #print >> output_handle, str(att_in)
    print "number of weights:" + str(len(att_weights))
    #for i in range(len(att_weights)):
    #outline = att_weights[0]
    cPickle.dump(att_weights, open('Data/beam_search_2000_2_weight.pkl', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(att_context, open('Data/beam_search_2000_2_context.pkl', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    #for i in range(len(att_context)):
        #print att_context[i]
    #print numpy.array(att_weights[0])
    #print type(att_weights[0])
    #aa = numpy.array(att_weights[0])
    #size  = aa.shape[1]
    #bb = aa.reshape(5,5,size/5)
    #print bb.shape
    
    output_handle.close()
    print('Saving to file finished.')
    print('All done!')

if __name__ == "__main__":
    main()

