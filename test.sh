# #!/usr/bin/env bash
for i in 68020
do
    THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu4 python sample.py menu1.8wanci/1546002307.2_TwitterModel__$i Data/menu/test_contexts.txt menuOut/beam_search_$i --beam_search --n-samples=5 --ignore-unk --verbose
done
