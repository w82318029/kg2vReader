# #!/usr/bin/env bash


printf "New start\n" > testLog
for i in 68020
do
   printf $i >> testLog
   printf  "\n" >> testLog
   python evluation/4to1.py menuOut/beam_search_$i
   perl evluation/multi-bleu.perl Data/menu/test_responses.txt < menuOut/beam_search_${i}___ >> testLog
   python evluation/embedding_metrics_test.py menuOut/beam_search_${i}___ >> testLog
   #python count_diversity.py nlpccOut/beam_search_${i}___ >> testLog
   python evluation/3entropy.py menuOut/beam_search_${i}___ >> testLog
   python evluation/entityCount.py menuOut/beam_search_${i}___ >> testLog
done

