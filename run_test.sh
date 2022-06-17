# yeast model vs yeast dataset
# python model_test.py \
# --model model_save_yeast.h5 \
# --dic yeast/preprocessed/protein.dictionary.tsv \
# --act yeast/preprocessed/protein.actions.tsv \
# --csv results/yeast_yeast.csv

# yeast model vs ath dataset
# python model_test.py \
# --model model_save_yeast.h5 \
# --dic ath/protein.dictionary.tsv \
# --act ath/protein.actions.tsv \
# --csv results/yeast_ath.csv

# MS model vs ath dataset
# python model_test.py \
# --model model_save_MS.h5 \
# --dic ath/protein.dictionary.tsv \
# --act ath/protein.actions.tsv \
# --csv results/ms_ath.csv

# MS model 2 vs ath dataset
# python model_test.py \
# --model model_save_MS10yeast_30.h5 \
# --dic ath/protein.dictionary.tsv \
# --act ath/protein.actions.tsv \
# --csv results/ms_2_ath.csv

# yeast model 2 vs ath dataset
# python model_test.py \
# --model model_save_yeast_45ep.h5 \
# --dic ath/protein.dictionary.tsv \
# --act ath/protein.actions.tsv \
# --csv results/yeast_ath.csv

# MS model 2 vs ath dataset
# python model_test.py \
# --model model_save_MS10yeast_30.h5 \
# --dic ath/protein.dictionary.tsv \
# --act ath/protein.actions.tsv \
# --csv results/ms_2_ath.csv

# MS model 3 vs ath dataset
python model_test.py \
--model model_save_MSfiltered25_10ep.h5 \
--dic ath/protein.dictionary.tsv \
--act ath/protein.actions.tsv \
--csv results/ms_3_ath.csv