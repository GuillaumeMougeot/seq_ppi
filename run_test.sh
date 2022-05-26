# yeast model vs yeast dataset
python model_test.py \
--model model_save_yeast.h5 \
--dic yeast/preprocessed/protein.dictionary.tsv \
--act yeast/preprocessed/protein.actions.tsv \
--csv results/yeast_yeast.csv

# yeast model vs ath dataset
python model_test.py \
--model model_save_yeast.h5 \
--dic ath/protein.dictionary.tsv \
--act ath/protein.actions.tsv \
--csv results/yeast_ath.csv

# MS model vs ath dataset
python model_test.py \
--model model_save_MS.h5 \
--dic ath/protein.dictionary.tsv \
--act ath/protein.actions.tsv \
--csv results/ms_ath.csv