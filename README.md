# results

## installation

install docker image:

```
docker run --gpus all -it --name pipr nvcr.io/nvidia/tensorflow:21.12-tf2-py3
```

in the docker container, install python packages:

```
pip install keras==2.2.4 numpy==1.12.0 scikit-learn
```

exit the docker container using CTRL + D.

commit the docker container to a new image:

```
docker commit pipr piper
```

run the new docker image (EDIT THE VOLUME PATH!):

```
docker run --gpus all -it --rm -v /home/gumougeot/all/codes/python/pipr/seq_ppi:/home -w /home piper
```

in the docker container, run the python script:

```
python model_test.py \
--model model_save_yeast.h5 \
--dic ath/protein.dictionary.tsv \
--act ath/protein.actions.tsv \
--csv results/yeast_ath.csv
```

or use the `run_test.sh` file:

```
./run_test.sh
```


## over the Arabidopsis thaliana dataset

| model                            | auc                 | acc                | precision          | recall             |
|----------------------------------|---------------------|--------------------|--------------------|--------------------|
| model_save2022_03_31_02_04_PM.h5 | 0.596352429902435   | 0.56636553161918   | 0.6385372714486639 | 0.5529841656516443 |
| yeast                            | 0.56558488565094    | 0.546907574704656  | 0.6011976047904192 | 0.6114494518879415 |
| yeast (45ep)                     | 0.5200393969057154  | 0.4815844336344684 | 0.5383043922369765 | 0.6419001218026796 |
| multi species (MS)               | 0.45185541654331995 | 0.4419735927727589 | 0.5671641791044776 | 0.0925700365408039 |
| multi species (MS10yeast_30)     | 0.4790570530747503  | 0.47810979847116053| 0.574468085106383  | 0.32886723507917176|
| multi species (MSfiltered25)     | 0.4273644300704924  | 0.45239749826268244| 0.6240601503759399 | 0.10109622411693057|
