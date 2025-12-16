# MOF Stability prediction Neural Network (MOFSNN)

This is a study of predicting various types of MOF stability using multi-task strategy.

## Raw data

The raw data is stored in the `raw_data` folder, which can get from literature.

### TSD & SSD

The thermal stability dataset and solvent stability dataset are get from [Nandy's work](https://pubs.acs.org/doi/10.1021/jacs.1c07217). The data is download from [here](https://zenodo.org/records/5737968/files/SciData.zip?download=1), which is published in [Nandy's another paper](https://www.nature.com/articles/s41597-022-01181-0). The CIF files are matched with MOFs in the CoREMOF2019 databse, which is available from [MOFX-DB](https://mof.tech.northwestern.edu/databases).

### WS24

The water stability, acid stability, base stability, and boiling stability dataset are get from [Terrones's work](https://pubs.acs.org/doi/10.1021/jacs.4c05879).
And the raw data is download from [here](https://zenodo.org/records/12110918).

### Download and extract data

```sh
wget https://zenodo.org/records/5737968/files/SciData.zip?download=1 -O Nandy_2022_SciData.zip
wget https://zenodo.org/api/records/12110918/files-archive -O WS24v2.zip
wget https://mof.tech.northwestern.edu/Datasets/CoREMOF%202019-mofdb-version:dc8a0295db.zip -O CoREMOF2019.zip
unzip Nandy_2022_SciData.zip
unzip WS24v2.zip
unzip CoREMOF2019.zip
```

Changes :

I have implemented the "Trend Detection" Verification Mode. You can now use the --verify flag to run a shorter, 100-epoch training session
(with reduced patience) to quickly gauge if your code changes are improving performance.

Here is the command to run your verification test (using main.py is recommended for checking code changes, but the flag works for
hyperopt.py too):

python -u CGCNN_MT/main.py --verify --progress_bar --task_cfg tsd_ssd_ws24 --model_cfg att_cgcnn --num_workers 14 --batch_size 32 --max_graph_len 200 --atom_fea_len 256 --extra_fea_len 16 --h_fea_len 128 --n_conv 6 --n_h 4 --dropout_prob 0.5 --use_cell_params --use_extra_fea --atom_layer_norm --loss_aggregation fixed_weight_sum --dl_sampler random --task_att_type self --lr 0.001 --lr_mult 10 --group_lr --optim_config fine --task_norm --log_dir logs

(Note: I removed --max_epochs 500 and --patience 50 from your original command because the --verify flag will override them to 100 and 20,
respectively.)
