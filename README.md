
# Spatiotemporal Latent Factors for Basketball

This is a repository 
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required libraries. Should include a requirements.txt file in the future.

## Usage

```bash
bash run_bball_fixed.sh data_folder output_folder
```

## Branches 

codewithtimedim branch gives us a representation with time as one of the latent factor dimensions. 

polarcoordbranch does not yet have the time dimension functioning yet 

## Pre-Processing

MZK fill this

## Training 

Run prepare_bball.py and run_bball.py with arguments to run a single experiment. prepare_bball.py creates the training dataset and sets miscellaneous parameters for a single trial (logger, seed, etc.)

python prepare_bball.py \
    --root-dir $RUN_DIR \
    -data-dir $DATA_DIR

python run_bball.py \
    --root-dir $RUN_DIR \
    --data-dir $DATA_DIR \ 
    --type $TYPE \ 
    --stop-cond $STOP_CONDITION \ 
    --batch_size $BATCH_SIZE \ 
    --sigma $SIGMA \ 
    --K $K \ 
    --step_size 1 \ 
    --gamma 0.95 \ 
    --full_lr $FULL_LR \ 
    --full_reg $FULL_REG \ 
    --low_lr $LOW_LR \ 
    --low_reg $LOW_REG


Helper scripts are provided in src/ to do 10 trials of fixed vs multi resolution and stop_condition experiments.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
