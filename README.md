# Spatiotemporal Latent Factors for Basketball

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required libraries. Should include a requirements.txt file in the future.

## Usage

```bash
bash run_bball_fixed.sh data_folder output_folder
```

## Branches 

codewithtimedim branch gives us a representation with time as one of the latent factor dimensions. 

evolving-time-dimension branch gives us a representation per each quarter, with temporal regularization, that results in dynamic performance profiles.

playstyle branch gives us a representation with playstyle as one of the latent factor dimensions. 

different-playstyles branch gives us a representation per each playstyle.

## Dataset

The spatial data we used is available [here](https://github.com/sealneaward/nba-movement-data).

## Pre-Processing

The data was regularized to a 40x50 court, and truncated to remove players with few to no shot attempts, as well as edge cases (like someone shooting from behind the basket). We also added the quarter number and playstyle (determined through a K-means clustering algorithm run on an [NBA](https://www.nba.com) dataset) number and name. Each player was also encoded an 'a' value used for training, similar to a player_id.

## Training
Run `prepare_bball.py` and `run_bball.py` with arguments to run a single experiment. `prepare_bball.py` creates the training dataset and sets miscellaneous parameters for a single trial (logger, seed, etc.)
```bash
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
```

Helper scripts are provided in `src/` to do a trial of fixed vs multi resolution and stop_condition experiments.

To change which players to get a performance profile representation for, edit the player_ids array (or similar) to the 'a' value from the dataset being used corresponding to the desired player. For playstyle and different-playstyles, you must also switch the additional playstyle_id value for each player.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
