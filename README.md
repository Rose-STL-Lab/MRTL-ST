# Spatiotemporal Latent Factors for Basketball

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required libraries. Should include a requirements.txt file in the future.

## Usage

It is recommended to run all scripts on a server/machine with >128GB memory, >4 CPUs, >1 GPUs.
For UCSD Students, request access for a server on DataHub.

## Branches 

Each branch creates one of the four types of models described in the paper:

codewithtimedim branch gives us a representation with time as one of the latent factor dimensions. 

evolving-time-dimension branch gives us a representation per each quarter, with temporal regularization, that results in dynamic performance profiles.

playstyle branch gives us a representation with playstyle as one of the latent factor dimensions. 

different-playstyles branch gives us a representation per each playstyle.

## Dataset

The spatial data we used is available [here](https://github.com/sealneaward/nba-movement-data).

The input files to the scripts are .pkl files. In order to generate such .pkl file, follow the following steps.

Step 1:
Decompress the .7z files into .json files. Each of these files represent one NBA basketball game.

Step 2:
Compile the .json files into a .pkl file by running `new_read_raw.py`:

python new_read_raw.py \
    --input-dir $INPUT_DIR \ # location of the .json files
    --output-dir $OUTPUT_DIR \ # desired location of the .pkl file

## Pre-Processing

The data was regularized to a 40x50 court, and truncated to remove players with few to no shot attempts, as well as edge cases (like someone shooting from behind the basket). We also added the quarter number and playstyle (determined through a K-means clustering algorithm run on an [NBA](https://www.nba.com) dataset) number and name. Each player was also encoded an 'a' value used for training, similar to a player_id.

## Training
Step 1:
Run `prepare_bball.py` to create the training dataset and set miscellaneous parameters for a single trial (logger, seed, etc.)

python prepare_bball.py \
    --root-dir $RUN_DIR \
    -data-dir $DATA_DIR
    
Step 2:
Run `run_bball_stop_cond.sh` (can be found `src/`) in with arguments to run a single experiment. This bash file is a helper script that runs `run_bball.py` with preset parameters.

bash run_bball_stop_cond.sh \
    --root-dir $RUN_DIR \
    --data-dir $DATA_DIR \

```

To change which players to get a performance profile representation for, edit the player_ids array (or similar) to the 'a' value from the dataset being used corresponding to the desired player. For playstyle and different-playstyles, you must also switch the additional playstyle_id value for each player.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
