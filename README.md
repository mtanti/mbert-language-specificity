# MUFINS project

## Introduction

This project contains the framework and set of experiments used in the MUFINS project (Multilingual Financial News Summarisation).
The MUFINS project is a research project for analysing and improving ways to create multilingual natural language processing with minimal resources.

## Getting started

To be able to run the experiments in this project, you will need to run `create_venv.sh` (in Linux) or `create_venv.bat` (in Windows).
This will create a Python virtual environment in a folder called venv_mufins using the `venv` module and install the project locally (as module `mufins`) in the virtual environment.
The script will also compile the code documentation and test the program.

## Glossary

- `data processor`: A script that takes in raw data (as downloaded) and outputs it in a standard format.
- `experiment`: A script that takes in a processed data set, and trains/evaluates a number of models on it.

## Project structure

- `bin`: Contains command line runnable Python programs to run experiments and data processors (use `--help` for help on command line arguments).
- `docs`: Contains documentation written in Sphinx.
- `tools`: Contains command line runnable Python programs to help with development of the project.
- `mufins/common`: Contains the common Python modules to be imported by experiments and data processors.
- `mufins/dataprocs`: Contains the data processors, organised by data set.
- `mufins/experiments`: Contains the experiments.
- `mufins/resources`: Contains constants such as the chart colour list and JSON schemas.
- `mufins/tests`: Contains unit tests.

## Data processors

All data sets are processed into a standard format which consists of the following files:

- `about.csv`: Contains information about the time the data processor was run (timestamp, version, hostname, and directory path).
- `log.txt`: Contains everything that was displayed in the program whilst running.
- `dataset_version.txt`: Contains the version of the processed data set (each time the data processor is changed in a way that changes the data set, the version number is incremented).
- `label_names.txt`: Contains a line separated list of label names in the data set in the order in which it will be indexed in the neural network.
- `lang_names.txt`: Contains a line separated list of language names in the data set in the order in which it will be indexed in the neural network.
- `data_spec.json`: Contains JSON encoded information needed to load the data set using the data processor's `DataSpec` class.
- `dataset_train.hdf`: The training set HDF5 file to be loaded using the main `DataSpec` class.
- `dataset_val.hdf`: The validation set HDF5 file to be loaded using the main `DataSpec` class.
- `dataset_dev.hdf`: The development set HDF5 file to be loaded using the main `DataSpec` class.
- `dataset_test.hdf`: The test set HDF5 file to be loaded using the main `DataSpec` class.
- `dataset_train.csv`: The training set in human readable form.
- `dataset_val.csv`: The validation set in human readable form.
- `dataset_dev.csv`: The development set in human readable form.
- `dataset_test.csv`: The test set in human readable form.

## Experiments

Experiments generate the following files:

- `about.csv`: Contains information about the time the experiment was run (timestamp, version, hostname, and directory path).
- `log.txt`: Contains everything that was displayed in the program whilst running.
- `checkpoint.sqlite3`: Contains the SQLite3 database which stores checkpointing information to resume progress in case of interruption (loaded using the `CheckpointManager` class).
- `parameter_space.txt`: Contains the hyperparameters to use in the experiment, with missing hyperparameters being taken from the default values used in the experiment's command line arguments.
- `results.csv`: Contains the results of the experiment with a row for each set of hyperparameters in `parameter_space.txt`.
- `results/<exp_id>/lang_outputs.csv`: Contains human readable language classifier outputs from the test set.
- `results/<exp_id>/label_outputs.csv`: Contains human readable label classifier outputs from the test set.
- `results/<exp_id>/model.pkl`: The trained PyTorch model to be loaded using the experiment's `Model` class.
- `results/<exp_id>/train_history.csv`: Contains the validation set performance for each epoch.
- `results/<exp_id>/model.json`: Contains JSON encoded information needed to load the model using the experiment's `Model` class.
- `results/<exp_id>/plots/test/legend_lang.pdf`: Contains the chart legend with the colours of the different languages.
- `results/<exp_id>/plots/test/legend_label.pdf`: Contains the chart legend with the colours of the different labels.
- `results/<exp_id>/plots/test/lang/plot.pdf`: Contains the chart t-SNE plot of the language data set's test set representations coloured by language.
- `results/<exp_id>/plots/test/lang/plot.csv`: Contains the t-SNE plot coordinates of the language data set's test set representations.
- `results/<exp_id>/plots/test/lang/plot_fullenc.csv`: Contains the uncompressed representations of the language data set's test set.
- `results/<exp_id>/plots/test/label/lang_plot.pdf`: Contains the chart t-SNE plot of the label data set's test set representations coloured by language.
- `results/<exp_id>/plots/test/label/label_plot_no-en.pdf`: Contains the chart t-SNE plot of the label data set's test set label representations coloured by label with the English data being left out.
- `results/<exp_id>/plots/test/label/label_plot.pdf`: Contains the chart t-SNE plot of the label data set's test set label representations coloured by label.
- `results/<exp_id>/plots/test/label/plot.csv`: Contains the t-SNE plot coordinates of the label data set's test set representations.
- `results/<exp_id>/plots/test/label/plot_fullenc.csv`: Contains the uncompressed representations of the label data set's test set.

## Hyperparameter search

To tune hyperparameters, there is a script in the `bin` directory called `random_parameter_space_generator.py`.
You must provide a hyperparameter specification consisting of a JSON file with the following format:

```
{
    "<hyperparameter name>": {
        "dtype": "bool"|"str"|"int"|"float",
        "values": [<list of possible values to use>]
    },
    ...
}
```

The script will then generate a number of random hyperparameter combinations by independently sampling from the list of values provided.
Only unique combinations are returned.
The resulting combinations are saved in a `parameter_space.txt` file.

You are then expected to run an experiment over all of these randomly generated hyperparameter combinations using the experiment command line argument `--hyperparameter_search_mode yes` which will run the experiment in minimalist mode to save time.
The results will contain the label classifier's performance for each hyperparameter combination, allowing you to pick the best combination and rerun the experiment in normal mode on said hyperparameters.
