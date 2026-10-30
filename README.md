# CS6350 Assignment 2

Veer Shah, VMS220003. Individual submission.

## Setup

Use Python 3.9 to 3.11 and Java 17. Set JAVA_HOME if Java is not already configured. Open a terminal in the folder containing assignment2.py and run:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
spark-submit --master 'local[4]' --driver-memory 4g assignment2.py --part all
```

No compilation is needed. The tested setup uses local Spark with four CPU workers. Internet access is needed for the first run, which downloads both datasets into `data/` under the working directory.

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell and use `python` in place of `python3`. Spark on Windows may require additional Hadoop configuration; the Windows setup was not tested.

## Running individual parts

Use `--part friends` to run both recommendation methods or `--part nb` to run Naive Bayes. The default `--part all` runs everything. ALS checks eight parameter combinations across two folds before fitting the selected model on the full graph.

The default seed is 42. Use `--seed` to change it. The same ten sampled users are used for both recommendation methods. Dataset and output paths are relative to the working directory; running on a distributed cluster would require shared input storage.

## Output

The program prints both sets of recommendations as user ID, tab, and comma-separated recommended IDs. It also prints Naive Bayes accuracy and class priors.

Each run creates a `results/` folder containing the recommendation lists and JSON files with model settings, scores and classification metrics. Rerunning replaces those files. Use `--output another_results` to save a separate run. The submitted report includes the measured results and both recommendation lists.

## Assignment files

- `assignment2.py`: code for all three parts.
- `requirements.txt`: Python dependencies.
- `README.md`: setup and run instructions.
- `algorithm.md`: mutual-friend algorithm and pseudocode.
- The local submission also contains `report.pdf`, with preprocessing, ALS comparison, Naive Bayes pseudocode and measured results.
- `CoverPage.docx` is included in the local submission.

## Dataset sources

LiveJournal graph supplied with the assignment:
https://an-ml.s3.us-west-1.amazonaws.com/soc-LiveJournal1Adj.txt

20 Newsgroups corpus hosted for this submission:
https://raw.githubusercontent.com/veermshah/Big-Data-Assignment-2/08e03d3d51175d897c6c16279c33cb237a9b4c05/datasets/news20.tar.gz

This is an unchanged copy of the archive distributed by CMU. The download URL is pinned to a Git commit so later repository changes do not change the dataset. Original source and attribution:
https://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html

Repository with the code and dataset:
https://github.com/veermshah/Big-Data-Assignment-2

References and assistance are listed in the report and cover sheet.
