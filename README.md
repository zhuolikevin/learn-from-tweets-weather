# Learn Sentiment from Tweets with Weather

## Run Scripts

### Local Run

- Split `train.csv` to two files. 80% for training and 20% for testing. In `src/`

  ```bash
  $ python split_csv.py
  ```

- Then we have `output_1.csv` and `output_2.csv` in `res/`. Rename them to `new_train.csv` and `new_test.csv` respectively

  ```bash
  $ mv res/output_1.csv res/new_train.csv
  $ mv res/output_2.csv res/new_test.csv
  ```

- In `src/`, run training

  ```bash
  $ python main.py
  ```

### Submit in Kaggle

- In `src/`, use the following command to generate output for Kaggle

  ```bash
  $ python main.py ../res/output.csv
  ```
