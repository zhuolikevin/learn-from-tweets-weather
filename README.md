# Learn Sentiment from Tweets with Weather

## Run Scripts

### Local Run

- Split `train.csv` to two files. 80% for training and 20% for testing

  ```bash
  $ python src/split_csv.py
  ```

- Then we have `output_1.csv` and `output_2.csv` in `res/`. Rename them to `new_train.csv` and `new_test.csv` respectively

  ```bash
  $ mv res/output_1.csv res/new_train.csv
  $ mv res/output_2.csv res/new_test.csv
  ```

- Run training

  ```bash
  $ python src/main.py
  ```

### Submit in Kaggle

- Use the following command to generate output for Kaggle

  ```bash
  $ python src/main.py res/output.csv
  ```
