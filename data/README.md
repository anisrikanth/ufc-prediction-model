# Local data

Place the training CSV in this directory, for example:

```text
data/ufc-master.csv
```

The repository intentionally does not redistribute the dataset. Before using
or publishing a dataset, review its original license and terms of use.

The loader validates the columns required by the model. They include:

- bout metadata: `date`, `Winner`, `title_bout`, `gender`, `weight_class`
- fighter identity and stance fields for the red and blue corners
- reach, age, striking, takedown, submission, experience, and win-method fields

Run the training command to receive a complete list if any required columns
are missing.
