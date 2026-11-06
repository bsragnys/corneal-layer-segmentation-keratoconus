# Data-partitioning protocol

All partitions are made at the person level: all eyes, series and sections of a person belong to one
partition only. The unit of clustering in every validation design and confidence interval is the person.

## Segmentation pool (29 persons, 53 eyes, 674 manually labelled images)

* Held-out test split: 11 persons (151 images) fixed before training (`fixed_test_persons`).
* Remaining persons: divided into training and validation with `sklearn.model_selection.train_test_split`
  (`test_size=0.25`, `random_state=42`) after forcing the persons listed in `forced_train_persons` into the
  training split. Result in the study: training 16 persons (439 images), validation 2 persons (84 images).
* The validation split is used only for early stopping and model selection; the test split is used once,
  with the locked model.

Implementation: `segmentation/data.py::create_split`, configuration `config/segmentation_split.json`.

## Classification cohort (180 persons, 354 eyes, 3725 images)

* Development set / independent set: a fixed rule computed from the person identifier string
  (`hashlib.sha1(identifier).hexdigest()` read as an integer; even = development, odd = independent).
  Result in the study: development 83 persons (165 eyes, 1683 images), independent 97 persons
  (189 eyes, 2042 images).
* Within the development set: leave-one-person-out cross-validation (all eyes of one person held out per fold).
* The independent set is evaluated once with the classifiers trained on the whole development set and the
  thresholds carried unchanged; it is not used in any selection or tuning decision.

Implementation: `classification/partition.py`, `classification/chain.py`.

## Overlaps between samples (Section 2.2 of the paper)

22 of the 27 common-set images of the reliability sub-study were also in the segmentation pool; all 29
persons of the segmentation pool were in the classification cohort (11 development, 18 independent);
94 of the 104 persons of the device-agreement sample were in the classification cohort.
