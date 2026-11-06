# Deep learning segmentation of corneal layers in AS-OCT and layer-specific biomarkers for keratoconus detection

Code of the study *Deep Learning Segmentation of Corneal Layers in Anterior Segment OCT: Fuzzy Kappa
Reliability and Layer-Specific Biomarkers for Keratoconus Detection* (Diagnostics, MDPI; manuscript
diagnostics-4629115). Section numbers below refer to the paper.

The repository contains the implementation of the locked analysis chain: preprocessing, the
segmentation network and its training configuration, inference with the locked model, the layer
measurement chain and the 49 predefined biomarkers, the keratoconus classifier with its validation
design, the agreement coefficients of the reliability sub-study, the sensitivity analysis of the
Bowman's layer boundaries and the data-partitioning protocol.

Not included: the OCT images and the label masks (patient privacy), the trained network weights, and
the device-comparison module of the CorneApp web application (Section 2.7). The web application is
available for research use at https://corneapp.com.

## Layout

| Folder | Content | Paper |
|---|---|---|
| `segmentation/preprocessing.py` | DICOM reading, lateral rescaling to 1024 columns, 640 x 640 window, 1st-99th percentile normalisation, coordinate channels, axial resampling to 9.243 um in the measurement chain | 2.3 |
| `segmentation/model.py` | Four-level U-Net (48-96-192-384, bottleneck 768) with two deep-supervision heads | 2.5, Fig. 2B |
| `segmentation/losses.py` | Class-wise focal Tversky, Dice, OHEM (top 25 %); boundary and continuity terms as run (no meaningful contribution, see module docstring) | 2.5, Fig. 2C |
| `segmentation/data.py` | Dataset, augmentation, Bowman-centred 256 x 256 patches, sampling weights, person-level split | 2.5 |
| `segmentation/prepare_data.py` | DICOM + mask -> processed arrays | 2.3 |
| `segmentation/train.py` | Training, model selection by validation macro-Dice, early stopping, locking | 2.5, Fig. 2D |
| `segmentation/infer.py` | Inference with the locked model | 2.5 |
| `segmentation/evaluate.py` | Dice, NSD, HD95, ASSD in the held-out test split | 3.2, Table S6 |
| `measurement/features.py` | Perpendicular thickness, quality filter, conic fits, ring zones, apex window, asymmetry, the 49 biomarkers, format-specific scale factors | 2.6, Fig. 3, Table S2 |
| `measurement/build_feature_table.py` | Segment and measure a cohort -> image-level biomarker table | 2.5-2.6 |
| `classification/partition.py` | Development / independent split from the person identifier | 2.2 |
| `classification/chain.py` | Eye-level table, LOPO, within-fold selection, STABLE core, Youden thresholds, independent evaluation, person-clustered bootstrap, paired AUC differences, out-of-fold SHAP | 2.8-2.9, Table 4, Fig. 5-6 |
| `classification/stable8_run.py` | Fixed eight-feature model (separate run) | 2.8, Table 4 |
| `classification/univariate_tests.R` | Eye-level AUC, Rosner-Glynn-Lee clustered Wilcoxon test (clusrank), Benjamini-Hochberg | 2.9, Table 5, Table S5 |
| `classification/sensitivity_bowman.py` | +-1-pixel shift of the Bowman's layer boundaries | 2.9, 3.4, Table S7, Fig. S2 |
| `classification/compare_classifiers.py` | Standard parameters only (eight pachymetry and keratometry parameters), logistic regression and random forest with the same validation design; paired AUC differences | 2.9, 3.4, Table S8 |
| `agreement/kappa.py` | Cohen's kappa, Fleiss' kappa, fuzzy kappa (IFK-2) in the central +-5 mm band, person-clustered bootstrap | 2.4, Table 2, Table S3 |
| `agreement/layerwise_tolerant_kappa.py` | Layer-wise tolerant kappa with class-specific tolerances | 2.4, Table S4 |
| `config/` | Training and classifier settings, scale factors, split configuration, partitioning protocol | |

## Running the chain

```
pip install -r requirements.txt

# 1. processed arrays for training (DICOM sections + label masks of the segmentation pool)
python -m segmentation.prepare_data --dicom-dir DICOM --mask-dir MASKS --out-dir processed

# 2. training and locking (person lists in config/segmentation_split.json)
python -m segmentation.train --data-dir processed --split config/segmentation_split.json --out-dir model

# 3. layer-wise test metrics of the locked model (predictions saved as <stem>_pred.npy on the training grid coding)
python -m segmentation.evaluate --pred-dir predictions --data-dir processed --stems model/split_images.json

# 4. biomarker table of the classification cohort and the person-level split
python -m measurement.build_feature_table --weights model/best_unet.pth --index cohort_index.csv --out features.csv
python -m classification.partition --features features.csv --out features_split.csv

# 5. classification, validation design, SHAP
python -m classification.chain --features features_split.csv --out-dir results --shap
python -m classification.stable8_run --features features_split.csv

# 6. univariate discrimination with the clustered test
python -m classification.export_eye_table --features features_split.csv --out eye_table_development.csv
Rscript classification/univariate_tests.R eye_table_development.csv univariate_results.csv

# 7. sensitivity of the classification to a one-pixel shift of the Bowman's layer boundaries
python -m classification.sensitivity_bowman features_split.csv sensitivity_results.json

# 8. standard parameters only and other classifiers with the same validation design
python -m classification.compare_classifiers --features features_split.csv --out table_s8.json
```

`cohort_index.csv` lists the DICOM sections with the columns `path, stem, pid, eye, label`
(`stem` unique per image, `pid` the person identifier, `eye` R or L, `label` 1 keratoconus / 0 healthy).

Label coding of the masks: 0 background, 2 anterior tear-film band, 3 epithelium, 4 Bowman's layer,
5 stroma, 6 Descemet-endothelium complex, 7 unlabelled.

## Reproducibility notes

* All scripts use fixed seeds (42). Training on a GPU with mixed precision is not bit-wise
  reproducible; in the study the best validation macro-Dice was reached at epoch 36 and training
  stopped at epoch 51.
* The XGBoost results depend on the library version and on row/column subsampling; re-running the
  classification chain reproduces the reported AUC values within about 0.01 (the sensitivity analysis
  in Table S7 reports its own reproduced baseline).
* Software used in the study: Python 3, PyTorch, XGBoost, scikit-learn, SciPy, SHAP; R with clusrank.
* The numerical values of the format-specific scale factors (`config/scale_factors.json`) are not
  distributed with this repository; they are available from the authors on request and must be entered
  before steps 4 and 7.

## Citation

Gunay Sener, A.B.; Goksuluk, D.; Ozturk, A.; Sener, H.; Unlu, M.; Ozer, F. Deep Learning Segmentation of
Corneal Layers in Anterior Segment OCT: Fuzzy Kappa Reliability and Layer-Specific Biomarkers for
Keratoconus Detection. Diagnostics (under review).

## Licence

MIT (see `LICENSE`).

## Column conventions of the biomarker table

`stem` image id, `pid` person id, `goz` eye (R/L), `goz_id` eye id (`pid_eye`), `label` (1 keratoconus,
0 healthy), `format` export format (`wide` / `narrow`), `kume` partition (`gelistirme` = development,
`bagimsiz` = independent); the 49 biomarkers carry the prefixes `A_`, `B_`, `C_` (codes in Table S2 of the paper).
