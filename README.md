# Deep learning segmentation of corneal layers in AS-OCT and layer-specific biomarkers for keratoconus detection

Code of the study *Deep Learning Segmentation of Corneal Layers in Anterior Segment OCT: Fuzzy Kappa
Reliability and Layer-Specific Biomarkers for Keratoconus Detection* (Diagnostics, MDPI; manuscript
diagnostics-4701353). Section numbers below refer to the paper.

The repository contains the implementation of the locked analysis chain: preprocessing, including the
identification of the export type and the refraction correction of uncorrected sections by ray tracing,
the segmentation network and its training configuration, inference with the locked model, the layer
measurement chain and the 49 predefined biomarkers, the keratoconus classifier with its validation
design, the agreement coefficients of the reliability sub-study, the additional analyses of the paper
(standard parameters alone, other classifiers, export format, persons outside the development steps,
sensitivity to the Bowman's layer boundaries, column-wise boundary comparison) and the data-partitioning
protocol.

Not included: the OCT images and the label masks (patient privacy), the trained network weights, and
the device-comparison module of the CorneApp web application (Section 2.7). The web application is
available for research use at https://corneapp.com.

## Layout

| File | Content | Paper |
|---|---|---|
| `segmentation/preprocessing.py` | DICOM reading, lateral rescaling to 1024 columns, 640 x 640 window, 1st-99th percentile normalisation, coordinate channels, axial resampling to 9.243 um in the measurement chain, identification of the export type from the image | 2.3 |
| `segmentation/model.py` | Four-level U-Net (48-96-192-384, bottleneck 768) with two deep-supervision heads | 2.5, Fig. 2B |
| `segmentation/losses.py` | Class-wise focal Tversky, Dice, OHEM (top 25 %); boundary and continuity terms as run (no meaningful contribution, see module docstring) | 2.5, Fig. 2C |
| `segmentation/data.py` | Dataset, augmentation, Bowman-centred 256 x 256 patches, sampling weights, person-level split | 2.5 |
| `segmentation/prepare_data.py` | DICOM + mask -> processed arrays | 2.3 |
| `segmentation/train.py` | Training, model selection by validation macro-Dice, early stopping, locking | 2.5, Fig. 2D |
| `segmentation/infer.py` | Inference with the locked model | 2.5 |
| `segmentation/evaluate.py` | Dice, NSD, HD95, ASSD in the held-out test split | 3.2, Table S6 |
| `segmentation/boundary_columnwise.py` | Column-wise boundary distances and thickness error of Bowman's layer and the Descemet-endothelium complex in the held-out test split | 2.9, 3.2, Table S11 |
| `measurement/refraction.py` | Refraction correction of uncorrected sections by ray tracing: anterior surface from a first segmentation pass, Snell's law, group refractive index 1.389, resampling | 2.3 |
| `measurement/features.py` | Perpendicular thickness, quality filter, conic fits, ring zones, apex window, asymmetry, the 49 biomarkers, common scale factor | 2.6, Fig. 3, Table S2 |
| `measurement/build_feature_table.py` | Export type, refraction correction, segmentation and measurement of a cohort -> image-level biomarker table | 2.3, 2.5-2.6 |
| `classification/partition.py` | Development / independent split from the person identifier | 2.2 |
| `classification/chain.py` | Eye-level table, seven feature arms, LOPO, Youden thresholds, final models, independent evaluation, person-clustered bootstrap, paired AUC differences, out-of-fold SHAP | 2.8-2.9, Table 4, Fig. 5-6 |
| `classification/univariate_tests.R` | Eye-level AUC, Rosner-Glynn-Lee clustered Wilcoxon test (clusrank), Benjamini-Hochberg | 2.9, Table 5, Table S5 |
| `classification/compare_classifiers.py` | Eight standard pachymetric and keratometric parameters alone; logistic regression and random forest with the same validation design; paired AUC differences | 2.9, 3.4, Table 4, Table S8 |
| `classification/format_analyses.py` | By export format: confusion and missing-value matrices, export format and missing values as predictors, arms retrained in wide-format eyes alone, separation of the formats by each biomarker in healthy eyes | 2.9, 3.3-3.4, Tables S9-S10 |
| `classification/subset_analyses.py` | Persons of the independent set who took part in no development step: AUC, sensitivity and specificity at the locked threshold | 2.2, 2.9, 3.4 |
| `classification/sensitivity_bowman.py` | +-1-pixel shift of the Bowman's layer boundaries with the saved full model and its locked threshold | 2.9, 3.4, Table S7, Fig. S2 |
| `classification/scale_invariance.py` | Classification repeated without the common scale factor | 2.7 |
| `agreement/kappa.py` | Cohen's kappa, Fleiss' kappa, fuzzy kappa (IFK-2) in the central +-5 mm band, person-clustered bootstrap | 2.4, Table 2, Table S3 |
| `agreement/layerwise_tolerant_kappa.py` | Layer-wise tolerant kappa with class-specific tolerances | 2.4, Table S4 |
| `config/` | Training and classifier settings, scale factor, split configuration, partitioning protocol | |

## Running the chain

```
pip install -r requirements.txt

# 1. processed arrays for training (DICOM sections + label masks of the segmentation pool)
python -m segmentation.prepare_data --dicom-dir DICOM --mask-dir MASKS --out-dir processed

# 2. training and locking (person lists in config/segmentation_split.json)
python -m segmentation.train --data-dir processed --split config/segmentation_split.json --out-dir model

# 3. test metrics of the locked model (predictions saved as <stem>_pred.npy on the training grid coding)
python -m segmentation.evaluate --pred-dir predictions --data-dir processed --stems model/split_images.json
python -m segmentation.boundary_columnwise --pred-dir predictions --data-dir processed --stems model/split_images.json

# 4. biomarker table of the classification cohort (export type, refraction correction, measurement)
#    and the person-level split
python -m measurement.build_feature_table --weights model/best_unet.pth --index cohort_index.csv --out features.csv
python -m classification.partition --features features.csv --out features_split.csv

# 5. classification: seven feature arms, validation design, final models, SHAP
python -m classification.chain --features features_split.csv --out-dir results --shap

# 6. univariate discrimination with the clustered test
python -m classification.export_eye_table --features features_split.csv --out eye_table_development.csv
Rscript classification/univariate_tests.R eye_table_development.csv univariate_results.csv

# 7. standard parameters alone and other classifiers with the same validation design
python -m classification.compare_classifiers --features features_split.csv --out results/table_s8.json

# 8. analyses by export format (uses the outputs of step 5)
python -m classification.format_analyses --features features_split.csv --results results --out results/format_analyses.json

# 9. persons of the independent set outside the development steps (person lists: one identifier per line)
python -m classification.subset_analyses --features features_split.csv --results results \
    --exploratory exploratory_persons.txt --segmentation-pool pool_persons.txt --out results/subset_analyses.json

# 10. sensitivity to a one-pixel shift of the Bowman's layer boundaries (saved full model, locked threshold)
python -m classification.sensitivity_bowman --features features_split.csv --results results --out results/sensitivity_bowman.json

# 11. classification repeated without the common scale factor
python -m classification.scale_invariance --features features_split.csv --remove-scale-factor
```

`cohort_index.csv` lists the DICOM sections with the columns `path, stem, pid, eye, label`
(`stem` unique per image, `pid` the person identifier, `eye` R or L, `label` 1 keratoconus / 0 healthy).

Label coding of the masks: 0 background, 2 anterior tear-film band, 3 epithelium, 4 Bowman's layer,
5 stroma, 6 Descemet-endothelium complex, 7 unlabelled.

## Reproducibility notes

* All scripts use fixed seeds (42). Training on a GPU with mixed precision is not bit-wise
  reproducible; in the study the best validation macro-Dice was reached at epoch 36 and training
  stopped at epoch 51.
* The XGBoost results depend on the library version and on the order of the rows and columns of the
  table (row and column subsampling). The values in the paper were obtained with XGBoost 3.2.0; with
  that version, the same table and the same order of rows and columns, the classification scripts
  give the reported AUC values, confidence intervals, thresholds and confusion matrices.
* Person-clustered bootstrap intervals were computed in two runs of the study with two NumPy random
  generators (`generator="pcg64"` for the seven arms and their paired comparisons,
  `generator="mt19937"` for the additional analyses; see `classification/chain.py`). Each script
  uses the generator with which its table was computed.
* The numerical value of the common scale factor (`config/scale_factors.json`) is not distributed
  with this repository; it is available from the authors on request and must be entered before
  steps 4 and 10. The classification does not depend on it (step 11).
* Software used in the study: Python 3, PyTorch, XGBoost, scikit-learn, SciPy, SHAP; R with clusrank.

## Citation

Gunay Sener, A.B.; Goksuluk, D.; Ozturk, A.; Sener, H.; Unlu, M.; Ozer, F. Deep Learning Segmentation of
Corneal Layers in Anterior Segment OCT: Fuzzy Kappa Reliability and Layer-Specific Biomarkers for
Keratoconus Detection. Diagnostics (under review).

## Licence

MIT (see `LICENSE`).

## Column conventions of the biomarker table

`stem` image id, `pid` person id, `goz` eye (R/L), `goz_id` eye id (`pid_eye`), `label` (1 keratoconus,
0 healthy), `format` export format (`wide` / `narrow`), `refraction_corrected` export type determined
from the image, `kume` partition (`gelistirme` = development, `bagimsiz` = independent); the 49
biomarkers carry the prefixes `A_`, `B_`, `C_` (codes in Table S2 of the paper).
