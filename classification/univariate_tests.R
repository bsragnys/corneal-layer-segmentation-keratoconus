# Univariate discrimination of the 49 biomarkers (Section 2.9; Table 5; Table S5).
# Eye-level AUC with person-clustered bootstrap CI, the Rosner-Glynn-Lee clustered Wilcoxon
# rank-sum test with the person as the cluster (R package clusrank) and Benjamini-Hochberg
# correction over the family of 49 tests. The eye-level Mann-Whitney U test is reported for comparison.
#
# usage: Rscript classification/univariate_tests.R eye_table_development.csv univariate_results.csv
# (eye_table_development.csv: one row per eye with the columns pid, label and the 49 biomarkers;
#  produced by classification/export_eye_table.py)
suppressPackageStartupMessages(library(clusrank))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[1], stringsAsFactors = FALSE)
feat <- grep("^(A_|B_|C_)", names(d), value = TRUE)
set.seed(42)
auc_fn <- function(y, s) { r <- rank(s); n1 <- sum(y == 1); n0 <- sum(y == 0); (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0) }
res <- data.frame()
persons <- unique(d$pid)
for (f in feat) {
  v <- d[!is.na(d[[f]]), c("pid", "label", f)]
  if (length(unique(v$label)) < 2) next
  a <- auc_fn(v$label, v[[f]]); a_dir <- max(a, 1 - a)
  bs <- replicate(2000, { sel <- sample(unique(v$pid), replace = TRUE); vv <- do.call(rbind, lapply(sel, function(p) v[v$pid == p, ])); aa <- auc_fn(vv$label, vv[[f]]); max(aa, 1 - aa) })
  ct <- tryCatch(clusWilcox.test(v[[f]] ~ v$label + cluster(v$pid), data = v, method = "rgl"), error = function(e) NULL)
  mw <- wilcox.test(v[[f]][v$label == 1], v[[f]][v$label == 0])
  res <- rbind(res, data.frame(biomarker = f, direction = ifelse(a >= 0.5, "higher", "lower"), auc = a_dir,
                               ci_low = quantile(bs, 0.025), ci_high = quantile(bs, 0.975),
                               p_clustered = if (is.null(ct)) NA else ct$p.value, p_mannwhitney = mw$p.value, n_eyes = nrow(v)))
}
res$q_clustered <- p.adjust(res$p_clustered, method = "BH")
res$q_mannwhitney <- p.adjust(res$p_mannwhitney, method = "BH")
res <- res[order(-res$auc), ]
write.csv(res, args[2], row.names = FALSE)
cat("significant after BH (clustered test):", sum(res$q_clustered < 0.05, na.rm = TRUE), "of", nrow(res), "\n")
