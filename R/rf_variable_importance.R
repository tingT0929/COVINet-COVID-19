library(ranger)
library(data.table)

dat_cov <- read.csv("data/data-14-new-.csv")
dat_basic <- dat_cov[, 1:4]
dat_cov$date <- as.Date(dat_cov$date)
dat_cov <- dat_cov[, -(1:4)]
dat_cov <- na.omit(dat_cov)

dat_death <- dat_cov[, -c(1:14 * 2 - 1, 76)]
dat_case <- dat_cov[, -c(1:14 * 2, 77)]
print(colnames(dat_case))
print(colnames(dat_death))

# # ------------ dead cases ------------ #
rg.dead <- ranger(formula = cum_dead ~ ., data = dat_death, importance = "impurity")
print(rg.dead$prediction.error)
rg.dead.imp <- sort(rg.dead$variable.importance, decreasing = TRUE)

save(rg.dead.imp, file = "result/rf_fit_dead_imp.rda")
write.csv(data.frame(rg.dead.imp), file = "result/rf_fit_death_imp.csv")
save(rg.dead, file = "result/rf_fit_dead.rda")

# ------------ cumulative confirmed cases ------------ #
rg.confirm <- ranger(formula = cum_confirm ~ ., data = dat_case, importance = "impurity")
print(rg.confirm$prediction.error)
rg.confirm.imp <- sort(rg.confirm$variable.importance, decreasing = TRUE)

save(rg.confirm.imp, file = "result/rf_fit_confirm_imp.rda")
write.csv(data.frame(rg.confirm.imp), file = "result/rf_fit_confirm_imp.csv")
save(rg.confirm, file = "result/rf_fit_confirm.rda")