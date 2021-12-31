rm(list = ls())
library(ranger)
library(data.table)

setwd("~/GitHub/COVINet-COVID-19")
dat_cov <- fread("data/usa_county_latlomg.csv", data.table = F)

dat_cov$date <- as.Date(dat_cov$date)
date_all <- unique(dat_cov$date)
dat_cov <- dat_cov[, -1]
names(dat_cov)

days <- 30 # 7, 30

for (days in c(7, 30)) {
  # ------------------- Select 3 features ------------------- #
  sel_feature <- c("% Drive Alone to Work",
                   "Average Traffic Volume per Meter of Major Roadways",
                   "Income Ratio",
                   "LON",
                   "LAT")
  
  any(is.na(dat_cov$LAT))
  
  ind1 <- names(dat_cov) %in% sel_feature
  ind1[c(1:17, 65:66)] <- TRUE
  dat_ <- dat_cov[, ind1]
  names(dat_) <- gsub("% ", "", names(dat_))
  names(dat_) <- gsub(" ", "_", names(dat_))
  dat_train <- dat_[dat_$date %in% date_all[1:(length(date_all) - days)], ]
  names(dat_train)
  
  # ------------------- dead
  dead_del <- c("date", "cum_confirm", "county", "state")
  rg.dead <- ranger(cum_dead ~ ., dat_train[, !names(dat_) %in% dead_del])
  
  # ------------------- confirm
  confirm_del <- c("date", "cum_dead", "county", "state")
  rg.confirm <- ranger(cum_confirm ~ ., dat_train[, !names(dat_) %in% confirm_del])
  
  pred_ <- list()
  for (i in 1:days) {
    dat_test <- dat_[dat_$date == date_all[(length(date_all) - days + i)], 
                     !names(dat_) %in% "date"]
    if (i > 1) {
      k <- 7
      for (j in (i - 1):1) {
        dat_test[, names(dat_test) == paste0("dead_", k)] <- pred_[[j]]$pred_dead
        dat_test[, names(dat_test) == paste0("confirm_", k)] <- pred_[[j]]$pred_confirm
        k <- k - 1
        if (k == 0) break
      }
    }
    
    pred_confirm <- predict(rg.confirm, data = dat_test)
    pred_dead <- predict(rg.dead, data = dat_test)
    dat_test$pred_confirm <- round(pred_confirm$predictions)
    dat_test$pred_dead <- round(pred_dead$predictions)
    pred_[[i]] <- dat_test
  }
  save(pred_, file = paste0("result/pred_", days, "_3+2fea_nobase.rda"))
}


# ------------------ Calculate MRE and RMSE
load("~/GitHub/COVID-COV-DL/result/pred_30_3+2fea_nobase.rda")
# load("~/GitHub/COVID-COV-DL/result/pred_7_3+2fea_nobase.rda")

##
mre_confirm <- mre_dead <- NULL
for (i in 1:length(pred_)) {
  mre_confirm_temp <- abs(pred_[[i]]$cum_confirm - pred_[[i]]$pred_confirm) / pred_[[i]]$cum_confirm
  mre_dead_temp <- abs(pred_[[i]]$cum_dead - pred_[[i]]$pred_dead) / pred_[[i]]$cum_dead
  mre_confirm <- cbind(mre_confirm, mre_confirm_temp)
  mre_dead <- cbind(mre_dead, mre_dead_temp)
}

TransData <- function(mre) {
  mre_con <- data.frame(mre)
  mre_con$county <- pred_[[1]]$county
  mre_con$state <- pred_[[1]]$state
  mre_con$mre <- apply(mre, 1, mean, na.rm = T)
  return(mre_con)
}

mre_con_confirm <- TransData(mre_confirm)
mre_con_dead <- TransData(mre_dead)

cat("MRE_cases: ", mean(mre_con_confirm$mre), "\n")
cat("MRE_deaths: ", mean(mre_con_dead$mre[mre_con_dead$mre != Inf], na.rm = T), "\n")

## Top 10
mre_con_confirm$StateCounty <- paste(mre_con_confirm$state, mre_con_confirm$county)
mre_con_dead$StateCounty <- paste(mre_con_dead$state, mre_con_dead$county)
top10_names <- c("California Orange", "Texas Dallas", "California Riverside",
                 "California San Bernardino", "Texas Harris", "Florida Miami-Dade",
                 "Illinois Cook", "Arizona Maricopa", "New York New York",
                 "California Los Angeles")


mre_confirm_top10 <- mre_con_confirm[mre_con_confirm$StateCounty %in% top10_names, ]
mre_dead_top10 <- mre_con_dead[mre_con_dead$StateCounty %in% top10_names, ]
cat("MRE_top10_cases: ", mean(mre_confirm_top10$mre), "\n")
cat("MRE_top10_deaths: ", mean(mre_dead_top10$mre[mre_dead_top10$mre != Inf], na.rm = T), "\n")


## Remaining
mre_confirm_ex10 <- mre_con_confirm[!mre_con_confirm$StateCounty %in% top10_names, ]
mre_dead_ex10 <- mre_con_dead[!mre_con_dead$StateCounty %in% top10_names, ]
cat("MRE_remaining_cases: ", mean(mre_confirm_ex10$mre), "\n")
cat("MRE_remaining_deaths: ", mean(mre_dead_ex10$mre[mre_dead_ex10$mre != Inf], na.rm = T), "\n")


## ----------------------------------- CV -----------------------------------

cv_counties <- fread("data/cvTestCounty.csv", data.table = F)

sel_feature <- c("% Drive Alone to Work",
                 "Average Traffic Volume per Meter of Major Roadways",
                 "Income Ratio",
                 "LON",
                 "LAT")

ind1 <- names(dat_cov) %in% sel_feature
ind1[c(1:17, 65:66)] <- TRUE
dat_ <- dat_cov[, ind1]
names(dat_) <- gsub("% ", "", names(dat_))
names(dat_) <- gsub(" ", "_", names(dat_))

dat_$StateCounty <- paste(dat_$state, dat_$county)
dat_test_all <- data.frame()

for (k in 1:10) {
  dat_test_cv <- dat_[dat_$StateCounty %in% cv_counties[k, -1], ]
  dat_train_cv <- dat_[!dat_$StateCounty %in% cv_counties[k, -1], ]
  
  # ------------------- confirm
  confirm_del <- c("date", "cum_dead", "county", "state", "StateCounty")
  rg.confirm <- ranger(cum_confirm ~ ., dat_train_cv[, !names(dat_) %in% confirm_del])
  
  # ------------------- dead
  dead_del <- c("date", "cum_confirm", "county", "state", "StateCounty")
  rg.dead <- ranger(cum_dead ~ ., dat_train_cv[, !names(dat_) %in% dead_del])
  
  pred_confirm <- predict(rg.confirm, data = dat_test_cv)
  pred_dead <- predict(rg.dead, data = dat_test_cv)
  
  dat_test_cv$pred_confirm <- pred_confirm$predictions
  dat_test_cv$pred_dead <- pred_dead$predictions
  dat_test_all <- rbind(dat_test_all, dat_test_cv)
  
  cat("-------------- CV: ", k, "----------------\n\n\n")
}

save(dat_test_all, file = "result/pred_cv_test.rda")


# ------------------ Calculate MRE
load("~/GitHub/COVID-COV-DL/result/pred_cv_test.rda")

dat_test_all$date <- as.Date(dat_test_all$date)
dates <- sort(unique(dat_test_all$date))
dat_test_all <- dat_test_all[dat_test_all$date > dates[length(dates) - 30], ]

mre_confirm_all <- abs(dat_test_all$pred_confirm - dat_test_all$cum_confirm) / dat_test_all$cum_confirm
mre_dead_all <- abs(dat_test_all$pred_dead - dat_test_all$cum_dead) / dat_test_all$cum_dead
cat("MRE_CV_cases: ", mean(mre_confirm_all[mre_confirm_all != Inf], na.rm = T), "\n")
cat("MRE_CV_deaths: ", mean(mre_dead_all[mre_dead_all != Inf], na.rm = T), "\n")



