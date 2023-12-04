cd("C:\\Users\\aledo\\Documents\\Uni\\Semester 9\\STAT-665\\Categorical-Final-Project\\FinalProj")
using Pkg; Pkg.activate(".")
using CSV, DataFrames, TidierData, StatsPlots, GLM, StatsBase, RCall
# Data Source
# https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
raw_dt = CSV.read("..//healthcare-dataset-stroke-data.csv",DataFrame)


toFloat(x) = parse(Float64, x)

 # There are ~200 instances of N/A bmi
 # All these obs were omitted giving us a total of 4909 obs
 dt = @chain raw_dt begin
    @filter( bmi ∉ [ raw"N/A" ] )
    @mutate( bmi = toFloat(bmi) )
    @select( -id)
end

@chain dt begin
    @group_by(gender)
    @summarize(n = n(),
               eldest=maximum(age),
               youngest = minimum(age),
               mean=mean(skipmissing(age)),
               median=median(skipmissing(age)))
end

@chain dt begin
    @group_by(gender)
    @summarize(n = n(),

               heart_disease=count(==(1), skipmissing(heart_disease)),
               hypertension=count(==(1), skipmissing(hypertension)))
end

hist(dt.age)

dt = @chain raw_dt begin
    @filter( bmi ∉ [ raw"N/A" ] )
    @mutate( bmi = log(toFloat(bmi)), avg_glucose_level=log(avg_glucose_level) )
    @select( bmi, avg_glucose_level, stroke)
end

describe(dt)

theme(:juno)
p1 = @df dt boxplot(:bmi,
             color=[:darkblue],
             legend=false,
             ylab = "Log Body Mass Index",
             xlims=(-0,2),
             title="Distribution of Log BMI Measurements")

p2 = @df dt boxplot(:avg_glucose_level,
             color=[:darkblue],
             legend=false,
             title="Distribution of Log Average Glucose Levels",
             ylab = "Log Glucose Levels",
             xlims=(-0,2),
             )
savefig(p1, "logbmi.png")
savefig(p2, "logglucose.png")

R"""
 library("robustbase")
 library("ROSE")
 """


@rput dt

R""" 
samp = function (df, n) { 
    return( dt[sample(nrow(df),n),] ) 
 }
"""

 #### N = 4908
R"""
 mle= glm(stroke ~ bmi + avg_glucose_level, dt, family="binomial")
 summary(mle)

 mleR= glmrob(stroke ~ bmi + avg_glucose_level, dt, family="binomial", method="BY")

 #calculate p-value of overall Chi-Square statistic
 1 - pchisq(mleR$null.deviance - mleR$deviance, mleR$df.null-mleR$df.residual)
 summary(mleR)

set.seed(3)
sdt = samp(dt, 40)
smle  = glm(stroke ~ bmi + avg_glucose_level, sdt, family="binomial")
smleR = glmrob(stroke ~ bmi + avg_glucose_level, sdt, family="binomial", method="BY")
"""


 ### Sample from the data and extract the coefficient estimates
 R"""
 
cofs <- function (dt) {
    set.seed(1)

    models =  data.frame( "(Intercept)"= NA, "bmi"=NA, "avg_glucose_level"= NA, N = 0)
    rmodels =  data.frame( "(Intercept)"= NA, "bmi"=NA, "avg_glucose_level"= NA, N = 0)

    rang = seq(50, 1000, by = 10)
    i=1
    for (n in rang) {
        smalldf = samp(dt, n)

        ml = glm(stroke ~ bmi + avg_glucose_level,smalldf, family=binomial("logit") )
        models[i, 1:4] <- c( summary(ml)$coef[,"Estimate"],  n )

        rb = glmrob(stroke ~ bmi + avg_glucose_level, smalldf, family="binomial", method="BY")
        rmodels[i, 1:4] <- c( summary(rb)$coef[,"Estimate"], n )
        i = i+1
    }
    return( list(models, rmodels) )
}

res = cofs(dt)
models = res[[1]]
rmodels = res[[2]]
# 0.0425 is the prop of strokes in the dataset so we will try to get such a proportion in smaller subsamples
"""

@rget models
@rget rmodels

# PLOT OF COEFFICIENT ESTIMATES BY METHOD AND SAMPLE SIZE
@df models plot(:N, :X_Intercept_,
  lab="MLE",
  ylab="Value",
   ylim=(-40,10))
@df rmodels plot!(:N, :X_Intercept_, lab="Robust",xlab="Sample Size")
p1 = hline!([ -11.53 ], lab="Complete Estimate= -11.53")
savefig("inter.png")

@df models plot(:N, :bmi,
  lab="MLE",
  ylab="Value",
   ylim=(-2.5,5))
@df rmodels plot!(:N, :bmi, lab="Robust",xlab="Sample Size")
p2 = hline!([ 0.623 ], lab="Complete Estimate=0.623")
savefig("bmie.png")

@df models plot(:N, :avg_glucose_level,
  lab="MLE",
  ylab="Value",
   ylim=(-5,10))
@df rmodels plot!(:N, :avg_glucose_level, lab="Robust",xlab="Sample Size")
p3 = hline!([ 1.35 ], lab="Complete Estimate=1.35")
savefig("gluce.png")
plot([p1 p2 p3]..., layout=(3,1), legend=false, size=(720,800) )
savefig("Estimates.png")

R"""
#mutator function
modifyOutlier <- function(dt, s) {
  dt[c(nrow(dt)),] <- c(s,s,0)
  return(dt)
}

#dt[nrow(dt)+1,] <- c(0,0,0)
"""

#### TEST STABILITY OF THE P-VALUE
R"""
stability <- function (dt, N) {
    set.seed(35)
    smalldf = samp(dt, N)
    smalldf[nrow(smalldf)+1,] <- c(0,0,0)



    models =  data.frame( "(Intercept)"= NA, "bmi"=NA, "avg_glucose_level"= NA, s = 0)
    rmodels =  data.frame( "(Intercept)"= NA, "bmi"=NA, "avg_glucose_level"= NA, s = 0)

    rang = seq(0, 10, by = 0.1)
    i=1
    for (s in rang) {
        smalldf = modifyOutlier(smalldf, s)
        
        ml = glm(stroke ~ bmi + avg_glucose_level, smalldf, family=binomial("logit"), )
        models[i, 1:4] <- c( summary(ml)$coef[, "Pr(>|z|)"],s )

        rb = glmrob(stroke ~ bmi + avg_glucose_level, smalldf, family="binomial", method="BY")
        rmodels[i, 1:4] <- c( summary(rb)$coef[, "Pr(>|z|)"],s )
        i = i+1
    }
    return( list(models, rmodels) )
}
N=250
res = stability(dt,N)
models = res[[1]]
rmodels = res[[2]]
"""

@rget models
@rget rmodels

@df models plot(:s, :X_Intercept_, title="P-Value of Intercept (N=250)", lab="MLE", ylab="P-value")
p1 = @df rmodels plot!(:s, :X_Intercept_, lab="Robust",xlab="Outlier Value")

@df models plot(:s, :bmi, title="P-Value of BMI (N=250)", lab="MLE",ylab="P-value")
p2 = @df rmodels plot!(:s, :bmi, lab="Robust",xlab="Outlier Value")

@df models plot(:s, :avg_glucose_level, title="P-Value of Glucose (N=250)", lab="MLE",ylab="P-value")
p3 = @df rmodels plot!(:s, :avg_glucose_level, lab="Robust",xlab="Outlier Value")

plot(p1,p2,p3, legend=false, layout=(3,1), size=(720,800), )
savefig("stability250.png")
# stability at N=40 is done by swapping out values
# ROSE synthetic sampling
@rput dt
R"""
rdt = ROSE(stroke ~., dt, 2000)$data
table(rdt$stroke)
"""
R"""
res = stability(rdt,40)
models = res[[1]]
rmodels = res[[2]]
"""

@rget models
@rget rmodels

p1 = @df models plot(:s, :X_Intercept_, title="P-Value of Intercept (N=40)", lab="MLE", ylab="P-value")
@df rmodels plot!(:s, :X_Intercept_, lab="Robust",xlab="Outlier Value")

p2 = @df models plot(:s, :bmi, title="P-Value of BMI (N=40)", lab="MLE",ylab="P-value")
@df rmodels plot!(:s, :bmi, lab="Robust",xlab="Outlier Value")

p3 = @df models plot(:s, :avg_glucose_level, title="P-Value of Glucose (N=40)", lab="MLE",ylab="P-value")
@df rmodels plot!(:s, :avg_glucose_level, lab="Robust",xlab="Outlier Value")

plot(p1, p2, p3, layout=(3,1), size=(800,720),legend=false)
savefig("rstab.png")
# N = 40 and N = 250 were used 

### POWER ANALYSIS
R"""
full_m = glmrob(stroke ~ bmi + avg_glucose_level, data=dt, family="binomial", method="BY")
logodds = predict(full_m, dt[,c(1,2)], type="link")
set.seed(123)
p <- 1 / (1 + exp(-logodds))
simulated_strokes <- rbinom(n = length(p), size = 1, prob = p)
"""
@rget simulated_strokes

R"""
actual_outcomes <- dt$stroke
conf_matrix <- confusionMatrix(factor(simulated_strokes), factor(actual_outcomes))
"""

@rget predicted_probabilities
R"""
BMI <- c()
BMIrob <- c()
GLUrob <- c()
GLU <- c()
# Extract coefficients
intercept <- coef(full_m)[1]
effect_size_bmi <- coef(full_m)["bmi"]
effect_size_glucose <- coef(full_m)["avg_glucose_level"]

set.seed(1)
# Parameters for the simulation
num_simulations <- 1000  # Number of simulated datasets
# sample_size <- 500  # Number of observations in each dataset
j = 1
for (sample_size in seq(50, 1500, by=25) ) { 
# Initialize counters for significant results
significant_bmirob <- 0
significant_glucoserob <- 0
significant_bmi <-0
significant_glucose <-0

# Simulation loop
for (i in 1:num_simulations) {
    # Simulate data
    simulated_bmi <- (rnorm(sample_size, mean = mean(dt$bmi), sd = sd(dt$bmi)))
    simulated_glucose <- (rnorm(sample_size, mean = mean(dt$avg_glucose_level), sd = sd(dt$avg_glucose_level)))
  
    # Simulate stroke outcomes (using the logistic model)
    log_odds <- intercept + effect_size_bmi * simulated_bmi + effect_size_glucose * simulated_glucose
    prob_stroke <- 1 / (1 + exp(-log_odds))
    simulated_stroke <- rbinom(sample_size, 1, prob_stroke)
  
    # Run logistic regression on the simulated data
    simulated_data <- data.frame(stroke = simulated_stroke, bmi = simulated_bmi, avg_glucose_level = simulated_glucose)
    modelrob <- tryCatch( glmrob(stroke ~ bmi + avg_glucose_level, data = simulated_data, family = binomial(), method="BY"), error=function(e) e, warning=function(w) w)
    if(is(modelrob,"warning")) next
    model <- glm(stroke ~ bmi + avg_glucose_level, data = simulated_data, family = binomial())

    # Check for statistical significance
    if (summary(modelrob)$coefficients["bmi", "Pr(>|z|)"] < 0.05) {
      significant_bmirob <- significant_bmirob + 1
    }
    if (summary(modelrob)$coefficients["avg_glucose_level", "Pr(>|z|)"] < 0.05) {
      significant_glucoserob <- significant_glucoserob + 1
    }
    if (summary(model)$coefficients["bmi", "Pr(>|z|)"] < 0.05) {
        significant_bmi <- significant_bmi + 1
    }
    if (summary(model)$coefficients["avg_glucose_level", "Pr(>|z|)"] < 0.05) {
        significant_glucose <- significant_glucose + 1
    }

  }
print(sample_size)

# Calculate power
BMI[j] <- significant_bmi / num_simulations
GLU[j] <- significant_glucose / num_simulations
BMIrob[j] <- significant_bmirob / num_simulations
GLUrob[j] <- significant_glucoserob / num_simulations
j = j + 1
#cat("(Classic) Power for BMI:", power_bmi, "\n")
#cat("(Classic) Power for Average Glucose Level:", power_glucose, "\n")
#cat("(Robust)  Power for BMI:", power_bmirob, "\n")
#cat("(Robust)  Power for Average Glucose Level:", power_glucoserob, "\n")

}
"""

@rget BMI
@rget BMIrob
@rget GLU
@rget GLUrob


theme(:juno)
plot(50:25:1500, BMI, title="Power (BMI) vs Sample Size",lab="MLE")
p1 = plot!(50:25:1500, BMIrob, lab="Robust")

plot(50:25:1500, GLU, legend=false, title="Power (Glucose) vs Sample Size")
p2 = plot!(50:25:1500, GLUrob)

plot(p1,p2, layout=(2,1), xlab="N", ylab="Power", size=(800,720))
savefig("power.png")