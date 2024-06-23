  #packages
  install.packages("iraceplot")# may need some other libraries (curl, systemfonts), and those need system 



  load("../irace/irace.Rdata")#to test if the PATH is correct

  library(iraceplot)

  # Get number of iterations
  iters <- unique(iraceResults$experimentLog[, "iteration"])
  # Get number of experiments (runs of target-runner) up to each iteration
  #fes <- cumsum(table(iraceResults$experimentLog[,"iteration"]))
  # Get the mean value of all experiments executed up to each iteration
  # for the best configuration of that iteration.
  configs <- iraceResults$allConfigurations

  conf_id <- iraceResults$allConfigurations$.ID.


  populations <- iraceResults$allConfigurations$populations_number

  values <- colMeans(iraceResults$experiments[,conf_id])#if with tests, add $tests before experiments
  #min(data[,2], na.rm=T)

  stderr <- function(x) sqrt(var(x)/length(x))
  err <- apply(iraceResults$experiments[, elites], 2, stderr) #if with tests, add $tests before experiments
  plot(populations, values, type = "p",
  xlab = "Number of runs of the target algorithm",
  ylab = "Mean value over testing set") #ylim=c(23000000,23500000) for set y values
  points(populations, values, pch=19)
  #rrows(fes, values - err, fes, values + err, length=0.05, angle=90, code=3)
  #text(fes, values, elites, pos = 1)



  # for loading multiple Rdata files, each will have to be in different environment, or they will 
  # override each other
  e <- new.env(parent = emptyenv())
  load(FILE_PATH, envir = e)


  #merging
  df_list <- list(e1, e2)
  Reduce(function(x, y) merge(x$iraceResults$experiments, y$iraceResults$experiments, all=TRUE), df_list)  

  #lapply

  as.character(1)

  environment_list = c()

  experiments_list = c()

  for (i in 1:n) {
    path <- paste("./irace-",as.character(i),".Rdata", sep="", collapse=NULL)
    e <- new.env(parent = emptyenv())
    load(path, envir = e)
    environment_list <- append(environment_list, e)
    experiments_list <- append(experiments_list, e$iraceResults$experiments)
  }

  exp_list = Reduce(function(x, y) merge(x[1]$iraceResults$experiments, y[1]$iraceResults$experiments, all=TRUE), environment_list)  

  exp_list = Reduce(function(x, y) rbind(x$iraceResults$experiments, y$iraceResults$experiments), environment_list)  

  do.call(rbind, )

  rbind(a,b)



  ####### below is code


  environment_list = c()

  experiments_list = matrix(0, nrow = 11, ncol = 1)
  populations_list = c()

  for (i in 1:n) {
    path <- paste("./irace-",as.character(i),".Rdata", sep="", collapse=NULL)
    e <- new.env(parent = emptyenv())
    load(path, envir = e)
    environment_list <- append(environment_list, e)
    experiments_list <- cbind(experiments_list, e$iraceResults$experiments)
    #print(e$iraceResults$allConfigurations$populations_number)
    populations_list <- append(populations_list, e$iraceResults$allConfigurations$populations_number)
  }

  values <- colMeans(experiments_list, na.rm = TRUE)

  plot(populations_list, values[-1], type = "p",
  xlab = "Number of runs of the target algorithm",
  ylab = "Mean value over testing set") #ylim=c(23000000,23500000) for set y values
  points(populations_list, values[-1], pch=19)



#final solution

args <- commandArgs(trailingOnly = TRUE)
directory <- args[1]

if(is.null(directory)){
  print("Missing directory path")
  stop()
}

environment_list = c()

experiments_list = matrix(0, nrow = 11, ncol = 1)
populations_list = c()

#directory <- "./solutions/"

filenames <- list.files(directory, pattern="*.Rdata")

for(file in filenames) {
  path <- paste(directory, file, sep="", collapse=NULL)
  e <- new.env(parent = emptyenv())
  load(path, envir = e)
  environment_list <- append(environment_list, e)
  experiments_list <- cbind(experiments_list, e$iraceResults$experiments)
  #print(e$iraceResults$allConfigurations$populations_number)
  populations_list <- append(populations_list, e$iraceResults$allConfigurations$populations_number)
}

values <- colMeans(experiments_list, na.rm = TRUE)

plot(populations_list, values[-1], type = "p",
xlab = "Number of runs of the target algorithm",
ylab = "Mean value over testing set") #ylim=c(23000000,23500000) for set y values
points(populations_list, values[-1], pch=19)