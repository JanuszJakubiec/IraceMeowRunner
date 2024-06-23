#final solution
library(iraceplot)


args <- commandArgs(trailingOnly = TRUE)
directory <- args[1]

if(is.null(directory)){
  print("Missing directory path argument, assuming default \"./\"")
  directory = "./"
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