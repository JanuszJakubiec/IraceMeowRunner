#packages
install.packages("iraceplot")# may need some other libraries (curl, systemfonts), and those need system 



load("../irace/irace.Rdata")#to test if the PATH is correct

library(iraceplot)

# Get number of iterations
iters <- unique(iraceResults$experimentLog[, "iteration"])
# Get number of experiments (runs of target-runner) up to each iteration
fes <- cumsum(table(iraceResults$experimentLog[,"iteration"]))
# Get the mean value of all experiments executed up to each iteration
# for the best configuration of that iteration.
elites <- as.character(iraceResults$iterationElites)
values <- colMeans(iraceResults$experiments[, elites])#if with tests, add $tests before experiments
stderr <- function(x) sqrt(var(x)/length(x))
err <- apply(iraceResults$experiments[, elites], 2, stderr) #if with tests, add $tests before experiments
plot(fes, values, type = "s",
xlab = "Number of runs of the target algorithm",
ylab = "Mean value over testing set") #ylim=c(23000000,23500000) for set y values
points(fes, values, pch=19)
#arrows(fes, values - err, fes, values + err, length=0.05, angle=90, code=3)
text(fes, values, elites, pos = 1)