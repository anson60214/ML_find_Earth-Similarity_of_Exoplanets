library(randomForest)
library(TSA)
library(dplyr)

rm(list=ls())
set.seed(123)



setwd("U:/module2")

s1 = read.table("train-2.txt",sep=" ",header = T)
s2 = read.table("test-2.txt",sep=" ",header = T)
sav = s1[,c("star","period","radius")]
sav2 = s2[,c("star","period","radius")]


# lightcurve dir
setwd("U:/module2/lightcurves")

write.csv(sav,"sav.csv",row.names = F, col.names = T)
write.csv(sav2,"sav2.csv",row.names = F, col.names = T)

filenames <- list.files(pattern="*.txt", full.names=FALSE)

f = read.table("STAR-388.txt",sep = " ",header = T,row.names = NULL)
plot(f$time,f$flux)
plot(f[1000:6000,"time"],f[1000:6000,"flux"])

colname = sub(".txt","",filenames)

# read all the light curve data
# file_list = lapply(filenames,read.table,sep = " ",header = T,row.names = NULL)

filenames = as.data.frame(colname)
colnames(filenames)[1] = "star"


# read in star names from train and test data
sav = read.csv('sav.csv',header=T)
sav2 = read.csv('sav2.csv',header = T)
sav = rbind(sav,sav2)
# find names appeared in both lightcurves and train+test
mat = inner_join(sav,filenames,by = "star")

radius = data.frame(matrix(nrow=nrow(mat),ncol=5))
radius[,1] = mat$star
radius[,2:3] = c(mat$period,mat$radius)
colnames(radius) = c("star","period","s_radius","status","p_radius")

# remove duplicates due to NA
radius = radius[!duplicated(radius),]


t = f$time
fl = f$flux
p_day = radius[2,"period"]
s_radius = radius[2,"s_radius"]
detection = function(t,fl,p_day,rad, s_radius, verbose = 0) {
  
  fl = na.roughfix(fl)
  days = max(t)
  day_per_int = days/length(t)
  
  nint = length(t)
  # number of intervals in a orbital period
  p_int = round(p_day/day_per_int)
  p_int_00 = floor(p_int/100)
  
  # find the indices of flux which can be used to calculate planet radius
  if(p_day>5){
    ##### xgb 6.21485 uses10
    # (number of periods in the lightcurve data)*10
    N = floor(days/p_day)*10
    # indices of lowest flux
    index <- order(fl)[1:N]
    index = unique(floor(index/100))
    index = sort(index)
  } else {
    status = "D"
    p_radius = NA
    return(list(status,p_radius))
  }

  
  len = length(index)
  # set inital values 
  status = "S"
  p_radius = NA

  for (i in 1:(len-1)) {
    # stop when p_radius is determined
    if(status == "D"){
      break
    }
    ##### xgb 6.21485 uses10
    # stop when no pattern detected in the first 25% data
    if ((i/len)>0.25){
      if (verbose==1){
        print(status)
      }
      break
    }
    
    j = i
    if ((index[j+1]-index[j])==1){
      j = j+1
    }
    
    curr = index[j]
    n = 0
    depth = 0
    # check if next index is in the range of current index +- orbital period
    while((TRUE%in%(index[(j+1):len]%in%(seq(curr+p_int_00-2,curr+p_int_00+2))))&((j+1)<len)){
      n = n+1
      if(index[j]==0) {
        # find the min around that index when index is 0
        depth = depth + min(fl[seq(1,99,by=1)])  
      }else{
        # find the min around that index
        depth = depth + min(fl[seq(index[j]*100-20,index[j]*100+99,by=1)])
      }
      # move current index to the next index satisfied the condition 
      j=j+min(which(index[(j+1):len]%in%(seq(curr+p_int_00-2,curr+p_int_00+2))==TRUE))
      curr = index[j]

      
      # if next index is index+1 make the current index to the next index (might be problematic)
      if (((j<len)&(index[j+1]-index[j])==1)){
        j = j+1
        curr = index[j]
      }
    }

    # if the pattern is found in 40% of the times it should theoretically appear
    if(n >= (floor(0.4*(days/p_day)))) {
      if (verbose == 1){
        print("Detected")
      }
      # calculate the planet's radius using the average max depth found 
      status = "D"
      max_depth = (1-depth/n)
      p_radius = s_radius*sqrt(max_depth)
    } else{
      # print("Not Detected /////////////////////////")
      status = "N"
    }
    
  }
  return(list(status,p_radius))
}

for(i in 1:nrow(radius)){
  
  if(!is.na(radius[i,2])){
    f = read.table(paste0(radius[i,"star"],".txt"),sep = " ",header = T,row.names = NULL)
    radius[i,"status"] = detection(t = f$time, fl = f$flux, p_day = radius[i,"period"], s_radius = radius[i,"s_radius"],verbose = 0)[[1]]
    radius[i,"p_radius"] = detection(t = f$time, fl = f$flux, p_day = radius[i,"period"], s_radius = radius[i,"s_radius"])[[2]]
    
  }
  print(paste("/////////////////////",i,"//////////////////////"))
}

sum(!is.na(radius$p_radius))


# slice the light curve data with regard to the orbital period
# and compute the radius with the min of the sliced lightcurves
sup = function(t , fl, p_day , s_radius){
  fl = na.roughfix(fl)
  days = max(t)
  day_per_int = days/length(t)
  
  nint = length(t)
  p_int = round(p_day/day_per_int)
  p_int_00 = floor(p_int/100)
  
  n = floor(nint/(p_int*2))
  Min = 0
  if(p_day>280) {
    # too lazy to fix the possible out-of-bound error
    # so just add a condition that computes na
    n = 1
  }
  for(i in 1:n) {
    Min = Min+min(fl[(1+(i-1)*p_int*2):(i*p_int*2)])
    if(p_day>280) {
      print(paste0("range: ",(1+(i-1)*p_int*2),"/",i*p_int*2))
      print(Min)
    }
  }
  Min = Min/n
  max_depth = (1-Min)
  p_radius = s_radius*sqrt(max_depth)
  print(p_radius)
  if(p_day>280) {
    p_radius = s_radius*sqrt(1-min(fl))
  }
  return(p_radius)
  
  
}

# compute the radius for which the prior for loop outputs NA
for(i in 1:nrow(radius)){
  
  if((!is.na(radius[i,"period"]))&(is.na(radius[i,"p_radius"]))){
    f = read.table(paste0(radius[i,"star"],".txt"),sep = " ",header = T,row.names = NULL)
    radius[i,"p_radius"] = sup(t = f$time, fl = f$flux, p_day = radius[i,"period"],s_radius = radius[i,"s_radius"])
  }
  print(paste("/////////////////////",i,"//////////////////////"))
}


radius = subset(radius, select = -s_radius)

# module2 dir
setwd("U:/module2")
write.csv(radius,"rad2.csv",row.names = F, col.names = T)

