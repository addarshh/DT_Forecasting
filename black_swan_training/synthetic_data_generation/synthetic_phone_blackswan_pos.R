rm(list = ls())

setwd("C:/Users/ssr/Desktop/Discount_tire/Blackswan_Positive_Syn")

Master_data=read.csv("phone_daily_actuals.csv")

Data=na.omit(cbind(Master_data$store_id,Master_data$store_code,Master_data$effective_date,Master_data$metric_id,Master_data$actual))

u_stores=unique(Data[,1])
ind_stores=0
for(i in 1:length(u_stores))
{
  ind_stores[i]=min(which(Data[,1]==u_stores[i]))
}
ind_stores=c(ind_stores,length(Data[,1]))

e_vals=0
T_D=0
T_R=0
ext=0
ind=0
New_date=rep(0,5)
for(k in 1:length(u_stores))
{
  print(k)
  
  if(k==length(u_stores)){store1=data.frame(Data[ind_stores[k]:(ind_stores[k+1]),])}
  else{store1=data.frame(Data[ind_stores[k]:(ind_stores[k+1]-1),])}
  
  if(length(store1[,1])<=10)
  {
    New_date=rbind(New_date,cbind(store[,1],store[,2],store[,3],store[,4],store[,5]))
    next
  }
  
  store=store1[order(store1$X3),]
  
  ph=as.integer(store[,5])
  
  t=1:length(ph)
  n=length(t)
  e=sample(seq(.01,.5,0.01),1)
  e_vals[k]=e
  
  St=0
  
  t0=1
  t1=sample((.1*n):(.7*n),1)-1
  r_tau=round((t1-t0)/3)
  T_D[k]=r_tau
  St[t0:t1]=1-e^((t0-t[t0:t1])/r_tau)/(1-e^((t0-t1)/r_tau))
  
  t2=t1+1
  ext[k]=min(sample(1:24,1),sample(1:(.4*n),1))
  t3=t2+ext[k]
  
  St[t2:t3]=St[t1]
  
  t0=t3+1
  t1=round(sample(t0:(.9*n),1))
  d_tau=round((t1-t0)/3)
  T_R[k]=d_tau
  temp=1-e+(e^((t[t0:t1]-t0)/d_tau))/(1-(e^((t1-t0)/d_tau)))
  temp[which(temp>St[t3])]=St[t3]
  St[t0:t1]=temp
  
  St[(t1):n]=St[t1-1]
  
  P_pt=round(St*ph)
  
  New_date=rbind(New_date,cbind(store[,1],store[,2],store[,3],store[,4],P_pt))
}
New_date=New_date[-1,]

write.csv(New_date,"Phone_syn5.csv",row.names = F)


