rm(list = ls())

setwd("C:/Users/ssr/Desktop/Discount_tire/Blackswan_Negative_Syn")

Master_data=read.csv("invoice_daily_actuals.csv")

Data=cbind(Master_data$store_id,Master_data$store_code,Master_data$effective_date,Master_data$metric_id,Master_data$actual)

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
  
  e=sample(seq(.1,0.5,0.01),1)
  e_vals[k]=e
  
  if(k==length(u_stores)){store1=data.frame(Data[ind_stores[k]:(ind_stores[k+1]),])}
  else{store1=data.frame(Data[ind_stores[k]:(ind_stores[k+1]-1),])}

  if(length(store1[,1])<=10)
  {
    New_date=rbind(New_date,store1)
    next
  }
  
  store=store1[order(store1$X3),]
  
  inv=as.integer(store[,5])
  
  t=1:length(inv)
  n=length(t)
  
  St=0
  ind[k]=sample(5:(.5*n),1)
  
  t0=1
  t1=ind[k]-1
  d_tau=round((t1-t0)/3)
  T_D[k]=d_tau
  St[t0:t1]=1-(1-e)*((1-exp((t[t0:t1]-t0)/d_tau))/(1-exp((t1-t0)/d_tau)))
  
  t2=t1+1
  ext[k]=min(sample(1:24,1),sample(1:(.4*n),1))
  t3=t2+ext[k]
  
  St[t2:t3]=St[t1]
  
  t0=t3+1
  t1=n
  r_tau=round((t1-t0)/3)
  T_R[k]=r_tau
  St[t0:t1]=e+(1-e)*(1-exp((t0-t[t0:t1])/r_tau))/(1-exp((t0-t1)/r_tau))
  
  V_pt=round(St*inv)

  New_date=rbind(New_date,cbind(store[,1],store[,2],store[,3],store[,4],V_pt))
}
New_date=New_date[-1,]

write.csv(New_date,"Invoice_syn5.csv",row.names = F)
