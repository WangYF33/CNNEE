{
  argv <- commandArgs(TRUE)
  inputfile <- argv[1]
  tmpfile <- argv[2]
  outfile <- argv[3]
  RPMfile <- argv[4]
  eRNAfile <-argv[5]
}
#setwd("E:/1A-课题相关/1A结果整理/eRNA/RPM")
eRNA_count<-read.table(inputfile,sep="\t",header=T,check.names = F)
RPM<-data.frame()
for(i in 1:nrow(eRNA_count)){
  temp<-eRNA_count[i,3:ncol(eRNA_count)]*1000000/eRNA_count[i,2]
  RPM<-rbind(RPM,temp) 
}
write.table(RPM,file=tmpfile,sep="\t",row.names=FALSE,quote = FALSE)
RPM<-read.table(tmpfile,sep="\t",header=T,check.names = F)
rowname <- eRNA_count[,1]
rowname<-read.table(text=rowname)
RPM <- cbind(rowname,RPM)
colnames(RPM)[1] <- "eRNAid"
RPM_row2col <- as.data.frame(t(RPM))
write.table(RPM_row2col,file = outfile,sep = "\t",row.names=TRUE,col.names = FALSE,quote = FALSE)
data <- read.table(outfile,sep="\t",header = T)
RPM_mean <- rowMeans(data[,2:ncol(data)])
datamean <- data.frame(data[,1],RPM_mean,data[,2:ncol(data)])
colnames(datamean)[1] <- "eRNAid"
active_eRNA =subset(datamean, datamean$RPM_mean >= 1)
sort_eRNA <- active_eRNA[order(active_eRNA$RPM_mean,decreasing = T),]
write.table(sort_eRNA,file = RPMfile,sep = "\t",row.names=FALSE,quote = FALSE)
write.table(sort_eRNA[,1:2],file = eRNAfile,sep = "\t",row.names=FALSE,quote = FALSE)