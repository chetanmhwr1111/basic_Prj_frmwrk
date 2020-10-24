GrAd=read.csv(file.choose())
head(GrAd)
colnames(GrAd)=c('SNo.','GS','TS','UR','SOP','LOR','GPA','Res','COA')
tail(GrAd)

#1.1
o=ggplot(data=GrAd, aes(x=GS,y=TS,size= COA,colour=COA))
o+geom_point()+
xlab('GRE Scores')+ylab('TOEFL Scores')+ ggtitle('GRE & TOEFL Score vs Admission Probability')+
labs(size='Admission Probability', colour='Admission Probability')+
theme(  axis.title.x =element_text(colour = 'red',size = 10),
        axis.title.y =element_text(colour = 'red',size = 10),
        legend.title = element_text(size = 10,colour = 'magenta'),
        plot.title = element_text(size = 25,colour ='magenta') )

#1.2
p=ggplot(data=GrAd, aes(x=GS,y=TS,size=as.factor(Res)))
p+geom_point(colour='black')+facet_grid(.~Res, scales='free')+
xlab('GRE Scores')+ylab('TOEFL Scores')+ ggtitle('GRE & TOEFL Score vs Research Experience')+
scale_size_discrete(name='Research Experience',labels=c('0','1'))+
         theme( axis.title.x =element_text(colour = 'red',size = 10),
                axis.title.y =element_text(colour = 'red',size = 10),
                legend.title = element_text(size = 10,colour = 'magenta'),
                plot.title = element_text(size = 25,colour ='magenta') )

#1.3
r=ggplot(data=GrAd, aes(x=UR,y=COA))  
r+geom_boxplot(aes(group=UR,fill=as.factor(UR)))+
xlab('University Rating')+ylab('Admission Probability')+ ggtitle('University Rating vs Admission Probability')+
scale_fill_manual(name='University Rating',values=c('1','2','3','4','5'))+
        theme(  axis.title.x =element_text(colour = 'red',size = 10),
                axis.title.y =element_text(colour = 'red',size = 10),
                legend.title = element_text(size = 10,colour = 'magenta'),
                plot.title = element_text(size = 25,colour ='magenta') )

#1.4
GS=GrAd$GS[GrAd$COA==0.9]
TS=GrAd$TS[GrAd$COA==0.9]
SNo=GrAd$SNo.[GrAd$COA==0.9]
COA=GrAd$COA[GrAd$COA==0.9]
Scores=cbind(SNo,GS,TS,COA)
Scores

#1.5
q=ggplot(data=GrAd, aes(x=SOP,y=LOR,size=as.factor(UR),colour=as.factor(UR)))
q+geom_point()+
xlab('SOP')+ylab('LOR')+ ggtitle('SOP and LOR vs University Rating')+
scale_size_discrete(name='University Rating',labels=c('1','2','3','4','5'))+
scale_colour_manual(name='University Rating',values=c('1','2','3','4','5'))+
        theme(  axis.title.x =element_text(colour = 'red',size = 10),
                axis.title.y =element_text(colour = 'red',size = 10),
                legend.title = element_text(size = 10,colour = 'magenta'),
                plot.title = element_text(size = 25,colour ='magenta') )

#1.6
s=ggplot(data=GrAd, aes(x=GS,y=TS,size=COA,colour=GPA))
s+geom_point()+
xlab('GRE Scores')+ylab('TOEFL Scores')+ ggtitle('GRE, TOEFL and Undergrad GPA Score vs Admission Probability')+
labs(size='Admission Probability',colour='Undergrads GPA')+
        theme(  axis.title.x =element_text(colour = 'red',size = 10),
                axis.title.y =element_text(colour = 'red',size = 10),
                legend.title = element_text(size = 10,colour = 'magenta'),
                plot.title = element_text(size = 20,colour ='magenta') )

#1.7
v=ggplot(data=GrAd, aes(x=GS,y=TS,size= COA,colour=GPA))
v+geom_point()+facet_grid(Res~UR , scales='free')+
xlab('GRE Scores')+ylab('TOEFL Scores')+ ggtitle('Graduate Admission Analysis')+
labs(size='Admission Probability',colour='Undergrads GPA')+
        theme(  axis.title.x =element_text(colour = 'red',size = 10),
                axis.title.y =element_text(colour = 'red',size = 10),
                legend.title = element_text(size = 10,colour = 'magenta'),
                plot.title = element_text(size = 25,colour ='magenta') )


                 