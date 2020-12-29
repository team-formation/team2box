import sys 
import tensorflow as tf
#change based on knrm location
sys.path.append('team2box/aaaidata/knrm/model/')
import model_knrm
import MYMODEL
import re,string 
#import nltk
#from nltk.corpus import stopwords
import numpy as np
import os

class myknrm:
  def __init__(self): 
     #android: vocab size=10974 train=794 test=88 validation=37 qmaxlen=50 answermaxlen=100
     data=["android","dba","physics","mathoverflow","history"] 
     Qmaxlen=[20,20,20,20,20]
     Dmaxlen=[100,100,100,100,100]
     Vocabsize=[10974,25182,31616,58321,34869]
     Trainsize=[794,2616,4421,9116,1528]
     Testsize=[88,290,491,1012,169]
     
     data_ind=4
     #step 1: train
     #self.qmaxlen=Qmaxlen[data_ind]
     #self.dmaxlen=Dmaxlen[data_ind]
     #self.vocabsize= Vocabsize[data_ind]
     #for j in range(1):
     #     self.train(data[data_ind], Trainsize[data_ind],j+1)
     
     #step 2: test
     self.path="/home/etemadir/QA/team2box/aaaidata/"+data[data_ind]    
     self.vocabsize= Vocabsize[data_ind]
     self.testsize=Testsize[data_ind]  
     self.test2(50)
  
  def train (self,data,train_size,i):
      outdir=data+"/knrmformat/results/"
      if not os.path.exists(outdir):            
            os.mkdir(outdir) 
      outdir=data+"/knrmformat/results/model"+str(i)+"/"
      if not os.path.exists(outdir):            
            os.mkdir(outdir)       
      ob=model_knrm.Knrm(self.qmaxlen,self.dmaxlen,self.vocabsize)      
      ob.train(train_pair_file_path=data+"/knrmformat/train.txt", val_pair_file_path=data+"/knrmformat/validation.txt", train_size=train_size, checkpoint_dir=outdir)
      
  def test2(self,  TopK):
       ob=model_knrm.Knrm(20,100,self.vocabsize)
       testfile=open(self.path+"/team2box/testquestions.txt",encoding="utf-8")
       #results=open(self.path+"team2boxtestsets/results.txt","w", encoding="utf-8")
       path=self.path+"/team2box/results.txt"
       allquerycode=[]
       qcode=testfile.readline().strip()
       while qcode:
          #print(qcode) 
          q=qcode.split(",")
          querycode=q[0].strip()
          allquerycode.append(querycode)
          qcode=testfile.readline().strip()
       testfile.close()  
       print(allquerycode) 
       allscoresid, allscores=ob.test2(test_point_file_path=self.path+"/team2box/allquestions.txt"
                                    , test_size=self.testsize
                                    , output_file_path=self.path+"/team2box/output.txt"
                                    ,load_model=True
                                    ,checkpoint_dir=self.path+"/knrmformat/results/model10/"
                                    , AllQuery=allquerycode, topk=TopK, resultpath=path)
              
          
       #for zz in range(len(allscoresid)):
        #     scoresid= allscoresid[zz]
       #      scores=allscores[zz]
       #      for i in range(TopK):
       #           results.write(str(scoresid[i])+" "+str(scores[i])+" ")
       #      results.write("\n")
        #     results.flush()
          
       
       #results.close()
     
  def test3(self,  TopK):
       ob=MYMODEL.Knrm(20,100,self.vocabsize)
       testfile=open(self.path+"team2boxtestsets/testquestions.txt",encoding="utf-8")
       results=open(self.path+"team2boxtestsets/oursigirresults.txt","w", encoding="utf-8")
       
       path=self.path+"team2boxtestsets/results.txt"
       allquerycode=[]
       qcode=testfile.readline().strip()
       while qcode:
          #print(qcode) 
          q=qcode.split(",")
          querycode=q[0].strip()
          allquerycode.append(querycode)
          qcode=testfile.readline().strip()
       testfile.close()  
       print(allquerycode)
       
       scoresid, scores=ob.test2(test_point_file_path=self.path+"allposts.txt"
                                    , test_size=self.testsize
                                    , output_file_path=self.path+"team2boxtestsets/output.txt"
                                    ,load_model=True
                                    ,checkpoint_dir=self.path+"mymodelformat2/results/model6/"
                                    , AllQuery=allquerycode, topk=TopK,resultpath=path)
              
          #for i in range(TopK):
           #    results.write(str(scoresid[i])+" "+str(scores[i])+" ")
          #results.write("\n")
          #results.flush()
          #qcode=testfile.readline().strip()
       #testfile.close()
       #results.close()  

ob=myknrm()  
