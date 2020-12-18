#TeamFormation_v2.py

import numpy as np
import random
import sys
import math
import datetime as datetime

class TeamFormation:        
    def __init__(self,path,data): 
        self.path=path
        self.dataset=data 
        pfile=open(self.path+self.dataset+"/CQAG_properties.txt")
        pfile.readline()
        properties=pfile.readline().strip().split(" ")
        pfile.close()
        self.N=int(properties[0]) # number of nodes in the CQA network graph N=|Qestions|+|Answers|+|Experts|                
        self.qnum=int(properties[1])
        self.anum=int(properties[2])
        self.enum=int(properties[3])

        self.Expert_id_map={}
        pufile=open(self.path+self.dataset+"/answer_user_ids.txt")
        pufile.readline()
        line=pufile.readline().strip()
        eids=[]
        while line:
            ids=line.split(" ")
            eid=int(ids[1].strip())            
            if eid not in eids:
                eids.append(eid)
            line=pufile.readline().strip() 
        pufile.close()
        for i in range(len(eids)):
            self.Expert_id_map[i]=eids[i]

    def CompareExpertFinding(self,teamsembeding,teamsOffsets,team2box_w1_embedding,m2v_w1_embedding,teamsize,outputfile,Numq2map,Numtopkteams): 
        """ this function forms teams for the methods and compare team2box vs expert finding methods NeRank,Sqe,Meatpath2vec
        """
        self.teamcenters=np.loadtxt(self.path+self.dataset+teamsembeding)        
        self.teamoffsets=np.loadtxt(self.path+self.dataset+teamsOffsets)       
        self.eqembedding=np.loadtxt(self.path+self.dataset+team2box_w1_embedding)  
        self.testquestions=self.loadtestresults()       

        self.numq2map=Numq2map  #set between 1 .. n         
        self.testqmap2embeddings=self.maptestqtoembeddingspace(self.eqembedding)        
        self.testqtopteams=self.findtopkteams(Numtopkteams)
        self.loadTeams()
        self.testq_teamsize=self.getteamsize()        
        self.usertags=self.loadusertags()         
        self.questiontags=self.loadquestiontags()           
        
        fout=open("detailedresults/team2box_results_"+self.dataset+"_"+str(teamsize)+".txt","w")
        self.testq_teams=self.formteam(teamsize,fout)                
        SC,EL,CL=self.displayteams(self.testq_teams,fout)
        strtable3="\\textbf{"+ str(round(SC,1))+"} & \\textbf{"+str(round(EL,1))+"} & \\textbf{"+str(round(CL,1))+"}"
        outputfile.write("\nTeam2box:  SC="+str(SC)+" EL="+str(EL)+" CL="+str(CL)+"   : \\textbf{"+ str(round(SC,1))+"} & \\textbf{"+str(round(EL,1))+"} & \\textbf{"+str(round(CL,1))+"}")   
        #print("+++++++++++++++++++++++++++++++++++++++++++++")
        fout=open("detailedresults/meatpath_results_"+self.dataset+"_"+str(teamsize)+".txt","w")
        self.m2veqembedding=np.loadtxt(self.path+self.dataset+m2v_w1_embedding) #"/krnmdata1/HopeEmbeding.txt")            
        self.testqmap2m2vembeddings=self.maptestqtoembeddingspace(self.m2veqembedding)

        #print("\nmetapath2vec results:")
        self.m2vtestq_teams=self.m2vformteam(teamsize)

        SC,EL,CL=self.displayteams(self.m2vtestq_teams,fout) 
        outputfile.write("\nmetapath:  SC="+str(SC)+" EL="+str(EL)+" CL="+str(CL)+"   : "+ str(round(SC,1))+" & "+str(round(EL,1))+" & "+str(round(CL,1)))
        strtable1=str(round(SC,1))+" & "+str(round(EL,1))+" & "+str(round(CL,1))           

        #print("+++++++++++++++++++++++++++++++++++++++++++++")            
        #print("\nNeRank results:")    

        self.nerankteams=self.NeRank_formteam(teamsize)
        fout=open("detailedresults/nerank_results_"+self.dataset+"_"+str(teamsize)+".txt","w")
        SC,EL,CL=self.displayteams(self.nerankteams,fout)
        outputfile.write("\nNeRank:  SC="+str(SC)+" EL="+str(EL)+" CL="+str(CL)+"   : "+ str(round(SC,1))+" & "+str(round(EL,1))+" & "+str(round(CL,1)))
        strtable2=str(round(SC,1))+" & "+str(round(EL,1))+" & "+str(round(CL,1))
        #outputfile.write("\n"+strtable1+"   &        "+strtable2+"    &           "+strtable3)

        self.seqpointwise_teams=self.seqpointwise_formteam(teamsize)
        fout=open("detailedresults/seq_results_"+self.dataset+"_"+str(teamsize)+".txt","w")
        SC,EL,CL=self.displayteams(self.seqpointwise_teams,fout)
        outputfile.write("\nseqpointwise:  SC="+str(SC)+" EL="+str(EL)+" CL="+str(CL)+"   : "+ str(round(SC,1))+" & "+str(round(EL,1))+" & "+str(round(CL,1)))
        strtable4=str(round(SC,1))+" & "+str(round(EL,1))+" & "+str(round(CL,1))

        outputfile.write("\n"+strtable1+"   &        "+strtable2+"    &           "+strtable4+"    &           "+strtable3)

    def CompareTeamFormation(self,teamsembeding,teamsOffsets,team2box_w1_embedding,outputfile,Numq2map,Numtopkteam):  
        """
        team2box vs classic team formation methods CC CO SA-CA-CC 
        """
        fout_t2b=open("detailedresults/team2boxVS_SA-CA-CC_results_"+self.dataset+".txt","w")  
        self.teamcenters=np.loadtxt(self.path+self.dataset+teamsembeding)        
        self.teamoffsets=np.loadtxt(self.path+self.dataset+teamsOffsets)       
        self.eqembedding=np.loadtxt(self.path+self.dataset+team2box_w1_embedding)  

        self.testquestions=self.loadtestresults()#load from CC results because they miss to add teams for few test questions
        self.usertags=self.loadusertags()               
        self.questiontags=self.loadquestiontags()        
        self.numq2map=Numq2map #set between 1 .. n          

        self.testqmap2embeddings=self.maptestqtoembeddingspace(self.eqembedding)            
        self.testqtopteams=self.findtopkteams(Numtopkteam)
        self.loadTeams()
        self.testq_teamsize=self.getteamsize()         

        self.ccteams,av_teamsize=self.loadteams("allcc")
        outputfile.write("\nCC results:")
        fout=open("detailedresults/CC_results_"+self.dataset+"_"+str(av_teamsize)+".txt","w")
        SC,EL,CL=self.displayteams_2(self.ccteams,fout)
        outputfile.write(" SC="+str(SC)+" EL="+str(EL)+" CL="+str(CL)+ "   : "+ str(round(SC,1))+" & "+str(round(EL,1))+" & "+str(round(CL,1)))
        strtable1=str(round(SC,1))+" & "+str(round(EL,1))+" & "+str(round(CL,1))
        outputfile.write("\nteam2box results:") 
        fout=open("detailedresults/team2boxvsCC_results_"+self.dataset+"_"+str(av_teamsize)+".txt","w")               
        self.testq_teams=self.formteam(av_teamsize,fout)                    
        SC,EL,CL=self.displayteams(self.testq_teams,fout)
        outputfile.write(" SC="+str(SC)+" EL="+str(EL)+" CL="+str(CL)+ "   : \\textbf{"+ str(round(SC,1))+"} & \\textbf{"+str(round(EL,1))+"} & \\textbf{"+str(round(CL,1))+"}")
        strtable2="\\textbf{"+ str(round(SC,1))+"} & \\textbf{"+str(round(EL,1))+"} & \\textbf{"+str(round(CL,1))+"}"
        outputfile.write("\n"+strtable1+"    &    "+strtable2)


        self.rcoteams,av_teamsize_Approx=self.loadteams("allrco")
        outputfile.write("\nCO results:")#RCO method
        fout=open("detailedresults/Approx_results_"+self.dataset+"_"+str(av_teamsize_Approx)+".txt","w")
        SC,EL,CL=self.displayteams_2(self.rcoteams,fout)  
        outputfile.write(" SC="+str(SC)+" EL="+str(EL)+" CL="+str(CL)+ "   : "+ str(round(SC,1))+" & "+str(round(EL,1))+" & "+str(round(CL,1)))
        outputfile.write("\nteam2box results:")
        strtable1=str(round(SC,1))+" & "+str(round(EL,1))+" & "+str(round(CL,1))
        fout=open("detailedresults/team2boxVS_Approx_results_"+self.dataset+"_"+str(av_teamsize_Approx)+".txt","w")
        self.testq_teams=self.formteam(av_teamsize_Approx,fout)                           
        SC,EL,CL=self.displayteams(self.testq_teams,fout)
        outputfile.write(" SC="+str(SC)+" EL="+str(EL)+" CL="+str(CL)+ "   : \\textbf{"+ str(round(SC,1))+"} & \\textbf{"+str(round(EL,1))+"} & \\textbf{"+str(round(CL,1))+"}")
        strtable2="\\textbf{"+ str(round(SC,1))+"} & \\textbf{"+str(round(EL,1))+"} & \\textbf{"+str(round(CL,1))+"}"
        outputfile.write("\n"+strtable1+"    &    "+strtable2)

        self.ercoteams,av_teamsize_SA=self.loadteams("allerco")
        outputfile.write("\nSA-CA-CC results:") # erco method
        fout=open("detailedresults/SA-CA-CC_results_"+self.dataset+"_"+str(av_teamsize_SA)+".txt","w")
        SC,EL,CL=self.displayteams_2(self.ercoteams,fout)
        outputfile.write(" SC="+str(SC)+" EL="+str(EL)+" CL="+str(CL)+ "   : "+ str(round(SC,1))+" & "+str(round(EL,1))+" & "+str(round(CL,1)))
        strtable1=str(round(SC,1))+" & "+str(round(EL,1))+" & "+str(round(CL,1))
        outputfile.write("\nteam2box results:")       
        self.testq_teams=self.formteam(av_teamsize_SA,fout_t2b) 
        SC,EL,CL=self.displayteams(self.testq_teams,fout_t2b)
        outputfile.write(" SC="+str(SC)+" EL="+str(EL)+" CL="+str(CL)+ "   : \\textbf{"+ str(round(SC,1))+"} & \\textbf{"+str(round(EL,1))+"} & \\textbf{"+str(round(CL,1))+"}")
        strtable2="\\textbf{"+ str(round(SC,1))+"} & \\textbf{"+str(round(EL,1))+"} & \\textbf{"+str(round(CL,1))+"}"
        outputfile.write("\n"+strtable1+"    &    "+strtable2)

    def loadteams(self,name):#load CC,Approx, SA-CA-CC results
        filet=open(self.path+self.dataset+"/DBLPformat/"+name+"results.txt")        
        teams={}
        line=filet.readline().strip()
        totalsize=0.0
        while line:
            eids=line.split(" ")
            team=[]
            for eid in eids[1:]:
               team.append(int(eid)+self.qnum+self.anum)
            teams[int(eids[0])]=list(set(team)) 
            totalsize+=len(teams[int(eids[0])])  
            line=filet.readline().strip()

        av_team=int(math.ceil( totalsize/len(teams)))    
        print("average team size="+str(av_team))

        return teams,av_team

    def m2vformteam(self,teamsize):
        teams=[]
        for i in range(len(self.testqmap2m2vembeddings)):            
            test=self.testqmap2m2vembeddings[i]
            #print(test)
            c=np.square(test-self.m2veqembedding[self.qnum:self.qnum+self.enum])         
            d=np.sqrt(np.sum(c,axis=1))            
            ids=list(range(self.qnum+self.anum,self.qnum+self.anum+self.enum))          
            so,sids=(list(t) for t in zip(*sorted(zip(d.tolist(), ids),reverse=False)) )
            teams.append(sids[0:teamsize])
        return teams

    def displayteams(self,teams, outfile):
        coverness=0
        coverness_per_team_member=0
        total_scores_per_team_member=0
        total_scores_per_team_member2=0
        total_tags=0
        total_scores=0
        cover_per_q=0
        for i in range(len(self.testquestions)):
            #print(i)
            #print(teams[i])
            #print(self.testquestions[i][0])
            #print(self.questiontags[self.testquestions[i][0]]['tags'])
            qtag=set(self.questiontags[self.testquestions[i][0]]['tags'])
            total_tags+=len(qtag)
            outfile.write("\ntest q: "+ str(i)+" id in G="+str(self.testquestions[i][0])+"\n")
            outfile.write("test q tags: "+ ";".join(qtag))
            #print("\n\nquestion tags=",qtag)
            #print("teams tags:")
            inter=[]
            alltags=[]
            alltags_score=[]
            for e in teams[i]:
               utag=self.usertags[e-self.qnum-self.anum]['tags']
               inter=list(qtag.intersection(utag))
               outfile.write("\nteam member "+str(e)+" with original id "+str(self.Expert_id_map[e-self.qnum-self.anum])+" with tags:  \n")
               #alltags.extend(inter)
               for t in inter:
                 scor=self.usertags[e-self.qnum-self.anum]['scores'][self.usertags[e-self.qnum-self.anum]['tags'].index(t)] 
                 #print(t," ",scor," ")
                 outfile.write("tag: "+t+" score:"+str(scor)+";")
                 if t not in alltags:
                     alltags.append(t)
                     alltags_score.append(scor)
                 else:
                     indx=alltags.index(t) 
                     #if alltags_score[indx]<scor:
                     alltags_score[indx]+=scor   
            #alltags=list(set(alltags))
            coverness+=len(alltags)
            coverness_per_team_member+=  ((len(alltags)/len(qtag))/ len(teams[i])) 
            cover_per_q+= (len(alltags)/len(qtag))
            total_scores+=np.sum(np.array(alltags_score))

            len_alltags=len(alltags)
            if len_alltags>0:
               qsumscores=np.sum(np.array(alltags_score))/len_alltags
            else:
               qsumscores=0  
            total_scores_per_team_member+= ( qsumscores/len(teams[i]) )
            qsumscores=np.sum(np.array(alltags_score))/ len(qtag)            
            total_scores_per_team_member2+= ( qsumscores/len(teams[i]) )
            #print(alltags)    
               #print(self.usertags[e-self.qnum-self.anum]['scores'])
            #print("intersection:"+inter)   
        #print("total question tags=",total_tags)
        #print("total tags covered in teams=",coverness)
        #print("total tag scores in teams (tag_score or EL):",total_scores/coverness)
        #print("percent of tags covered in teams=",coverness/total_tags)
        #print("percent of tags covered per question (tag_cover or SC):",cover_per_q/len(self.testquestions))
        #print("average of common quetions answered by team members (past_work or CL):",self.findnumcommonquestions(teams))

        SC=(cover_per_q/len(self.testquestions))*100
        EL=(total_scores_per_team_member2/len(self.testquestions))        
        CL=self.findnumcommonquestions(teams,outfile)
        outfile.close()
        return SC,EL,CL

    def displayteams_2(self,teams,outfile):
        coverness=0
        total_tags=0
        total_scores_per_team_member=0
        coverness_per_team_member=0
        total_scores=0
        cover_per_q=0
        total_scores_per_team_member2=0
        for i in range(len(self.testquestions)):
            if i not in teams:
               continue
            #print(i)
            #print(teams[i])
            #print(self.testquestions[i][0])

            #print(self.questiontags[self.testquestions[i][0]]['tags'])
            qtag=set(self.questiontags[self.testquestions[i][0]]['tags'])
            total_tags+=len(qtag)
            outfile.write("\ntest q: "+ str(i)+"\n")
            outfile.write("test q tags: "+ ";".join(qtag))
            #print("\n\nquestion tags=",qtag)
            #print("teams tags:")
            inter=[]
            alltags=[]
            alltags_score=[]
            for e in teams[i]:
               utag=self.usertags[e-self.qnum-self.anum]['tags']
               inter=list(qtag.intersection(utag))
               outfile.write("\nteam member: "+str(e)+" with tags:  \n")
               #alltags.extend(inter)
               for t in inter:
                 scor=self.usertags[e-self.qnum-self.anum]['scores'][self.usertags[e-self.qnum-self.anum]['tags'].index(t)] 
                 #print(t," ",scor," ")
                 outfile.write("tag: "+t+" score:"+str(scor)+";")
                 if t not in alltags:
                     alltags.append(t)
                     alltags_score.append(scor)
                 else:
                     indx=alltags.index(t) 
                     #if alltags_score[indx]<scor:
                     alltags_score[indx]+=scor   
            #alltags=list(set(alltags))
            coverness+=len(alltags)
            coverness_per_team_member+=  ((len(alltags)/len(qtag))/ len(teams[i]))
            cover_per_q+= (len(alltags)/len(qtag))
            total_scores+=np.sum(np.array(alltags_score))
            len_alltags=len(alltags)
            if len_alltags>0:
               qsumscores=np.sum(np.array(alltags_score))/len_alltags
            else:
               qsumscores=0  
            total_scores_per_team_member+= ( qsumscores/len(teams[i]) )


            qsumscores=np.sum(np.array(alltags_score))/ len(qtag)            
            total_scores_per_team_member2+= ( qsumscores/len(teams[i]) )
            #print(alltags)    
               #print(self.usertags[e-self.qnum-self.anum]['scores'])
            #print("intersection:"+inter)   
        #print("total question tags=",total_tags)
        #print("total tags covered in teams=",coverness)
        #print("total tag scores in teams (tag_score):",total_scores/coverness)
        #print("percent of tags covered in teams=",coverness/total_tags)
        #print("percent of tags covered per question (tag_cover):",cover_per_q/len(self.testquestions))
        #print("average of common quetions answered by team members (past_work):",self.findnumcommonquestions_2(teams))
        #print("%.2f & %.2f & %.2f"%((cover_per_q/len(self.testquestions))*100,total_scores/coverness,self.findnumcommonquestions_2(teams)))

        SC=(cover_per_q/len(self.testquestions))*100        
        EL=(total_scores_per_team_member2/len(self.testquestions))
        CL=self.findnumcommonquestions_2(teams,outfile)
        outfile.close()
        return SC,EL,CL

    def formteam(self,teamsize1,fout):
       fout.write("\n\n\n\n\n\n\nTeam members of top k temas:")       
       teams=[]
       self.teamssizes=[]
       teamsizetotal=0
       for i in range(len(self.testquestions)):
           fout.write("\n\n****\n test q "+str(i)+" teams: {"+ " ".join([str(tt) for tt in self.testqtopteams[i]])+"}")
           qteamsize=self.testq_teamsize[i]
           experts=[]
           for t in self.testqtopteams[i]:
               experts.extend(self.Teams[t])
               fout.write("\n----------\nteam "+str(t)+" members={"+" ".join([ "id: "+str(tm)+" original id: "+str(self.Expert_id_map[tm-self.qnum-self.anum]) for tm in self.Teams[t]])+"}")
               for tm in self.Teams[t]:
                   fout.write("\nmember "+"id: "+str(tm)+" original id:"+str(self.Expert_id_map[tm-self.qnum-self.anum])+ " embeding: "+" ".join([str(em) for em in self.eqembedding[tm-self.anum] ]))

           experts=list(set(experts))
           qtag=set(self.questiontags[self.testquestions[i][0]]['tags'])
           e_score=[]
           for e in experts:
              utag=self.usertags[e-self.qnum-self.anum]['tags']
              inter=list(qtag.intersection(utag))

              sumscor=0
              for t in inter:
                sumscor+=self.usertags[e-self.qnum-self.anum]['scores'][self.usertags[e-self.qnum-self.anum]['tags'].index(t)] 

              if len(inter)>0:
                  e_score.append(sumscor/len(inter))
              else:
                  e_score.append(0)     

           sdist,sexperts=(list(t) for t in zip(*sorted(zip(e_score, experts),reverse=True)) ) 
           teams.append(sexperts[0:teamsize1])           
           self.teamssizes.append(teamsize1) 
           teamsizetotal+=qteamsize 
       #print("Average team size for similar quetions:",teamsizetotal/len(self.testquestions))
       return teams

    def NeRank_formteam(self,teamsize):
       pufile=open(self.path+self.dataset+"/answer_user_ids.txt")
       pufile.readline()
       line=pufile.readline().strip()
       eids=[]
       while line:
            ids=line.split(" ")
            eid=int(ids[1].strip())            
            if eid not in eids:
                eids.append(eid)
            line=pufile.readline().strip() 
       pufile.close()

       #load results
       data_dir=self.path+self.dataset+"/NeRankFormat/"
       fin=open(data_dir+"results.txt")

       line=fin.readline().strip()
       teams=[]
       while line:
           answerers=[]
           scores=[]
           elements=line.split(" ")[:-1]
           for el in elements:
                ids=el.split(":")
                if ids[0]=="aid":
                    answerers.append(self.anum+self.qnum+eids.index(int(ids[1])))
                elif ids[0]=="score":
                    scores.append(float(ids[1]))

           #sort answeres beased on the scores
           scores,answerers=(list(t) for t in zip(*sorted(zip(scores, answerers),reverse=True)) )
           teams.append(answerers[:teamsize])

           line=fin.readline().strip()       
       return teams

    def seqpointwise_formteam(self,teamsize):
       pufile=open(self.path+self.dataset+"/answer_user_ids.txt")
       pufile.readline()
       line=pufile.readline().strip()
       eids=[]
       while line:
            ids=line.split(" ")
            eid=int(ids[1].strip())            
            if eid not in eids:
                eids.append(eid)
            line=pufile.readline().strip() 
       pufile.close()

       #load results
       data_dir=self.path+self.dataset+"/ColdEndFormat/"
       fin=open(data_dir+"seq_pointwise_test_results.txt")

       line=fin.readline().strip()
       teams=[]
       while line:
           answerers=[]           
           elements=line.split(" ")
           for el in elements:                
               answerers.append(self.anum+self.qnum+eids.index(int(el)))
           teams.append(answerers[:teamsize])           
           line=fin.readline().strip()       
       return teams
    def loadquestiontags(self):
        qtagsfile=open(self.path+self.dataset+"/Q_tags.txt")
        #line=qtagsfile.readline()
        line=qtagsfile.readline().strip()
        qtags={}

        while line:
            token=line.split(" ")
            qid=int(token[0])           
            tags=token[1:]             
            qtags[qid]={'tags':tags}                               
            line=qtagsfile.readline().strip() 
        qtagsfile.close()
        #print(utags)
        return qtags

    def loadusertags(self):        
        pufile=open(self.path+self.dataset+"/answer_user_ids.txt")
        pufile.readline()
        line=pufile.readline().strip()
        eids=[]
        while line:
            ids=line.split(" ")
            eid=int(ids[1].strip())            
            if eid not in eids:
                eids.append(eid)
            line=pufile.readline().strip() 
        pufile.close()
        #print(len(eids))

        utagsfile=open(self.path+self.dataset+"/user_tags.txt")
        line=utagsfile.readline().strip()
        line=utagsfile.readline().strip()
        utags={}
        while line:
            token=line.split(" ")
            eid=int(token[0].strip())  
            if eid in eids:
               tags=token[1::3]
               scores=list(map(int, token[3::3]))
               enewid=eids.index(eid)
               if enewid not in utags: 
                  utags[enewid]={'tags':tags,'scores':scores}
            else:    
                print("error !!! eid is not in eids!! eid="+str(eid))          
            line=utagsfile.readline().strip() 
        utagsfile.close()
        #print(utags)
        return utags



    def getteamsize(self):
       tsize=[]
       for i in range(len(self.testquestions)):
         size=0
         teamsids=self.testqtopteams[i]
         for teamid in teamsids:
            size+=len(self.Teams[teamid])
         avsize=size/len(teamsids) 
         tsize.append(int(avsize))  
       return tsize

    def loadTeams(self):        
        gfile=open(self.path+self.dataset+"/teams.txt")
        gfile.readline()
        e=gfile.readline()
        self.Teams=[]
        while e:
            ids=e.strip().split(" ")
            i=int(ids[0])                    
            if i not in self.Teams:
                self.Teams.append([])            
            for j in ids[1:]:    
                        self.Teams[i].append(int(j))                  
            e=gfile.readline()
        #print(self.Teams)
        self.N=len(self.Teams)
        #print("N=",self.N)
        gfile.close()

    def loadtestresults(self):   
        res=np.loadtxt(self.path+self.dataset+"/team2box/results.txt")
        testqidmap=np.loadtxt(self.path+self.dataset+"/team2box/testquestionsids.txt")
        quetionids=np.loadtxt(self.path+self.dataset+"/team2box/allquestionsids.txt")

        r=[]
        for i in range(len(res)):
            t=res[i]
            #print(t)
            r.append([])
            r[i].append(testqidmap[i])
            j=0
            while j<len(t):
                r[i].append(quetionids[int(t[j])-1])
                r[i].append(t[j+1])
                j=j+2
        return np.array(r,dtype=np.float32)       

    def maptestqtoembeddingspace(self,eqembedding):        
        qidmap={}
        mapfile=open(self.path+self.dataset+"/Q_ID_Map.txt","r")
        line=mapfile.readline()
        line=mapfile.readline().strip()
        while line:
            ids=line.split(" ")
            qidmap[int(ids[1])]=int(ids[0])
            line=mapfile.readline().strip()
        mapfile.close()    
        testembed=np.zeros((len(self.testquestions),len(eqembedding[0])))
        for i in range(len(self.testquestions)):
             qids=self.testquestions[i,1::2]             
             #print("qids=",qids)
             sims=self.testquestions[i,2::2]

             qnum=int(len(qids)/1)
             qnum=self.numq2map
             ssims=sims[0:qnum]

             sumsims=np.sum(ssims)
             normsims=ssims/sumsims
             #print(normsims)

             for j in range(qnum):
                #testembed[i]+=normsims[j]*eqembedding[qidmap[int(qids[j])]]
                testembed[i]+=eqembedding[qidmap[int(qids[j])]]               
             #testembed[i]=testembed[i]/qnum
             testembed[i]=testembed[i]/qnum             
        return testembed

    def findtopkteams(self,topk): 
        topteams=[]       
        for i in range(len(self.testquestions)):
            test=self.testqmap2embeddings[i]
            #print(test)
            c=np.square(test-self.teamcenters)         
            d=np.sqrt(np.sum(c,axis=1))
            #print("d=",d)  
            ids=list(range(0,len(self.teamcenters)))           
            #o=d.reshape((len(d),1))
            so,sids=(list(t) for t in zip(*sorted(zip(d.tolist(), ids),reverse=False)) )
            #print("so=",so[0:topk])
            #print("ids=",sids[0:topk])            
            topteams.append(sids[0:topk])
        return topteams

    def findnumcommonquestions(self,teams,outfile):

        gfile=open(self.path+self.dataset+"/CQAG.txt")        
        e=gfile.readline()
        G={}
        while e:
            ids=e.split(" ")
            i=int(ids[0])
            j=int(ids[1])
            w=int(ids[2])

            if i not in G:
                G[i]={'n':[],'w':[]}

            if j not in G:    
                        G[j]={'n':[],'w':[]}

            G[i]['n'].append(j)
            G[i]['w'].append(w) 

            G[j]['n'].append(i)
            G[j]['w'].append(w)
            e=gfile.readline()
        N=len(G)
        gfile.close()
        outfile.write("+++++++++\n++++++++++\n common questions answered by team members:")
        totalcommonq=0
        for i in range(len(teams)):
             team=teams[i]
             #print(team)
             outfile.write("\n\n q "+str(i)+" team={"+",".join([str(e) for e in team])+"}")
             tq=[]
             for e in team:
                eq=[] 
                for a in G[e]['n']:
                    for q in G[a]['n']:
                       if q!=e:
                          eq.append(q)
                tq.append(eq)
             #print("i=",i,tq)
             commonq=0
             for ii in range(len(tq)):
                qii=set(tq[ii])
                jj=ii+1
                while jj<len(tq):
                   qjj=set(tq[jj])
                   commonq+=len(list(qii.intersection(qjj)))
                   outfile.write("\ne "+str(team[ii])+ "has answered "+str(len(list(qii.intersection(qjj))))
                                 +" common questions with e "+str(team[jj])+ "common qs={"+",".join([str(q) for q in list(qii.intersection(qjj))])+"}")
                   jj+=1
             numpair=len(team)*(len(team)-1)/2.0
             totalcommonq+= (commonq/numpair)
             #totalcommonq+= commonq  
        teamsize=len(teams)   

        return (totalcommonq)/(teamsize)

    def findnumcommonquestions_2(self,teams,outfile):
        gfile=open(self.path+self.dataset+"/CQAG.txt")        
        e=gfile.readline()

        G={}
        while e:
            ids=e.split(" ")
            i=int(ids[0])
            j=int(ids[1])
            w=int(ids[2])

            if i not in G:
                G[i]={'n':[],'w':[]}

            if j not in G:    
                        G[j]={'n':[],'w':[]}

            G[i]['n'].append(j)
            G[i]['w'].append(w) 

            G[j]['n'].append(i)
            G[j]['w'].append(w)
            e=gfile.readline()
        N=len(G)
        gfile.close()
        outfile.write("+++++++++\n++++++++++\n common questions answered by team members:")
        totalcommonq=0
        for i in teams:
             team=teams[i]
             #print(team)
             outfile.write("\n\n q "+str(i)+" team={"+",".join([str(e) for e in team])+"}")
             tq=[]
             for e in team:
                eq=[] 
                for a in G[e]['n']:
                    for q in G[a]['n']:
                       if q!=e:
                          eq.append(q)
                tq.append(eq)
             #print("i=",i,tq)
             commonq=0
             for ii in range(len(tq)):
                qii=set(tq[ii])
                jj=ii+1
                while jj<len(tq):
                   qjj=set(tq[jj])
                   commonq+=len(list(qii.intersection(qjj)))
                   outfile.write("\ne "+str(team[ii])+ "has answered "+str(len(list(qii.intersection(qjj))))
                                 +" common questions with e "+str(team[jj])+ "common qs={"+",".join([str(q) for q in list(qii.intersection(qjj))])+"}")
                   jj+=1
             #totalcommonq+= ((2*commonq)/(len(team)*(len(team)-1)))
             if len(team)>1:
                numpair=len(team)*(len(team)-1)/2.0
             else:
                numpair=1
             totalcommonq+= (commonq/numpair)

        teamsize=len(teams)        
        return (totalcommonq)/(teamsize)

    def run_test():        
        dataset=["android","history","dba","physics","mathoverflow"]        

        Numq2maps={"3":12,"4":12,"5":12}
        Numtopkteams={"3":14,"4":14,"5":14}

        outputfile=open("Results_vs_ExpertFinding.txt","w")
        
        for data in dataset:
            print(data)
            teamsembeding="/team2box/teamsembeding.txt"
            teamsOffsets="/team2box/teamsOffsets.txt"
            team2box_w1_embedding="/team2box/expert_question_w1_embedding.txt"
            m2v_w1_embedding="/metapath/m2v_expert_question_w1_embedding.txt"
            teamsizes=[3,4,5]            
            outputfile.write("\n\n\n**********************\ndata="+data)      

            for teamsize in teamsizes:
                Numq2map=Numq2maps[str(teamsize)]
                Numtopkteam=Numtopkteams[str(teamsize)]  
                outputfile.write("\n----------------------------------\nteam size="+str(teamsize))
                outputfile.write("\nnumq2map="+str(Numq2map)+" Numtopkteams="+str(Numtopkteam))                
                ob=TeamFormation("../data/",data)
                ob.CompareExpertFinding(teamsembeding,teamsOffsets,team2box_w1_embedding,m2v_w1_embedding,teamsize,outputfile,Numq2map,Numtopkteam)  
                outputfile.flush() 
        outputfile.close()

    def run_test_comparison_with_classicTF():      
        dataset=["android","history","dba","physics","mathoverflow"]  

        Numq2maps={"android":11,"history":11,"dba":11,"physics":11,"mathoverflow":11}
        Numtopkteams={"android":13,"history":13,"dba":13,"physics":13,"mathoverflow":13}

        outputfile=open("Results_vs_ClassicTeamFormation.txt","w")
        #outputfile.write(str(datetime.datetime.now()))
        for data in dataset:
            print(data)
            outputfile.write("\n**********************\ndata="+data)
            teamsembeding="/team2box/teamsembeding.txt"
            teamsOffsets="/team2box/teamsOffsets.txt"
            team2box_w1_embedding="/team2box/expert_question_w1_embedding.txt"
            Numq2map=Numq2maps[data]
            Numtopkteam=Numtopkteams[data]
            ob=TeamFormation("../data/",data)       
            ob.CompareTeamFormation(teamsembeding,teamsOffsets,team2box_w1_embedding,outputfile,Numq2map,Numtopkteam)
            outputfile.flush()
        outputfile.close()    
       

TeamFormation.run_test()
TeamFormation.run_test_comparison_with_classicTF()

      
