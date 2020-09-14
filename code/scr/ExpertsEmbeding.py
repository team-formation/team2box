from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk
from matplotlib.patches import Rectangle
import networkx as nx
import tensorflow as tf
import numpy as np
import random
import gensim 
from gensim.models import Word2Vec 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
class ExpertsEmbeding:        
    def __init__(self,hsize,data,runname,num):        
        self.dataset=data  
        self.runname=runname
        self.num=num
        #3self.load_graph()
        #self.save_qraph()
        #sys.exit()
        pfile=open(self.dataset+"/CQAG_properties.txt")
        pfile.readline()
        properties=pfile.readline().strip().split(" ")
        pfile.close()
        self.N=int(properties[0]) # number of nodes in the CQA network graph N=|Qestions|+|Answers|+|Experts|                
        self.qnum=int(properties[1])
        self.anum=int(properties[2])
        self.enum=int(properties[3])
        self.G={}  
        self.loadG()               
        #self.displayG()
        
        # load team centers and offsets
        self.teamcenters=np.loadtxt(self.dataset+"/team2box/"+self.runname+"teamsembeding_"+self.num+".txt")
        print(self.teamcenters)
        self.teamoffsets=np.loadtxt(self.dataset+"/team2box/"+self.runname+"teamsOffsets_"+self.num+".txt")
        print(self.teamoffsets)
        
        
        # load expert best team
        gfile=open(self.dataset+"/ExpertBestQuetionAnswer.txt")        
        gfile.readline()
        line=gfile.readline().strip()
        self.ebt={}
        while line:
            ids=line.split(" ")
            self.ebt[int(ids[0])]=[int(ids[1]),int(ids[2]),float(ids[3])]
            line=gfile.readline()
        print("ebt=",self.ebt)
        gfile.close()
        
        self.hidden_size=hsize
        self.W1=ExpertsEmbeding.weight_variable((self.qnum+self.enum,self.hidden_size))  #only embeding for questions and experts             
        self.W1[self.qnum:self.qnum+self.enum].assign(self.weight_variable_experts((self.enum,self.hidden_size)))
        self.W2=ExpertsEmbeding.weight_variable((self.qnum+self.enum,self.hidden_size)) 
        #self.W1=self.weight_variable_experts((self.enum,self.hidden_size))
        #self.W2=ExpertsEmbeding.weight_variable((self.enum,self.hidden_size))
        #self.displayEmbedding()
    
    def weight_variable(shape):
        tmp = np.sqrt(6.0) / np.sqrt(shape[0] + shape[1])
        initial = tf.random.uniform(shape, minval=-tmp, maxval=tmp)
        return tf.Variable(initial)
    
    def weight_variable_experts(self,shape):  
        
        #initial=[]
        

        x=[]
        for i in range(self.enum):
            expertid=i+self.qnum+self.anum
            eteamid=self.ebt[expertid][0]
            offsets= self.teamoffsets[eteamid]
            #print(offsets)
            r = offsets * np.sqrt(np.random.uniform(0,1))
            #print("r",r)
            theta = np.random.uniform(0,1) * 2 * 3.14
            
            x.append([])
           #print("shape",shape[1])
            #print(self.teamcenters[self.ebt[expertid][0]])
            for j in range(shape[1]):
                if j%2==0:
                    #print("j=",j)
                    x[i].append(self.teamcenters[self.ebt[expertid][0]][j]+ r[j] * np.cos(theta))
                else:
                    x[i].append( self.teamcenters[self.ebt[expertid][0]][j]+ r[j] * np.sin(theta))
        #for i in rangwe()
        
        #templow=[]
        #temphigh=[]
        #for i in range(self.enum):
        #    expertid=i+self.qnum+self.anum
        #    templow.append(self.teamcenters[self.ebt[expertid][0]]-self.teamoffsets[self.ebt[expertid][0]])
        #    temphigh.append(self.teamcenters[self.ebt[expertid][0]]+self.teamoffsets[self.ebt[expertid][0]])
        
        
        
        initial=np.array(x,dtype=np.float32)
        #tmp = np.sqrt(6.0) / np.sqrt(shape[0] + shape[1])
        #initial = tf.random.uniform(shape, minval=-tmp, maxval=tmp)
        #print("initial=",initial)
        return tf.Variable(initial)
    
    def loadG(self):        
        gfile=open(self.dataset+"/CQAG.txt")
        
        e=gfile.readline()
        self.G={}
        while e:
            ids=e.split(" ")
            i=int(ids[0])
            j=int(ids[1])
            w=int(ids[2])
            
            if i not in self.G:
                self.G[i]={'n':[],'w':[]}
            
            if j not in self.G:    
                        self.G[j]={'n':[],'w':[]}
                    
            self.G[i]['n'].append(j)
            self.G[i]['w'].append(w) 
            
            self.G[j]['n'].append(i)
            self.G[j]['w'].append(w)
            e=gfile.readline()
        self.N=len(self.G)
        #print(self.G)
        gfile.close()   
            
    def load_graph(self):   
        self.G={}
        qpfile=open(self.dataset+"q_answer_ids_score.txt")
        qpfile.readline()
        line=qpfile.readline().strip()
        qids=[]
        aids=[]
        eids=[]
        while line:
            qp=line.split(" ")
            qid=int(qp[0].strip())            
            if qid not in qids:
                qids.append(qid)
            caids=qp[1::2] 
            for aid in caids:
                if int(aid) not in aids:
                    aids.append(int(aid))
            line=qpfile.readline().strip()    
        qpfile.close()  
        print(len(qids))
        print(len(aids))
        pufile=open(self.dataset+"postusers.txt")
        pufile.readline()
        line=pufile.readline().strip()
        while line:
            ids=line.split(" ")
            eid=int(ids[1].strip())            
            if eid not in eids:
                eids.append(eid)
            line=pufile.readline().strip() 
        pufile.close()
        print(len(eids))
        
        self.qnum, self.anum, self.enum=len(qids), len(aids), len(eids)
        self.N=len(qids)+len(aids)+len(eids)
        
        #create CQA network graph
        qpfile=open(self.dataset+"/krnmdata1/questionposts.txt")
        qpfile.readline()
        line=qpfile.readline().strip()        
        while line:
            qp=line.split(" ")
            qid=qids.index(int(qp[0].strip()))           
            if qid not in self.G:
                self.G[qid]={'n':[],'w':[]}
                
            caids=qp[1::2] 
            #print(caids)
            caidsscore=qp[2::2] 
            #print(caidsscore)
            for ind in range(len(caids)):
                aid=aids.index(int(caids[ind]))+len(qids)
                if aid not in self.G:
                    self.G[aid]={'n':[qid],'w':[int(caidsscore[ind])]}
                self.G[qid]['n'].append(aid)
                self.G[qid]['w'].append(int(caidsscore[ind]))
            line=qpfile.readline().strip()    
        qpfile.close() 
        pufile=open(self.dataset+"/krnmdata1/postusers.txt")
        pufile.readline()
        line=pufile.readline().strip()
        while line:
            ids=line.split(" ")
            aid=aids.index(int(ids[0].strip()))+len(qids)
            eid=eids.index(int(ids[1].strip()))+len(qids)+len(aids)           
                      
            if eid not in self.G:
                self.G[eid]={'n':[aid],'w':[self.G[aid]['w'][0]]}
                
            else:
                self.G[eid]['n'].append(aid)
                self.G[eid]['w'].append(self.G[aid]['w'][0])
            self.G[aid]['n'].append(eid)
            self.G[aid]['w'].append(self.G[aid]['w'][0])    
            line=pufile.readline().strip() 
        pufile.close()
    
        
    def save_qraph(self):
        qfile=open(self.dataset+"/krnmdata1/CQAG1.txt","w")
        #qfile.write("N="+str(self.N)+" Questions= "+str(self.qnum)+" index=0.."+str(self.qnum-1)
        #            +"; Answers= "+str(self.anum)+" index="+str(self.qnum)+".."+str(self.qnum+self.anum-1)
        #            +"; Experts= "+str(self.enum)+" index="+str(self.qnum+self.anum)+".."+str(self.qnum+self.anum+self.enum-1)+"\n")
        
        for node in self.G:
            for i in range(len(self.G[node]['n'])):
                if node< self.G[node]['n'][i]:
                    qfile.write(str(node)+" "+str(self.G[node]['n'][i])+" "+str(self.G[node]['w'][i])+"\n")
        
        qfile.close()
        pfile=open(self.dataset+"/krnmdata1/properties.txt","w")
        pfile.write("N="+str(self.N)+" Questions= "+str(self.qnum)+" index=0.."+str(self.qnum-1)
                    +"; Answers= "+str(self.anum)+" index="+str(self.qnum)+".."+str(self.qnum+self.anum-1)
                    +"; Experts= "+str(self.enum)+" index="+str(self.qnum+self.anum)+".."+str(self.qnum+self.anum+self.enum-1)+"\n")
        pfile.write(str(self.N)+" "+str(self.qnum)+" "+str(self.anum)+" "+str(self.enum))
        pfile.close()
    
    def displayG(self):
        G=nx.Graph();
        G=nx.read_weighted_edgelist(self.dataset+"/CQAG.txt")
        edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0]
        pos = nx.spring_layout(G,k=0.2,iterations=50)  # positions for all nodes
        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=100)#,node_color='g',fill_color='w')
        # edges
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=2)
        # labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        
        labels = nx.get_edge_attributes(G,'weight')
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
        
        plt.axis('off')
        plt.show()
        
    def displayEmbedding(self):       
        Centers=self.teamcenters.copy()
        Offsets=self.teamoffsets.copy()
        #print(embed)
        #model = TSNE(n_components=2, random_state=0)
        #y=model.fit_transform(embed) 
        y=Centers
        max1=np.amax(y,axis=0)+1.1*np.amax(Offsets,axis=0)
        min1=np.amin(y,axis=0)-1.1*np.amax(Offsets,axis=0)
        
        plt.figure(figsize=(5,5))
        plt.plot(y[0:len(Centers),0],y[0:len(Centers),1],'gx');
        
        for i in range(len(Centers)):
            plt.text(y[i,0],y[i,1], "t"+str(i), fontsize=8)  
        
        ax = plt.gca()    
        ax.set_aspect(1)
        for i in range(len(Centers)):
            #c= Rectangle((y[i,0]-Offsets[i,0],y[i,1]-Offsets[i,0]), 2*Offsets[i,0],2*Offsets[i,0] , linewidth=1,edgecolor='r',facecolor='none')
            c=plt.Circle((y[i,0], y[i,1]), Offsets[i,0], color='b', clip_on=False,fill=False)
            ax.add_patch(c)
        
        
        plt.plot(self.W1.numpy()[:self.qnum,0],self.W1.numpy()[:self.qnum,1],'b+',markersize=14);
        
        for i in range(self.qnum):
            plt.text(self.W1.numpy()[i,0],self.W1.numpy()[i,1], "q"+str(i), fontsize=8)
        
        plt.plot(self.W1.numpy()[self.qnum:,0],self.W1.numpy()[self.qnum:,1],'r*',markersize=12);
        
        for i in range(self.enum):
            plt.text(self.W1.numpy()[self.qnum+i,0],self.W1.numpy()[self.qnum+i,1], "e"+str(self.qnum+i), fontsize=8)
        #ax.set_xlim([min1[0], max1[0]])
        #ax.set_ylim([min1[1], max1[1]])
        
       
        plt.show();
    
    
    def walker(self,start, walklen):
        walk=""
        ii=0
        
        #start=random.randint(self.qnum+self.anum,self.N) # start from expert
        prev=start
        while ii<walklen: 
            #print("st="+ str(start)+" pre="+str(prev))            
            ind=0
            if len(self.G[start]['n'])==1:
                neib=self.G[start]['n']
                #print(neib)
                ind=0  
            else:
                weights=self.G[start]['w'].copy()  
                neib=self.G[start]['n'].copy()
                #print(neib)
                #print(weights)
                if prev in neib:
                    indpre=neib.index(prev)                
                    del weights[indpre:indpre+1]
                    del neib[indpre:indpre+1]
                    #print(neib)
                    #print(weights)
                if len(neib)==1:
                    ind=0
                else:    
                    sumw=sum(weights)                
                    ranw=random.randint(1,sumw)
                    #print("sumw="+str(sumw)+" ranw="+str(ranw))                
                    for i in range(len(neib)):
                        if ranw<=sum(weights[0:i+1]):
                            ind=i
                            break
                        
            if start<self.qnum or start>self.qnum+self.anum:
                if start>self.qnum+self.anum:
                    start=start-(self.anum)
                walk+= " "+str(start )           
            prev=start
            start=neib[ind]
            
            #if start>self.qnum+self.anum:
            ii+=1
        return walk.strip()    
    
    def walks1(self,walklen,n):
        data=[]
        for i in range(self.qnum):
            for j in range(n):
                walk=self.walker(i,3).split(" ")
                data.append(walk)
        for i in range(self.enum):
            for j in range(n):
                walk=self.walker(self.qnum+self.anum+i,walklen).split(" ")
                data.append(walk)
        return data
   
    def get_train_pair(self,walks,windowsize, N): 
        #print(N)
        z=np.zeros((N))
        total=0
        for i in range(len(walks)):
            total+=len(walks[i])
            for j in walks[i]:
                z[int(j)]+=1
        #print(z) 
        #print(total)
        z1=z/total
        p=(np.sqrt(z1/0.001)+1)*(0.001/z1)  #probability of keeping a node in the traing
        #print(p)
        z2=np.power(z,.75)
        p2=z2/np.sum(z2)
        #print(p2)
        negsamples=[]
        for i in range(N):
            rep=int(p2[i]*100) 
            if rep==0:
               rep=1 
            for j in range(rep):
                negsamples.append(i) 
        #print(negsamples) 
        negs=np.array(negsamples)
        np.random.shuffle(negs)
        #print(negs)
        pairs=[]
        for i in range(len(walks)):
            walk=walks[i]                     
            for context_ind in range(len(walk)):            
                if context_ind>windowsize:
                    start=context_ind-windowsize
                else:
                    start=0
                for i in range(windowsize):
                    if i+start<context_ind:
                        x=np.random.randint(0,100)                    #print(x, p[int(walk[context_ind])]*100)
                        if (100-p[int(walk[context_ind])]*100)>x:
                            continue
                        if  walk[context_ind]!=walk[i+start] :  
                            pairs.append([int(walk[context_ind]),int(walk[i+start])])
                        if i+context_ind+1<len(walk):
                            x=np.random.randint(0,100)                    #print(x, p[int(walk[context_ind])]*100)
                            if (100-p[int(walk[context_ind])]*100)>x:
                                continue
                            if  walk[context_ind]!=walk[i+context_ind+1]:   
                                pairs.append([int(walk[context_ind]),int(walk[i+context_ind+1])])
        pairs=np.array(pairs)
        print("number of train samples:",len(pairs))
        return pairs,negs
    def get_train_pair2(self,walks,windowsize, N): 
        #print(N)
        z=np.zeros((N))
        total=0
        for i in range(len(walks)):
            total+=len(walks[i])
            for j in walks[i]:
                z[int(j)]+=1
        #print(z) 
        #print(total)
        z1=z/total
        p=(np.sqrt(z1/0.001)+1)*(0.001/z1)  #probability of keeping a node in the traing
        #print(p)
        z2=np.power(z,.75)
        p2=z2/np.sum(z2)
        #print(p2)
        negsamples=[]
        for i in range(N):
            rep=int(p2[i]*100)  
            for j in range(rep):
                negsamples.append(i) 
        #print(negsamples) 
        negs=np.array(negsamples)
        np.random.shuffle(negs)
        #print(negs)
        pairs=[]
        for i in range(len(walks)):
            walk=walks[i]                     
            for context_ind in range(len(walk)):            
                if context_ind>windowsize:
                    start=context_ind-windowsize
                else:
                    start=0
                for i in range(windowsize):
                    if i+start<context_ind:
                        x=np.random.randint(0,100)                    #print(x, p[int(walk[context_ind])]*100)
                        #if (100-p[int(walk[context_ind])]*100)>x:
                        #    continue
                        if  walk[context_ind]!=walk[i+start] :  
                            pairs.append([int(walk[context_ind]),int(walk[i+start])])
                        if i+context_ind+1<len(walk):
                            #x=np.random.randint(0,100)                    #print(x, p[int(walk[context_ind])]*100)
                            #if (100-p[int(walk[context_ind])]*100)>x:
                            #    continue
                            if  walk[context_ind]!=walk[i+context_ind+1]:   
                                pairs.append([int(walk[context_ind]),int(walk[i+context_ind+1])])
        pairs=np.array(pairs)
        print("number of train samples:",len(pairs))
        return pairs,negs
    def walks(self,walklen,n1):
        G=nx.Graph();
        G=nx.read_weighted_edgelist(self.dataset+"/CQAG.txt")
        
        rw = BiasedRandomWalk(StellarGraph(G))

        weighted_walks = rw.run(
        nodes=G.nodes(), # root nodes
        length=walklen,    # maximum length of a random walk
        n=n1,          # number of random walks per root node 
        p=0.5,         # Defines (unormalised) probability, 1/p, of returning to source node
        q=2.0,         # Defines (unormalised) probability, 1/q, for moving away from source node
        weighted=True, #for weighted random walks
        seed=42        # random seed fixed for reproducibility
        )
        print("Number of random walks: {}".format(len(weighted_walks)))
        #print(weighted_walks[0:10])
        
        #remove answer nodes
        walks=[]
        for i in range(len(weighted_walks)):
            walk=weighted_walks[i]
            w=[]
            for node in walk:
                if int(node)<self.qnum:
                    w.append(node)
                elif int(node)>(self.qnum+self.anum):
                    n=int(node)-self.anum
                    w.append(str(n))
            walks.append(w)        
        print(walks[0:10])
        return walks
    
    def loss(predicted_y, target_y):
        #current_loss=  tf.square(predicted_y[0] - target_y[0])+ tf.reduce_mean(tf.square(predicted_y[1:] - target_y[1:]))
        return tf.reduce_mean(tf.square(predicted_y - target_y))

    def model(self,inputs_i,inputs_j):    
        # look up embeddings for each term. [nbatch, qlen, emb_dim]
        i_embed = tf.nn.embedding_lookup(self.W1, inputs_i, name='iemb')
        j_embed = tf.nn.embedding_lookup(self.W2, inputs_j, name='jemb')          
        # Learning-To-Rank layer. o is the final matching score.
        temp=tf.transpose(j_embed, perm=[1, 0])
        o = tf.sigmoid(tf.matmul(i_embed, temp))
        o = tf.reshape(o, (len(o[0]),1))
        #print("o=")
        #print(o)
        return o
    
    def train(self, inputs_i, inputs_j, outputs, learning_rate):
        #print(inputs_i)
        i_embed = tf.nn.embedding_lookup(self.W1, inputs_i, name='iemb')
        j_embed = tf.nn.embedding_lookup(self.W2, inputs_j, name='jemb')  
        with tf.GradientTape() as t:
            current_loss = ExpertsEmbeding.loss(self.model(inputs_i,inputs_j), outputs)
        dW1, dW2 = t.gradient(current_loss, [self.W1, self.W2])
        #print("dw1")
        #print(dW1)
        #print(dW2)
        
        #indexw1=tf.Variable(inputs_i,dtype=tf.int32)
        #indexw1=tf.reshape(indexw1,(indexw1.shape[0],1))
        i_embed=i_embed-(learning_rate * dW1.values)
        
                
        k1=0
        #print(inputs_i.numpy())
        for k in inputs_i.numpy():
            #print(k)
            if k<self.qnum:
                self.W1[k,:].assign(i_embed[k1,:])
            else:
                teamcenter=np.array(self.teamcenters[self.ebt[self.anum+k][0]])
                c=tf.square(tf.subtract(i_embed,teamcenter))         
                d=tf.sqrt(tf.reduce_sum(c,axis=1)).numpy()[0]
                #print("d=",d)
                if d<self.teamoffsets[self.ebt[self.anum+k][0]][0]:
                    #print("offset=",self.teamoffsets[self.ebt[self.anum+k][0]][0])
                    self.W1[k,:].assign(i_embed[k1,:])
            k1+=1
        #self.W1.assign(tf.tensor_scatter_nd_update(self.W1,indexw1,i_embed))
        
        #indexw2=tf.Variable(inputs_j,dtype=tf.int32)
        #indexw2=tf.reshape(indexw2,(indexw2.shape[0],1))
        j_embed=j_embed-(learning_rate * dW2.values)
        #self.W2.assign(tf.tensor_scatter_nd_update(self.W2,indexw2,j_embed))
        k1=0
        #print(inputs_i.numpy())
        for k in inputs_j.numpy():
            #print(k)
            self.W2[k,:].assign(j_embed[k1,:])
            k1+=1
        #print(self.W1)
        #print(self.W2)
        return current_loss
    
    #def get_train_pair(self,walks)
    
    def run(self,walklen):
        #self.load_graph(dataset)        
        walks=self.walks(walklen,10) #n: number of walks start from a node
        #print(walks)
        pairs,negsamples=self.get_train_pair(walks,2,self.qnum+self.enum)
        lennegs=len(negsamples)
        print(pairs)
        epochs = range(2)
        loss_total=0
        train_len=len(pairs)
        logfile=open(self.dataset+"/team2box/"+self.runname+"ExpertsEmbedinglog"+self.num+"_.txt","w")
        
        for epoch in epochs:  
            #walks=self.walks(walklen)            
            loss=0
            for k in range(train_len):
                    tpairs_i=[]
                    tpairs_j=[]
                    tpairs_i.append(pairs[k][0])
                    tpairs_j.append(pairs[k][1])
                    #add negetive samples
                    #negsample=[]
                    nk=0
                    while nk<10:                        
                        neg=random.randint(0,lennegs-1)                        
                        if negsamples[neg] != tpairs_i and negsamples[neg] not in tpairs_j and negsamples[neg] not in self.G[pairs[k][0]]['n']:
                            tpairs_j.append(negsamples[neg])
                            nk+=1
                    #print(tpairs_i)
                    #print(tpairs_j)
                    inputs_i=tf.Variable(tpairs_i,dtype=tf.int32)
                    inputs_j=tf.Variable(tpairs_j,dtype=tf.int32)
                    
                    out=np.zeros(shape=(inputs_j.shape[0],1))
                    out[0]=1
                    outputs=tf.Variable(out,dtype=tf.float32) 
                    #outputs=tf.reshape(outputs,(outputs.shape[0],1))
                   # print("out=",outputs)
                    #print("current_loss= %2.5f"%(current_loss))
                    loss+=self.train( inputs_i, inputs_j, outputs, learning_rate=0.1)
                #if i%100==0:
                #    print('Epoch %2d: Node %4d: loss=%2.5f' % ( epoch, i ,  loss))
                    #print(self.W1)
            loss_total+=(loss/train_len)
            if epoch%1==0:
                #print('Epoch %2d: loss=%2.5f' % (epoch,  loss_total/(epoch+1)) )  
                l=loss_total/(epoch+1)
                logfile.write("Epoch: "+str(epoch)+" loss: "+str(l.numpy() )+"\n") 
                logfile.flush()    
            if epoch%1==0:                
                #self.displayG()
                #self.displayEmbedding()
                self.save_embeddings(epoch)
        logfile.close()
    def save_embeddings(self,i):
        #qfile=open(self.dataset+"/krnmdata1/teamsembeding.txt","w")
        w1=self.W1.numpy()
        w2=self.W2.numpy()
        np.savetxt(self.dataset+"/team2box/"+self.runname+"expert_question_w1_embedding"+self.num+"_"+str(i)+".txt",w1, fmt='%f')
        np.savetxt(self.dataset+"/team2box/"+self.runname+"expert_question_w2_embedding"+self.num+"_"+str(i)+".txt",w2, fmt='%f')

dataset=["android","dba","physics","history","mathoverflow"]
ob2=ExpertsEmbeding(32,dataset[2],"run1/","85")  
ob2.run(9)
