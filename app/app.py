import pickle
import os
import re
import requests
import pickle
import numpy as np 
import pandas as pd
import csv
import Levenshtein
import numpy as np
from scipy.spatial.distance import euclidean
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects import conversion, default_converter
import torch.nn as nn
import torch.nn.functional as F
from tensorflow import keras

import math

import torch
import time
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim

from bs4 import BeautifulSoup

with open("lncRNA_feature.pickle", 'rb') as file:
    lncRNA_feature=pickle.load(file)
l1=lncRNA_feature[:,0:4458]
l2=lncRNA_feature[:,4458:4458*2]
l3=lncRNA_feature[:,4458*2:4458*3]

with open("disease_feature.pickle", 'rb') as file:
    gdi=pickle.load(file)
    
d2=gdi[:,0:468]
d1=gdi[:,468:]
gdi=np.hstack((d1,d2))

with open("lda.pickle", 'rb') as file:
    lda=pickle.load(file)

with open("disease_names.pickle", 'rb') as file:
    diseases=pickle.load(file)

with open("lncRNA_names.pickle", 'rb') as file:
    lncRNA_names=pickle.load(file)


with open("lncTarget.pickle", 'rb') as file:
    lncTarget=pickle.load(file)
    
with open("sequences.pickle", 'rb') as file:
    sequences=pickle.load(file)

with open("model1.pickle", 'rb') as file:
    model1=pickle.load(file)

with open("model2.pickle", 'rb') as file:
    model2=pickle.load(file)

with open("model3.pickle", 'rb') as file:
    model3=pickle.load(file)

with open("model4.pickle", 'rb') as file:
    model4=pickle.load(file)

with open("targetNames.pickle", 'rb') as file:
    targetNames=pickle.load(file)

# with open("dis_doid_dic.pickle", 'rb') as file:
#     dis_doid_dic=pickle.load(file)

with open("doid_dic.pickle", 'rb') as file:
    dis_doid_dic=pickle.load(file)

dis_doid_dic = {key.lower(): value for key, value in dis_doid_dic.items()}

with open("genes_names_relatedToDisease.pickle", 'rb') as file:
    disease_genes=pickle.load(file)
    
with open("disease_genes_association.pickle", 'rb') as file:
    inter_disease_genes=pickle.load(file)

with open("doids.pickle", 'rb') as file:
    doids=pickle.load(file)

with open("gip_dis.pickle", 'rb') as file:
    result_dis=pickle.load(file)

with open("gip_lnc.pickle", 'rb') as file:
    result_lnc=pickle.load(file)




# with open("encoder_lnc.pickle",'rb') as file:
#     encoder_lnc=pickle.load(file)

# with open("encoder_dis.pickle",'rb') as file:
#     encoder_dis=pickle.load(file)

encoder_lnc = keras.models.load_model("encoder_lnc.h5")

encoder_dis = keras.models.load_model("encoder_dis.h5")

# with open("GCN_node1.pickle",'rb') as file:
#     GCN_node1=pickle.load(file)

# with open("GCN_node2.pickle",'rb') as file:
#     GCN_node2=pickle.load(file)
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import time
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, input, adj):
 
        # Convolution operation
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        return output


class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.W_l = nn.Parameter(torch.FloatTensor(input_dim, input_dim))
        self.W_r = nn.Parameter(torch.FloatTensor(input_dim, input_dim))
        self.W_h = nn.Parameter(torch.FloatTensor(input_dim, input_dim))
        self.W_g = nn.Parameter(torch.FloatTensor(input_dim, input_dim))
        self.gamma = nn.Parameter(torch.FloatTensor([0]))  # trainable parameter γ
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_l)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.xavier_uniform_(self.W_h)
        nn.init.xavier_uniform_(self.W_g)

    def forward(self, H, adj):
        # Compute attention scores
        H_l = torch.matmul(H, self.W_l)
        H_r = torch.matmul(H, self.W_r)
        S = torch.matmul(H_l, torch.transpose(H_r, 0, 1))

        # Apply softmax to normalize attention scores along the last dimension
        beta = F.softmax(S, dim=-1)

        # Weighted sum of input elements based on attention weights
        B = torch.matmul(beta, H)

        # Calculate attention feature
        O = torch.matmul(B, self.W_h)
        O = torch.matmul(O, self.W_g)

        # Interpolation step
        output = torch.matmul(adj,H) + self.gamma * O 

        return output
    
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.relu1 = nn.ReLU()
        self.gc2 = GraphConvolution(nhid, nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.relu2 = nn.ReLU()
        self.attention = AttentionLayer(nhid)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.gc2(x, adj)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.attention(x,adj)
        return x


def calculate_laplacian(adj):
    # Calculate the degree matrix
    degree = torch.sum(adj, dim=1)
    degree_matrix = torch.diag(degree)
    
    # Calculate the Laplacian matrix
    laplacian = degree_matrix - adj
    return laplacian

def adj_norm(adj):
    adj_hat = adj + torch.eye(adj.size(0), device=adj.device)

    # Compute degree matrix
    degree = torch.sum(adj_hat, dim=1)
    degree = torch.diag(degree)

    # Compute D^-0.5
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0

    # Normalize adjacency matrix
    adj_normalized = torch.mm(torch.mm(degree_inv_sqrt, adj_hat), degree_inv_sqrt)
    
    return adj_normalized

    
    
GCN_node1 = GCN(nfeat=512, nhid=256,dropout=0.4)
GCN_node2 = GCN(nfeat=512, nhid=256,dropout=0.4)

GCN_node1 = torch.load('GCN_node1.pth')

GCN_node2 = torch.load('GCN_node2.pth')

with open("scaler1.pickle",'rb') as file:
    scaler1=pickle.load(file)

with open("base_models1.pickle", 'rb') as file:
    base_models1=pickle.load(file)

with open("meta_model1.pickle", 'rb') as file:
    meta_model1=pickle.load(file)

with open("scaler2.pickle", 'rb') as file:
    scaler2=pickle.load(file)

with open("base_models2.pickle", 'rb') as file:
    base_models2=pickle.load(file)

with open("meta_model2.pickle", 'rb') as file:
    meta_model2=pickle.load(file)



diseases=np.array(diseases)
lncRNA_names=np.array(lncRNA_names)
dis_genes=gdi[:,0:468]
ddsim=gdi[:,468:2*468]


from flask import Flask, render_template, request, redirect, jsonify,url_for
from Bio import SeqIO
from io import TextIOWrapper
app = Flask(__name__)


disease_list = diseases 
target_list = targetNames
selected_lncRNA=None
selected_diseases=[]
selected_target=[]
selected_lncRNAs_list=[]
selected_item=None
selected_list=[]
seq=None
dis_doid=None
selected_dis=None
selected_lncRNAs_list=[]
selected_genes_dis=[]
information_dic={}
similarity_stats=None
d2l=[]
l2d=[]



from celery import Celery

celery = Celery(__name__)
celery.conf.broker_url = 'redis://localhost:6379/0'

pandas2ri.activate()
@celery.task
def run_r_code():
    with conversion.localconverter(default_converter):

        robjects.r.library("DOSE")

        robjects.r('''data <- read.csv("create_x1_forDisease.csv", header = FALSE)  
                    disease_list <- as.list(data[[1]])''')

        robjects.r('''target <- tail(disease_list, n = 1)
                                target''')

        robjects.r('''disease_list <- disease_list[-length(disease_list)]''')

        # Execute doSim function
        robjects.r('''ddsim<-doSim(disease_list, target, measure = "Wang")''')

        # Manipulate ddsim data
        robjects.r('''ddsim[is.na(ddsim)] <- 0''')

        # Write ddsim to a CSV file
        robjects.r('''write.csv(ddsim, file = "ddsim_target.csv", row.names = FALSE)''')
def find_diseases_details(query):
    base_url = "https://disease-info-api.herokuapp.com/diseases"
    params = {"name": query}
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data= response.json()
        return data['name'],data['description'],data['symptoms']
    return "Information not available","Information not available","Information not available"
def find_lncRNA_details(query):
    url = f"https://rest.ensembl.org/lookup/id/{query}?content-type=application/json"
    response = requests.get(url)
    if response.status_code == 200:
        lncrna_info = response.json()
        return lncrna_info["display_name"],lncrna_info['biotype'],lncrna_info['description']
    return "Information not available","Information not available","Information not available"
def find_info(sel_item,sel_list,f):
    global information_dic,selected_item,selected_list
    selected_item=sel_item
    selected_list=sel_list
    information_dic={}
    API_KEY = 'c112bfffe2fe8a14645942743d2b4fc72008'
    base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
    details=[]
    for x in sel_list:
        details=[]
        query=sel_item+" "+x
        search_query = f'{base_url}esearch.fcgi?db=pubmed&api_key={API_KEY}&term={query}&retmode=json'
        response = requests.get(search_query)
        count = 0
        # Checking the status of the request
        if response.status_code == 200:
            data = response.json()
            # Extracting the PubMed IDs (PMIDs) of the articles
            pmids = data['esearchresult']['idlist']
            
            # Access individual articles using their PMIDs
            for pmid in pmids:
                if count==5:
                        break
                q={}
                article_query = f'{base_url}efetch.fcgi?db=pubmed&api_key={API_KEY}&id={pmid}&retmode=xml'
                article_response = requests.get(article_query)
                
                # Process the article data (in XML format) or perform other operations
                if article_response.status_code == 200:
                    count+=1
                    
                    article_data = article_response.text
                    soup = BeautifulSoup(article_data, 'xml')
                    
                    # Extracting title, link, and abstract
                    article_title = soup.find('ArticleTitle').text
                    abstract = soup.find('AbstractText')
                    if abstract:
                        abstract_text = abstract.text
                    else:
                        abstract_text = "Abstract not available."
                    
                    # Constructing the link to the article
                    article_link = f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/'
                    
                    # Print title, link, and 
                    # if f==1:
                    #     n,b,d=find_lncRNA_details(x)
                    #     q["name"]=n
                    #     q["biotype"]=b
                    #     q["description"]=d
                    # else:
                    #     n,b,d=find_diseases_details(x)
                    #     q["name"]=n
                    #     q["symptoms"]=d
                    #     q["description"]=b
                    q["title"]=article_title
                    q["link"]=article_link
                    q["abstract"]=abstract_text
                    details.append(q)
                    print(f"Title: {article_title}")
                    print(f"Link: {article_link}")
                    print(f"Abstract: {abstract_text}\n")
                    
                else:
                    print(f"Error fetching article with PMID: {pmid}")

            information_dic[x]=details
        else:
            print("Failed to retrieve data")

        
    





@app.route('/run-r-code')
def execute_r_code():
    run_r_code.delay()  # Execute R code asynchronously
    return 'R code execution triggered!'


@app.route('/input_diseases', methods=['POST'])
def input_diseases():
    global selected_diseases
    selected_diseases = request.form.getlist('diseases', type=str)
    
    if 'fileInput' in request.files:
        uploaded_file = request.files['fileInput']
        if uploaded_file:
            file_content = uploaded_file.read().decode('utf-8')
            file_diseases = [disease.strip() for disease in file_content.split('\n') if disease.strip()]
            
            # Extend the selected diseases with the file-based diseases
            selected_diseases.extend(file_diseases)

    # Print or process the selected diseases
    print("Selected Diseases:", selected_diseases)
    # Process the selected diseases further if needed
    # ...

    return render_template('topDiseasepred.html', selected_diseases=selected_diseases)




@app.route("/input_lncRNA_forDiseease", methods=['POST'])
def input_lncRNA_forDiseease():
    global selected_lncRNAs_list
    selected_lncRNAs_list = request.form.getlist('lncRNAs', type=str)
    
    if 'fileInput' in request.files:
        uploaded_file = request.files['fileInput']
        if uploaded_file:
            file_content = uploaded_file.read().decode('utf-8')
            file_lncRNAs = [lncRNA.strip() for lncRNA in file_content.split('\n') if lncRNA.strip()]
            
            # Extend the selected diseases with the file-based diseases
            selected_lncRNAs_list.extend(file_lncRNAs)

    # Print or process the selected diseases
    print("Selected lncRNAs:", selected_lncRNAs_list)
    # Process the selected diseases further if needed
    # ...

    return render_template('toplncRNApred.html', selected_lncRNAs_list=selected_lncRNAs_list)











@app.route("/input_target",methods=['POST'])
def input_target():
    global selected_target
    selected_target = request.form.getlist('targets',type=str)
    if 'fileInput' in request.files:
        uploaded_file = request.files['fileInput']
        if uploaded_file:
            file_content = uploaded_file.read().decode('utf-8')
            file_targets = [target.strip() for target in file_content.split('\n') if target.strip()]
            
            # Extend the selected diseases with the file-based diseases
            selected_target.extend(file_targets)

    # Print or process the selected diseases
    print("Selected Targets:", selected_target)
 
    return render_template('topDiseasepred.html', selected_target=selected_target)

@app.route("/input_genes_disease",methods=['POST'])
def input_genes_disease():
    global selected_genes_dis
    selected_genes_dis=request.form.getlist('genes',type=str)
    if 'fileInput' in request.files:
        uploaded_file = request.files['fileInput']
        if uploaded_file:
            file_content = uploaded_file.read().decode('utf-8')
            file_targets = [target.strip() for target in file_content.split('\n') if target.strip()]
            
            # Extend the selected diseases with the file-based diseases
            selected_genes_dis.extend(file_targets)

    # Print or process the selected diseases
    print("Selected Genes:", selected_genes_dis)
 
    return render_template('toplncRNApred.html', selected_genes_dis=selected_genes_dis)


@app.route("/input_sequence",methods=['POST'])
def input_sequence():
    global seq
    seq=""
    # print(request.files)
    if 'file' in request.files:
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            # Wrap the file object in TextIOWrapper to open it in text mode
            file_wrapper = TextIOWrapper(uploaded_file, encoding='utf-8')

            # Parse the uploaded file as a FASTA file
            sequences = list(SeqIO.parse(file_wrapper, "fasta"))

            # Process the sequences (e.g., print their IDs and sequences)
            for record in sequences:
                seq+=str(record.seq)

    if 'sequence' in request.form:
        # Get sequence from manual input
        seq = request.form['sequence'] + seq

    seq = re.sub(r"\s+", "", seq)
    print("Sequence:", seq)
    return render_template('topDiseasepred.html', seq=seq)


@app.route("/input_lncRNA",methods=['POST'])
def input_lncRNA():
    global selected_lncRNA,selected_diseases,selected_target,seq,selected_dis
    selected_lncRNA=None
    selected_diseases=[]
    selected_target=[]
    seq=None
    selected_dis=None
    selected_lncRNA = request.form.get('lncRNA')
    # Check if the value was selected from the dropdown or typed manually
    if selected_lncRNA == '':
        selected_lncRNA = request.form.get('custom-lncRNA')
    show_prediction_button = selected_lncRNA in lncRNA_names
    print(f"Selected or typed lncRNA: {selected_lncRNA}")

    # Return JSON response instead of directly rendering the template
    return jsonify(selected_lncRNA=selected_lncRNA, show_prediction_button=show_prediction_button)

def check_doid(dis):
    d=None
    flag=False
    base_url = "https://www.ebi.ac.uk/ols/api"
    search_url = f"{base_url}/search?q={dis}&ontology=doid"

    response = requests.get(search_url)
    data = response.json()

    if 'response' in data and 'docs' in data['response']:
        docs = data['response']['docs']
        try:
            if docs:
                doid = docs[0]['obo_id']
                term_label = docs[0]['label']
                d=doid
                flag=True
                # print(f"Disease Name: {disease_name}")
                # print(f"DOID: {doid}")
                    
                # print(f"Term Label: {term_label}")
            
        except:
                print("Doid not found")

    else:
        print("Doid not found")
    
    return d,flag


@app.route("/input_disease_selected",methods=['POST'])
def input_disease_selected():
    global selected_dis,selected_lncRNAs_list,selected_genes_dis,dis_doid,selected_lncRNA
    selected_dis=None
    selected_lncRNAs_list=[]
    selected_genes_dis=[]
    dis_doid=None
    selected_lncRNA=None
    selected_dis=request.form.get('disease')
    if selected_dis == '':
        selected_dis=request.form.get('custom-disease')
    
    print(f"Selected disease: {selected_dis}")
    show_button1=False
    show_button2=False
    if selected_dis in diseases:
        dis_doid=dis_doid_dic[selected_dis]
        show_button1=True
    else:
        d,flag=check_doid(selected_dis)
        if flag==True:
            dis_doid=d
        else:
            show_button2=True
            dis_doid=request.form.get('doid')

    print("Doid for this disease: ",dis_doid)
    return jsonify(selected_dis=selected_dis,show_button1=show_button1,show_button2=show_button2,dis_doid=dis_doid)

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/applications', methods=['GET'])
def applications():
    return render_template('applications.html')


@app.route('/get_doid', methods=['GET'])
def get_doid():
    return jsonify({'doid': dis_doid})

@app.route('/submit', methods=['POST'])
def submit():
    # Process form data here if needed
    # Redirect to result.html after form submission
    print("Hello, world!")
    # return redirect('/result.html')
    printlist(seq)
    return redirect(url_for('result'))

@app.route('/contribute', methods=['GET'])
def contribute_page():
    return render_template('contribute.html')

@app.route('/evidences', methods=['GET'])
def evidences():
    return render_template('evidences.html')


@app.route('/about-us', methods=['GET'])
def about_us():
    return render_template('about-us.html')


@app.route('/submit_contribution', methods=['POST'])
def submit_contribution():
    text_data = request.form['text_data']
    file_data = request.files['file_data']
    
    # Process or print the received data
    print("Text Data:", text_data)
    print("File Name:", file_data.filename)
    
    # Additional processing or saving the file can be done here
    
    return render_template('contribute.html')


@app.route('/result', methods=['GET','POST'])
def result():
    check_item=None
    if selected_dis==None:
        check_item="lncRNA"

    else:
        check_item="disease"
    return render_template('result.html',check_item=check_item,information_dic=information_dic,selected_item=selected_item,selected_list=selected_list)

def find_simi(similarity_stats):
    ans=None
    print(similarity_stats)
    if len(similarity_stats)==len(lncRNA_names):
        y=np.array(similarity_stats)
        idx=y.argsort()[-10:][::-1]
        lnc=lncRNA_names[idx]
        find_info(selected_lncRNA,lnc,0)
    else:
        y=np.array(similarity_stats)
        idx=y.argsort()[-10:][::-1]
        lnc=diseases[idx]
        find_info(selected_dis,lnc,1)

@app.route('/goToSimilarity', methods=['POST'])
def goToSimilarity():
    
    return redirect(url_for('similarity'))


@app.route('/similarity', methods=['GET','POST'])
def similarity():
    check_item=None
    if selected_dis==None:
        check_item="lncRNA"

    else:
        check_item="disease"

    find_simi(similarity_stats)
    return render_template('show_similarity.html',check_item=check_item,information_dic=information_dic,selected_item=selected_item,selected_list=selected_list)


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/toplncRNApred',methods=['GET','POST'])
def toplncRNApred():
    global selected_dis,selected_lncRNAs_list,selected_genes_dis,dis_doid
    print("I am the best")
    return render_template('toplncRNApred.html',diseases=diseases,lncRNA_names=lncRNA_names,disease_genes=disease_genes)

@app.route('/topDiseasepred', methods=['GET', 'POST'])
def topDiseasepred():
    global selected_lncRNA, sequence,selected_diseases
    return render_template('topDiseasepred.html', lncRNA_names=lncRNA_names, disease_list=diseases,target_list=target_list)



def gKernel(nl, nd, inter_lncdis):
    # Compute Gaussian interaction profile kernel of lncRNAs
    sl = np.zeros(nl)
    for i in range(nl):
        sl[i] = np.linalg.norm(inter_lncdis[i, :]) ** 2
    gamal = nl / np.sum(sl) * 1
    pkl = np.zeros((nl, nl))
    for i in range(nl):
        for j in range(nl):
            pkl[i, j] = float(np.exp(-gamal * (np.linalg.norm(inter_lncdis[i, :] - inter_lncdis[j, :])) ** 2))

    # Compute Gaussian interaction profile kernel of diseases
    sd = np.zeros(nd)
    for i in range(nd):
        sd[i] = np.linalg.norm(inter_lncdis[:, i]) ** 2
    gamad = nd / np.sum(sd) * 1
    pkd = np.zeros((nd, nd))
    for i in range(nd):
        for j in range(nd):
            pkd[i, j] = float(np.exp(-gamad * (np.linalg.norm(inter_lncdis[:, i] - inter_lncdis[:, j])) ** 2))


    return pkl, pkd


def create_similarity_matrix(association_matrix):
    # Calculate cosine similarity between lncRNAs
    similarity_matrix = cosine_similarity(association_matrix)
    return similarity_matrix

import numpy as np
import numpy.linalg as LA
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import csv
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf


def topk(ma1,gip,nei):
    for i in range(ma1.shape[0]):
        ma1[i,i]=0
        gip[i,i]=0
    ma=np.zeros((ma1.shape[0],ma1.shape[1]))
    for i in range(ma1.shape[0]):
        if sum(ma1[i]>0)>nei:
            yd=np.argsort(ma1[i])
            ma[i,yd[-nei:]]=1
            ma[yd[-nei:],i]=1
        else:
            yd=np.argsort(gip[i])
            ma[i,yd[-nei:]]=1
            ma[yd[-nei:],i]=1
    return ma

def adj_matrix(lnc_dis_matrix, lnc_matrix, dis_matrix):
    mat1 = np.hstack((lnc_matrix, lnc_dis_matrix))
    mat2 = np.hstack((lnc_dis_matrix.T, dis_matrix))
    return np.vstack((mat1, mat2))

def compute_known_features():
    global l1,l3,d1,d2,result_lnc,result_dis
    d2=gdi[:,0:468]
    d1=gdi[:,468:]
    encoded_lnc=encoder_lnc.predict(lncRNA_feature)
    encoded_dis=encoder_dis.predict(gdi)
    lnc1=topk(l1,result_lnc,10)
    lnc2=topk(l3,result_lnc,10)
    print("harshu")
    print(d1.shape)
    dis1=topk(d1,result_dis,10)
    dis2=topk(d2,result_dis,10)
    adj1=adj_matrix(lda,lnc1,dis1)
    adj2=adj_matrix(lda,lnc2,dis2)
    features=np.vstack((encoded_lnc,encoded_dis))
    features_tensor = torch.Tensor(features)
    adj1t = torch.Tensor(adj1)
    adj2t = torch.Tensor(adj2)
    GCN_node1.eval()
    GCN_node2.eval()
    node_output1 = GCN_node1(features_tensor, adj1t).detach().numpy()
    node_output2 = GCN_node2(features_tensor, adj2t).detach().numpy()
    return node_output1,node_output2

    # x1=x[:,0:4458*2]
    # similarity_stats=x[:,0:4458].tolist()[0]
    # x2=x[:,4458*2:3*4458]
    # x3=lda[idx]
    # # result_lnc, result_dis=gKernel(len(lncRNA_names), len(diseases),lda)
    # x4=result_lnc[idx]
    # d1=gdi[:,468:468*2]
    # d2=gdi[:,0:468]
    # xx1 = np.concatenate((x1.repeat(d1.shape[0], axis=0), d1), axis=1)
    # xx2 = np.concatenate((x2.repeat(d2.shape[0], axis=0), d2), axis=1)
    # xx3 = np.concatenate((x3.repeat(lda.T.shape[0], axis=0),lda.T), axis=1)
    # xx4 = np.concatenate((x4.repeat(result_dis.shape[0], axis=0),result_dis), axis=1)
    # return xx1,xx2,xx3,xx4

    
# def compute_features_diseases(dis):
#     global similarity_stats
#     idx=np.where(diseases==dis)
#     x=gdi[idx]
#     x1=x[:,468:2*468]
#     similarity_stats=x1.tolist()[0]
#     x2=x[:,0:468]
#     # x1=x1.reshape(-1,1).T
#     # x2=x2.reshape(-1,1).T
#     x3=lda.T[idx]
#     # result_lnc, result_dis=gKernel(len(lncRNA_names), len(diseases),lda)
#     x4=result_dis[idx]
#     d1=lncRNA_feature[:,0:4458*2]
#     d2=lncRNA_feature[:,4458*2:3*4458]
    
#     xx1 = np.concatenate((d1,x1.repeat(d1.shape[0], axis=0)), axis=1)
#     xx2 = np.concatenate((d2,x2.repeat(d2.shape[0], axis=0)), axis=1)
#     xx3 = np.concatenate((lda,x3.repeat(lda.shape[0], axis=0)), axis=1)
#     xx4 = np.concatenate((result_lnc,x4.repeat(result_lnc.shape[0], axis=0)), axis=1)
#     return xx1,xx2,xx3,xx4


def printlist(l):
    global g2l,l2g,similarity_stats,d1,d2,l1,l2,l3
    print(selected_lncRNA)
    print(l)
    print(selected_diseases)
    print(selected_target)
    print(selected_dis)
    print(selected_genes_dis)
    print(dis_doid)
    print(selected_lncRNAs_list)
    seq=l
    if selected_dis==None:
        if selected_lncRNA in lncRNA_names and seq==None and selected_diseases==[] and selected_target==[]:
          
            node_output1,node_output2=compute_known_features()
            idx=np.where(lncRNA_names==selected_lncRNA)
            x1=node_output1[idx]
            x2=node_output2[idx]
            dd1=node_output1[4458:4458+468]
            dd2=node_output2[4458:4458+468]
            xx1 = np.concatenate((dd1,x1.repeat(dd1.shape[0], axis=0)), axis=1)
            xx2 = np.concatenate((dd2,x2.repeat(dd2.shape[0], axis=0)), axis=1)
            y1=[]
            xx1s=scaler1.transform(xx1)
            for model in base_models1:
                y1.append(base_models1[model].predict_proba(xx1s)[:, 1])
            y1=np.array(y1)
            print(y1)
            yy1=meta_model1.predict_proba(y1.T)[:, 1]
            y2=[]
            xx2s=scaler2.transform(xx2)
            for model in base_models2:
                y2.append(base_models2[model].predict_proba(xx2s)[:, 1])
            y2=np.array(y2)


            yy2=meta_model2.predict_proba(y2.T)[:, 1]

            # y1 = model1.predict_proba(x1)[:, 1]
            # y2 = model2.predict_proba(x2)[:, 1]
            # y3 = model3.predict_proba(x3)[:, 1]
            # y4 = model4.predict_proba(x4)[:, 1]
            y=(yy1)+(0*yy2)
            print("burkit: ", y[386])
            print(y)
            top_10_indices = y.argsort()[-10:][::-1]
            l2d=(diseases[top_10_indices])
            find_info(selected_lncRNA,l2d,0)
            print(l2d)
            return " "
        
        else:
            selected_sequence=seq
            f1 = np.zeros((1, len(sequences)))

            # Create a dictionary to store the k-mer frequencies for each lncRNA ID
            
            kmer_frequencies = {}
            lncRNAseq=sequences
            k = 3
            possible_kmers = [''.join(p) for p in itertools.product('ATCGatcg', repeat=k)]
            
            # Iterate over the lncRNA sequences
            for lncRNA_id, sequence in lncRNAseq.items():
                # Initialize a dictionary to store the k-mer frequencies for the current sequence
                sequence_kmers = {kmer: 0 for kmer in possible_kmers}

                # Iterate over the sequence with a sliding window of size k
                for i in range(len(sequence) - k + 1):
                    kmer = sequence[i:i+k]
                    sequence_kmers[kmer] += 1

                # Store the k-mer frequencies for the current lncRNA ID
                kmer_frequencies[lncRNA_id] = sequence_kmers
            # Store the k-mer frequencies for the current lncRNA ID
            
            sequence_kmers = {kmer: 0 for kmer in possible_kmers}

            # Iterate over the sequence with a sliding window of size k
            for i in range(len(selected_sequence) - k + 1):
                kmer = selected_sequence[i:i+k]
                sequence_kmers[kmer] += 1

            # kmer_frequencies[selected_lncRNA] = sequence_kmers
            for j, (lncRNA_id2, sequence_kmers2) in enumerate(kmer_frequencies.items()):
                distance = euclidean(list(sequence_kmers.values()), list(sequence_kmers2.values()))
                similarity = 1 / (1 + distance)
                # print(j)
                f1[0, j] = similarity
                
            print("step 1 done")

            f2 = np.zeros((1, len(sequences)))
            
            for j, (lnc, seq1) in enumerate(sequences.items()):
                # Calculate the Levenshtein distance between the sequences
                distance = Levenshtein.distance(selected_sequence, seq1)
                similarity = 1 / (1 + distance)
                # Store the distance in the feature matrix
                f2[0, j] = similarity
            similarity_stats=f2.tolist()[0]
            x1=np.hstack((f1,f2))
            
            print("step 2 done")

            # xx1 = np.concatenate((x1.repeat(ddsim.shape[0], axis=0), ddsim), axis=1)



            target_input = selected_target
            unique_targets = list(set(targetNames))
            genes_found = []
            idx = []

            for idx_val, target in enumerate(unique_targets):
                if target in target_input:
                    genes_found.append(target)
                    idx.append(idx_val)


            tar_lnc = np.zeros(len(unique_targets))
            tar_lnc[idx]=1
            new_lncTarget=np.vstack((lncTarget,tar_lnc))
            new_lncTargetFeature=create_similarity_matrix(new_lncTarget)
            x2=new_lncTargetFeature[-1]
            f3=x2
            x2=x2[:-1]
            x2=x2.reshape(-1,1).T
            ll=np.hstack((x1,x2))
            ll=np.vstack((lncRNA_feature,ll))
            # xx2 = np.concatenate((x2.repeat(dis_genes.shape[0], axis=0),dis_genes), axis=1)

            print("step 3 done")


            disease_list = selected_diseases
            disease_found = []
            disidx = []

            for idx, disease in enumerate(diseases):
                if disease in disease_list:
                    disease_found.append(disease)
                    disidx.append(idx)

            dis_lnc=np.zeros(len(diseases))
            dis_lnc[disidx]=1
            x3=dis_lnc
            x3=x3.reshape(-1,1).T
            # xx3 = np.concatenate((x3.repeat(lda.T.shape[0], axis=0),lda.T), axis=1)

            new_lda=np.vstack((lda,dis_lnc))
        
            gip_lnc,gip_dis=gKernel(len(lncRNA_names)+1, len(diseases),new_lda)
            x4=gip_lnc[-1]
            x4=x4[:-1]
            x4 = x4.reshape(-1, 1).T
            # xx4 = np.concatenate((x4.repeat(gip_dis.shape[0], axis=0), gip_dis), axis=1)
            print("step 4 done")
            print(ll)
            encoded_lnc=encoder_lnc.predict(ll)           
            encoded_dis=encoder_dis.predict(gdi)
            f4=np.append(f1,1)
            l1=ll[:,0:4458]
            l1=np.hstack((l1,f4.reshape(-1,1)))
            l3=ll[:,4458*2:4458*3]
            l3=np.hstack((l3,f3.reshape(-1,1)))
            print(l1.shape)
            print(gip_lnc.shape)
            lnc1=topk(l1,gip_lnc,10)
            lnc2=topk(l3,gip_lnc,10)
            d1=gdi[:,0:468]
            d2=gdi[:,468:]
            dis1=topk(d1,gip_dis,10)
            dis2=topk(d2,gip_dis,10)
            adj1=adj_matrix(new_lda,lnc1,dis1)
            adj2=adj_matrix(new_lda,lnc2,dis2)
            features=np.vstack((encoded_lnc,encoded_dis))
            print(features.shape)
            print(adj1.shape)
            features_tensor = torch.Tensor(features)
            adj1t = torch.Tensor(adj1)
            adj2t = torch.Tensor(adj2)
            GCN_node1.eval()
            GCN_node2.eval()
            node_output1 = GCN_node1(features_tensor, adj1t).detach().numpy()
            node_output2 = GCN_node2(features_tensor, adj2t).detach().numpy()
           
            # print(node_output1.shape)

            x1=node_output1[4458].reshape(-1,1).T
            x2=node_output2[4458].reshape(-1,1).T
            dd1=node_output1[4458:4458+468]
            dd2=node_output2[4458:4458+468]
            xx1 = np.concatenate((dd1,(x1.repeat(dd1.shape[0], axis=0))), axis=1)
            xx2 = np.concatenate((dd2,(x2.repeat(dd2.shape[0], axis=0))), axis=1)
            y1=[]
            xx1s=scaler1.transform(xx1)
            for model in base_models1:
                y1.append(base_models1[model].predict_proba(xx1s)[:, 1])
            y1=np.array(y1)
            print(y1)
            yy1=meta_model1.predict_proba(y1.T)[:, 1]
            y2=[]
            xx2s=scaler2.transform(xx2)
            for model in base_models2:
                y2.append(base_models2[model].predict_proba(xx2s)[:, 1])
            y2=np.array(y2)


            yy2=meta_model2.predict_proba(y2.T)[:, 1]

            y=(yy1)+(0*yy2)
            print(y)
            top_10_indices = y.argsort()[-10:][::-1]
            l2d=(diseases[top_10_indices])
            find_info(selected_lncRNA,l2d,0)
            print(l2d)
            return " "
            

    else:
        if selected_dis in diseases and selected_lncRNAs_list==[] and selected_genes_dis==[]:
            node_output1,node_output2=compute_known_features()
            idx=np.where(diseases==selected_dis)
            x1=node_output1[idx[0]+4458].reshape(-1,1).T
            x2=node_output2[idx[0]+4458].reshape(-1,1).T
            
            print("udit")
            dd1=node_output1[0:4458]
            dd2=node_output2[0:4458]
            xx1 = np.concatenate((dd1,x1.repeat(dd1.shape[0], axis=0)), axis=1)
            xx2 = np.concatenate((dd2,x2.repeat(dd2.shape[0], axis=0)), axis=1)
            y1=[]
            xx1s=scaler1.transform(xx1)
            for model in base_models1:
                y1.append(base_models1[model].predict_proba(xx1s)[:, 1])
            y1=np.array(y1)
            print(y1)
            yy1=meta_model1.predict_proba(y1.T)[:, 1]
            y2=[]
            xx2s=scaler2.transform(xx2)
            for model in base_models2:
                y2.append(base_models2[model].predict_proba(xx2s)[:, 1])
            y2=np.array(y2)


            yy2=meta_model2.predict_proba(y2.T)[:, 1]

            # y1 = model1.predict_proba(x1)[:, 1]
            # y2 = model2.predict_proba(x2)[:, 1]
            # y3 = model3.predict_proba(x3)[:, 1]
            # y4 = model4.predict_proba(x4)[:, 1]
            y=(yy1)+(0*yy2)
            top_10_indices = y.argsort()[-10:][::-1]
            d2l=(lncRNA_names[top_10_indices])
            find_info(selected_dis,d2l,1)


        else:
            
            
            file_path = 'create_x1_forDisease.csv'
            d=np.array(doids)
            d=np.append(d,dis_doid)
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                for row in d:
                    writer.writerow([row])
            
            run_r_code()

            ddsim_data=pd.read_csv("ddsim_target.csv")
            target_ddsim=ddsim_data.iloc[:, 0] 
            target_ddsim=list(target_ddsim)
            similarity_stats=target_ddsim
            print("step 1 is done")
            x1=np.array(target_ddsim) 
            dd1=np.vstack((d1,x1))
            a1=np.append(x1,1)
            a1=np.hstack((dd1,a1.reshape(-1,1)))
            x1=x1.reshape(-1,1).T

            # x=lncRNA_feature
            # d1=x[:,468:2*468]
            # d2=x[:,0:468]
            # x3=lda.T[idx]
            # result_lnc, result_dis=gKernel(len(lncRNA_names), len(diseases),lda)
            # x4=result_dis[idx]
            # d1=lncRNA_feature[:,0:4458*2]
            # d2=lncRNA_feature[:,4458*2:3*4458]
            # xx1 = np.concatenate((d1,x1.repeat(d1.shape[0], axis=0)), axis=1)


            target_input = selected_genes_dis
            unique_targets = list(disease_genes)
            genes_found = []
            idx = []

            for idx_val, target in enumerate(unique_targets):
                if target in target_input:
                    genes_found.append(target)
                    idx.append(idx_val)


            tar_lnc = np.zeros(len(unique_targets))
            tar_lnc[idx]=1
            new_lncTarget=np.vstack((inter_disease_genes,tar_lnc))
            new_lncTargetFeature=create_similarity_matrix(new_lncTarget)
            x2=new_lncTargetFeature[-1]
            x2=x2[:-1]
            dd2=np.vstack((d2,x2))
            a2=np.append(x2,1)
            a2=np.hstack((dd2,a2.reshape(-1,1)))
            x2=x2.reshape(-1,1).T

            dd=np.hstack((x1,x2))
            dd=np.vstack((gdi,dd))
            # xx2 = np.concatenate((d2,x2.repeat(d2.shape[0], axis=0)), axis=1)
            disease_list = selected_lncRNAs_list
            disease_found = []
            disidx = []

            for idx, disease in enumerate(lncRNA_names):
                if disease in disease_list:
                    disease_found.append(disease)
                    disidx.append(idx)

            dis_lnc=np.zeros(len(lncRNA_names))
            dis_lnc[disidx]=1
            x3=dis_lnc
            x3=x3.reshape(-1,1).T
            # xx3 = np.concatenate((lda,x3.repeat(lda.shape[0], axis=0)), axis=1)
            new_lda=np.hstack((lda,x3.T))
        
            gip_lnc,gip_dis=gKernel(len(lncRNA_names), len(diseases)+1,new_lda)
            x4=gip_dis[-1]
            x4=x4[:-1]
            x4 = x4.reshape(-1, 1).T

            # xx4 = np.concatenate((gip_lnc,x4.repeat(gip_lnc.shape[0], axis=0)), axis=1)
            
            encoded_lnc=encoder_lnc.predict(lncRNA_feature)           
            encoded_dis=encoder_dis.predict(dd)
            dd1=dd[:,0:468]
            dd2=dd[:,468:]

            lnc1=topk(l1,gip_lnc,10)
            lnc2=topk(l3,gip_lnc,10)
            dis1=topk(a1,gip_dis,10)
            dis2=topk(a2,gip_dis,10)
            adj1=adj_matrix(new_lda,lnc1,dis1)
            adj2=adj_matrix(new_lda,lnc2,dis2)
            features=np.vstack((encoded_lnc,encoded_dis))
            features_tensor = torch.Tensor(features)
            adj1t = torch.Tensor(adj1)
            adj2t = torch.Tensor(adj2)
            GCN_node1.eval()
            GCN_node2.eval()
            node_output1 = GCN_node1(features_tensor, adj1t).detach().numpy()
            node_output2 = GCN_node2(features_tensor, adj2t).detach().numpy()

            x1=node_output1[4458+468].reshape(-1,1).T
            x2=node_output2[4458+468].reshape(-1,1).T
            dd1=node_output1[0:4458]
            dd2=node_output2[0:4458]
            xx1 = np.concatenate((dd1,(x1.repeat(dd1.shape[0], axis=0))), axis=1)
            xx2 = np.concatenate((dd2,(x2.repeat(dd2.shape[0], axis=0))), axis=1)
            y1=[]
            xx1s=scaler1.transform(xx1)
            for model in base_models1:
                y1.append(base_models1[model].predict_proba(xx1s)[:, 1])
            y1=np.array(y1)
            print(y1)
            yy1=meta_model1.predict_proba(y1.T)[:, 1]
            y2=[]
            xx2s=scaler2.transform(xx2)
            for model in base_models2:
                y2.append(base_models2[model].predict_proba(xx2s)[:, 1])
            y2=np.array(y2)


            yy2=meta_model2.predict_proba(y2.T)[:, 1]

            y=(yy1)+(0*yy2)
            print(y)
            top_10_indices = y.argsort()[-10:][::-1]
            l2d=(lncRNA_names[top_10_indices])
            find_info(selected_dis,l2d,0)
            print(l2d)
            return " "

    return " "


if __name__ == '__main__':
    app.run(debug=True)
