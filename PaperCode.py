# Computer Science Paper
# Jiayao Hu (SNR: 523775)

# Import packages
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from math import comb
import sys
from sklearn.cluster import AgglomerativeClustering
import itertools as it

np.random.seed(1)

# Load data
f = open("TVs-all-merged.json")
data = json.load(f)

def get_ID_title(data):
    modelID, title = [], []
    for i in data:
        for j in data[i]:
            modelIDs_list = j['modelID']
            modelID.append(modelIDs_list)
    
            titles_list = j['title']
            title.append(titles_list)
    return modelID, title

def inches(string):
    a = ['inch', 'Inch', 'inches', 'Inches', '"', '-inch', '-Inch']
    for j in a:
        if j in string:
            string = string.replace(j, 'inch')
    return string

def hertz(string):
    a = ['Hz', 'hz', 'Hertz', 'hertz', 'HZ', '-hz', '-Hz']
    for j in a:
        if j in string:
            string = string.replace(j, 'hz')
    return string

def interpunction(string): 
    chars = "!@#$%^&*_+~`;<>/|()[]?-'" 
    for j in chars:
        if j in string:
            string = string.replace(j, '')
    return string

def capitalization(string):
    return string.lower()

def remove(string):
    shop_name = ['amazon', 'newegg.com', 'best buy', 'thenerds']
    for j in shop_name:
        if j in string:
            string = string.replace(j, '')
    return string

def del_white_spaces(string):
    s = string.strip()
    j = s.find('  ')
    while j >= 0:
        s = s.replace('  ', ' ')
        j = s.find('  ')
    return s

def clean_title(title):
    cleaned_titles = []
    for i in title:
        inch = inches(i)
        Hz = hertz(inch)
        inter = interpunction(Hz)
        lower = capitalization(inter)
        removed = remove(lower)
        white = del_white_spaces(removed)
        cleaned_titles.append(white)
    return cleaned_titles

def bootstrapping(modelID, title):
    title = clean_title(title)
    new_modelID, test_modelID = [], []
    
    new_title, test_title = [], []
    
    bootstrap = []
    for loop in range(len(modelID)):
        bootstrap.append(np.random.randint(1,len(modelID)))
    new_list = list(set(bootstrap))
    for index in range(len(modelID)):
        if index in new_list:
            new_modelID.append(modelID[index])
            new_title.append(title[index])
        
        elif index not in new_list:
            test_modelID.append(modelID[index])
            test_title.append(title[index])

    return new_modelID, new_title, test_modelID, test_title

# Model Words
def model_words(lists):
    MW_title = []
    for i in lists:
        for j in i.split():
            match = re.search('([a-z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-z0-9]*)', j)
            if match:
                MW_title.append(match.group()) 
    return list(set(MW_title))

def binary_mat(MW, cleaned):
    input_matrix = np.zeros((len(MW), len(cleaned)), dtype=int)
    for i in range(len(cleaned)):
        for j in range(len(MW)):
            if MW[j] in cleaned[i]:
                input_matrix[j,i] = 1
            else:
                input_matrix[j,i] = 0
    return input_matrix

def isPrime(x):
    counter = x + 2
    for i in range(2, counter):
        if counter%i == 0:
            counter += 1
            i = 2
    return counter 

def hashing(a, b, x, prime):  
    hash_value = (a + b*x) % prime
    return hash_value

def list_a_b(N):   
    list_a, list_b = [], []
    for i in range(N):
        list_a.append(np.random.randint(1,N))
        list_b.append(np.random.randint(1,N))
    return list_a, list_b

# Minhashing
def minhashing_sig(input_matrix, N): 
    n_input, m_input = np.shape(input_matrix)
    prime = isPrime(n_input)
    sig_mat = np.full((N, m_input), np.inf)
    n_sig, m_sig = np.shape(sig_mat)
    # hash value for each row and each hash function
    hash_values = np.zeros((N, n_input), dtype=int) 
    a, b = list_a_b(N)
    for row in range(n_input):  
        for hash_func in range(len(a)):
            hash_val = hashing(a[hash_func], b[hash_func], row, prime)
            # Save the hash values     
            hash_values[hash_func,row] = hash_val
    
    # Signature Matrix      
    for row in range(n_input):        
        for col in range(m_sig):
            # Find the first 1
            if input_matrix[row,col] == 1:                             
    
                for value in range(len(hash_values)):
                    # Check if the hash value is smaller than the M matrix
                    if hash_values[value,row] < sig_mat[value,col]:           
                        sig_mat[value,col] = hash_values[value,row]
    return sig_mat                     

# Locality Sensitive Hashing (LSH)
def bandrow(sig_mat):  
    n_sig, m_sig = np.shape(sig_mat)  
    threshold = 0.5
    
    # Choose b and r, where n_sig = b * r
    band_row = []
    for b in range(1, n_sig+1):
        for r in range(1, n_sig+1):
            if np.isclose(n_sig, b * r, atol = 5):
                band_row.append([b,r])
                
    # The threshold corresponding to the b and r
    t_list = []
    for i in band_row:
        b = i[0]
        r = i[1]
        t = (1/b)**(1/r)
        t_list.append(t)
    
    # Select the b and r closest to the threshold
    selected_b_r = []
    for thres in t_list:
        if np.isclose(threshold, thres, atol=0.1):
            b_r = band_row[t_list.index(thres)]
            selected_b_r.append(b_r)
    # The threshold is chosen to be low such that false negatives are reduced 
    # as much as possible since false negatives are worse to recover from.
    return selected_b_r, t_list

def bucket_hash(selected_b_r, sig_mat):
    n_sig, m_sig = np.shape(sig_mat)
    
    all_bucket_hash = []
    for be in selected_b_r:
        bucket_hash = [] 
        b = be[0]
        r = be[1]
        for band in range(1, b+1):
            one_band = []
            # Consider the 1st, ..., b-th band with r rows.
            if band == b:
                # take all remaining rows for the last band
                mat = sig_mat[r*(band-1) :]
            else:
                mat = sig_mat[r*(band-1) : r*band]
            for col in range(m_sig):
                hash_value = ''.join(str(i) for i in mat[:,col])
                string = hash_value.replace('.', '')
                one_band.append(string) 
            bucket_hash.append(one_band)
        all_bucket_hash.append(bucket_hash)
    return all_bucket_hash

def all_bucks(all_bucket_hash):
    all_buckets = []
    copy_bucket_hash = all_bucket_hash.copy()
    for hasher in copy_bucket_hash:
        buckets = []
        for band in hasher:
            pairs_band = []
            n_bucket_hash, m_bucket_hash = np.shape(hasher)
            for s in range(m_bucket_hash-1):
                if band[s] == 'pass':
                    same = []
                elif band[s] != 'pass':
                    same = [s]
                    for s2 in range(s+1,m_bucket_hash):
                        if band[s] == band[s2]:
                            same.append(s2)
                            band[s2] = band[s2].replace(band[s2], 'pass')
                pairs_band.append(same)
            if band[m_bucket_hash-1] != 'pass':
                pairs_band.append([m_bucket_hash-1])
            elif band[m_bucket_hash-1] == 'pass':
                pairs_band.append([])
            buckets.append(pairs_band)
        all_buckets.append(buckets)
    return all_buckets

def pairs(all_buckets, N):
    all_pairs = []  
    for bucket in all_buckets: 
        n_bucket = len(bucket)
        m_bucket = N
        pairs = [[i] for i in range(m_bucket)]
        for col in range(m_bucket):
          for row in range(n_bucket):
              if bucket[row][col] == pairs[col]:
                  continue
              elif bucket[row][col] != pairs[col]:
                  for num in bucket[row][col]:
                      if num in pairs[col]:
                          continue
                      elif num not in pairs[col]:
                          pairs[col].append(num)  
        all_pairs.append(pairs)
    return all_pairs

def candidates(all_pairs, N):
    all_cands = []
    for pair_list in all_pairs:
        cand_pair = np.zeros((N, N), dtype = int)
        n_cand_pairs, m_cand_pairs = np.shape(cand_pair)
        for prod in pair_list:
            if len(prod) > 1:
                for same_prod in prod:
                    for same_prod2 in prod:
                        if same_prod != same_prod2:
                            cand_pair[same_prod, same_prod2] = 1
                            cand_pair[same_prod2, same_prod] = 1
        all_cands.append(cand_pair)
    return all_cands
  
def jaccard_dissim(A, B):
    numerator = 0
    denominator = 0
    for i in range(len(A)):
        if A[i] == B[i] and A[i] != 0 and B[i] != 0:
            # Intersection of two columns
            numerator += 1
        
        if A[i] != 0 or B[i] != 0:
            # The common elements of two columns
            denominator += 1

    # This gives the Jaccard similarity
    similarity = numerator/denominator
    # 1 minus the similarity gives the dissimilarity
    return 1 - similarity

def dissimilarity(all_cands, input_matrix, N):
    all_dissim_mat = []
    for mat in all_cands:
        dissim_mat = np.empty((N, N))
        for row in range(len(mat)):
            for col in range(len(mat)):
                if mat[row,col] == 0 and row != col:
                    dissim_mat[row,col] = sys.maxsize
                elif mat[row,col] == 1:
                    dissim_mat[row,col] = jaccard_dissim(list(input_matrix[:,row]),list(input_matrix[:,col]))
        all_dissim_mat.append(dissim_mat)
    return all_dissim_mat

def run_bootstrap(dataset):
    modelID, title = get_ID_title(data)
    bootstrap_ID, bootstrap_title, boot_testID, boot_testtitle = bootstrapping(modelID, title)
    unique_MW = model_words(bootstrap_title)
    if '' in unique_MW:
        unique_MW.remove('')
    unique_MW_test = model_words(boot_testtitle)
    if '' in unique_MW_test:
        unique_MW_test.remove('')
    input_matrix = binary_mat(unique_MW, bootstrap_title)
    input_matrix_test = binary_mat(unique_MW_test, boot_testtitle)
    
    signature_mat = minhashing_sig(input_matrix, round(len(unique_MW)/2)) 
    selected_b_r, t_list = bandrow(signature_mat)
    all_bucket_hash = bucket_hash(selected_b_r, signature_mat)
    all_buckets = all_bucks(all_bucket_hash)
    
    signature_mat_test = minhashing_sig(input_matrix_test, round(len(unique_MW_test)/2)) 
    selected_b_r_test, t_list_test = bandrow(signature_mat_test)
    all_bucket_hash_test = bucket_hash(selected_b_r_test, signature_mat_test)
    all_buckets_test = all_bucks(all_bucket_hash_test)
    
    all_pairs = pairs(all_buckets, len(bootstrap_title))
    all_cands = candidates(all_pairs, len(bootstrap_title))

    all_pairs_test = pairs(all_buckets_test, len(boot_testtitle))
    all_cands_test = candidates(all_pairs_test, len(boot_testtitle))

    training = [bootstrap_ID, bootstrap_title, all_cands, input_matrix]
    test = [boot_testID, boot_testtitle, all_cands_test, input_matrix_test]
    return training, test

def clustering(all_dissim_matrix, modelID, t):
    all_clusters = []
    for mat in all_dissim_matrix:
        cluster = AgglomerativeClustering(n_clusters=None, affinity="precomputed", 
                                           linkage="complete", distance_threshold=t).fit(mat)
        all_clusters.append(cluster.labels_)
    
    all_clusts = []
    for label in all_clusters:
        cluster_dict = {}
        for lab in range(label.max()+1):
            cluster_prods = list(np.where(label == lab))
            cluster_dict[lab] = list(modelID[i] for i in cluster_prods[0])
        all_clusts.append(cluster_dict)
    return all_clusts

def real_dups(ID):
    real_duplicates = 0
    for prod1 in range(len(ID)-1):
        for prod2 in range(prod1+1, len(ID)):
            if ID[prod1] == ID[prod2]:
                real_duplicates += 1
    return real_duplicates

def pair_compare_LSH(all_cands, ID):
    number_compare_LSH, pairs_LSH = [], []
    for matrix in all_cands:
        number_compare_LSH.append(np.count_nonzero(matrix)/2)
        pair_LSH = 0
        for row in range(np.shape(matrix)[0]-1):
            for col in range(row+1, np.shape(matrix)[1]):
                if matrix[row,col] == 1 and \
                ID[row] == ID[col]:
                    pair_LSH += 1
        pairs_LSH.append(pair_LSH)
    return number_compare_LSH, pairs_LSH

def LSH_measures(pairs, compares, real, max_comp):
    PC, PQ, frac_LSH, F1_star = [], [], [], []
    
    for i in range(len(pairs)):
        PC.append(pairs[i]/real)
        PQ.append(pairs[i]/compares[i])
        frac_LSH.append(compares[i]/max_comp)
        F1_star.append(2*PC[i]*PQ[i]/(PC[i]+PQ[i]))
    
    return PC, PQ, frac_LSH, F1_star

def clust_measures(all_clusters, real_duplicates):
    precision, recall, frac_clust, F_1 = [], [], [], []
    
    # Performance clustering
    number_compare_clust = []
    pairs_clust = []
    for cluster in all_clusters:
        compare = 0
        pair_clust = 0
        for i in range(len(cluster)):
            if len(cluster[i]) > 1:
                compare += comb(len(cluster[i]),2)
            for id1 in range(len(cluster[i])-1):
                for id2 in range(id1+1, len(cluster[i])):
                    if cluster[i][id1] == cluster[i][id2]:
                        pair_clust += 1
        pairs_clust.append(pair_clust)
        number_compare_clust.append(compare)
        
    f1s = []
    for i in range(len(pairs_clust)):
        recall.append(pairs_clust[i]/real_duplicates)
        precision.append(comb(pairs_clust[i],2)/number_compare_clust[i])
        frac_clust.append(number_compare_clust[i]/max_number_compare)
        f1s.append((2*precision[i]*recall[i])/(precision[i]+recall[i]))
    F_1.append(f1s)
    
    return pairs_clust, number_compare_clust, recall, precision, frac_clust, F_1

def high_f1(F_1, thres_list):
    highest_f1 = F_1[0]
    for f1 in range(1,len(F_1)):
        if max(highest_f1, F_1[f1]) == F_1[f1]:
                highest_f1 = F_1[f1]
    threshold = thres_list[F_1.index(highest_f1)]
    return highest_f1, threshold

#%%
all_PC, all_PQ, all_F1star, all_frac_LSH = [], [], [], []
all_precision, all_recall, all_F1, all_frac_clust = [], [], [], []
highf1, all_thresholds = [], []

all_PC_test, all_PQ_test, all_F1star_test, all_frac_LSH_test = [], [], [], []
all_precision_test, all_recall_test, all_F1_test, all_frac_clust_test = [], [], [], []

for i in range(5):
    train_set, test_set = run_bootstrap(data) 

    IDs = train_set[0]
    Title = train_set[1]
    all_cands = train_set[2]
    input_mat = train_set[3]
    
    testID = test_set[0]
    testTitle = test_set[1]
    all_cands_test = test_set[2]
    input_mat_test = test_set[3]

    real_duplicates = real_dups(IDs)
    real_duplicates_test = real_dups(testID)
    
    max_number_compare = comb(len(Title), 2)
    max_number_compare_test = comb(len(testTitle), 2)

    compares_LSH, pairs_LSH = pair_compare_LSH(all_cands, IDs)    
    PC, PQ, frac_LSH, F1_star = LSH_measures(pairs_LSH, compares_LSH, real_duplicates, max_number_compare)

    compares_LSH_test, pairs_LSH_test = pair_compare_LSH(all_cands_test, testID)    
    PC_test, PQ_test, frac_LSH_test, \
        F1_star_test = LSH_measures(pairs_LSH_test, 
                                    compares_LSH_test, real_duplicates_test, 
                                    max_number_compare_test)
    # Clustering
    all_F1_thres = []
    all_dissim_mat = dissimilarity(all_cands, input_mat, len(Title))
    thres_list = np.linspace(0.1,1,10) 
    for thres in thres_list:     
        print(thres)
        all_clusters = clustering(all_dissim_mat, IDs, thres)  
        pairs_clust, number_compare_clust, \
            recall, precision, frac_clust, \
                F_1 = clust_measures(all_clusters, real_duplicates)
        all_precision.append(precision)
        all_recall.append(recall)
        all_F1_thres.append(F_1)
        all_frac_clust.append(frac_clust)
    highest_f1, threshold = high_f1(all_F1_thres, thres_list)
    highf1.append(highest_f1)
    all_F1.append(all_F1_thres)
    
    all_dissim_mat_test = dissimilarity(all_cands_test, input_mat_test, len(testTitle))      
    all_clusters_test = clustering(all_dissim_mat_test, testID, threshold)  
    pairs_clust_test, number_compare_clust_test, recall_test, \
        precision_test, frac_clust_test, \
            F_1_test = clust_measures(all_clusters_test, real_duplicates_test)

    all_PC.append(PC)
    all_PQ.append(PQ)
    all_F1star.append(F1_star)
    all_frac_LSH.append(frac_LSH)
    all_thresholds.append(threshold)
    
    all_PC_test.append(PC_test)
    all_PQ_test.append(PQ_test)
    all_F1star_test.append(F1_star_test)
    all_frac_LSH_test.append(frac_LSH_test)  
    all_precision_test.append(precision_test)
    all_recall_test.append(recall_test)
    all_F1_test.append(F_1_test)
    all_frac_clust_test.append(frac_clust_test)

#%%
PCmean, PQmean, F1starmean, precisionmean, recallmean, F1mean = 0,0,0,0,0,0
for test in range(len(all_PC_test)):
    mean_PC = np.mean(all_PC_test[test])
    mean_PQ = np.mean(all_PQ_test[test])
    mean_F1star = np.mean(all_F1star_test[test])
    
    mean_precision = np.mean(all_precision_test[test])
    mean_recall = np.mean(all_recall_test[test])
    mean_F1 = np.mean(all_F1_test[test])
    PCmean += mean_PC 
    PQmean += mean_PQ 
    F1starmean += mean_F1star 
    precisionmean += mean_precision 
    recallmean += mean_recall 
    F1mean += mean_F1

PCmean = PCmean/len(all_PC_test)
PQmean = PQmean/len(all_PC_test)
F1starmean = F1starmean/len(all_PC_test)
precisionmean = precisionmean/len(all_PC_test)
recallmean = recallmean/len(all_PC_test)
F1mean = F1mean/len(all_PC_test)

thres_mean = np.mean(all_thresholds)

#%%
plt.plot(all_frac_LSH_test[0], all_PC_test[0], label ='Pair completeness')
plt.xlabel('Fraction of comparison')
plt.ylabel('Pair completeness')
plt.show()

plt.plot(all_frac_LSH_test[0], all_PQ_test[0], label ='Pair quality')
plt.xlabel('Fraction of comparison')
plt.ylabel('Pair quality')
plt.show()

plt.plot(all_frac_LSH_test[0], all_F1_test[0][0])
plt.xlabel('Fraction of comparison')
plt.ylabel('F1-measure')
plt.show()
    
