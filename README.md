## Scalable Product Duplicate Detection
# Jiayao Hu (523775)

This code in this file name 'PaperCode.py' contains several functions, which will be called at the end to 
compute the performance. In the following, an overview is given for the functions and a brief explanation of its use.
After importing the data the following functions are used:

get_ID_title(data):
    From the dataset extract the modelID and the title
    
    Input: data, this is the data set (json file in this case)
    Output: modelID, list of the extracted modelIDs
            title, list of the extracted titles

inches(string):
    Data Cleaning
    Normalize the inch definition. Change all variations to 'Inch'

    Input: string to check whether a prespecified version of inch is in there
    Output: string, where if found the inch-version is changed to 'Inch
    
hertz(string):
    Normalize the hertz definition. Change all variations to 'Hz'

    Input: string to check whether a version of herzt is in there
    Output: string, where hertz-version is changed to 'Hz'
    
interpunction(string): 
    Here, all interpunction is removed except for the following: 
    dot(.), comma (,), white space and bar (-)
    So the following are removed: !@#$%^&*_+~`;<>/|()[]?-'

    Input: string, possibly with interpunction
    Output: string without the specified interpunction
    
capitalization(string):
    Return the string in lower letters

    Input: string with both upper and lower case letters
    Output: string all in lower ca    
    
remove(string):
    Remove the shopname from the title

    Input: string, where the shopname might still be in
    Output: string with the shopname removed    
    
del_white_spaces(string):
    Delete unnecessary white spaces

    Input: string posisbly with multiple white spaces after each other
    Output: string with maximum one white space between words    
    
clean_title(title):
    Data cleaning is applied to the titles
    
    Input: title, list of titles not yet cleaned
    Output: cleaned_title, list of titles that are cleaned    
    
bootstrapping(modelID, title):
    Apply bootstrapping with replacement to the modelIDs and titles. We 
    randomly select N (the number of products) indices. Then, only the unique
    ones are kept and the product with the corresponding index is extracted.
    
    Input: modelID, the list of modelIDs
           title, list of titles
    Output: new_modelID, bootstrapped modelIDs
            new_title, bootstrapped titles   
    
model_words(lists):
    Extract the model words from the list of titles, and keep only the unique ones
    
    Input: lists, the list to extract the model words from
    Output: MW_title, list with the unique model words  
    
binary_mat(MW, cleaned):
    To create the binary matrix, where the entry is '1' if the title contains the 
    model word and '0' if it is not in there
    
    Input: MW, the list of model words
           cleaned, list of titles (cleaned)
    Output: input_matrix, the binary matrix    
    
isPrime(x):
    Return the first prime value that is larger than x

    Input: x, integer (here, the number of rows of the input_matrix)
    Output: counter, the first prime value above x    
    
hashing(a, b, x, prime):  
    Creates hash functions and returns the value 

    Input: a, integer between 0 and 49
        b, integer between 0 and 49
        x, integer. The x-th row of the input_matrix 
        prime, integer; obtained from the isPrime() function
    Output: hash_value, integer    
    
list_a_b(N):
    Determine integer values a and b to create the hash functions
    
    Input: N, (int) the number of hash functions to create and thus, the 
                number of random integers to generate 
    Output: list_a, list with N random integers between 1 and N
            list_b, list with N random integers between 1 and N    
    
minhashing_sig(input_matrix, N): 
    The minhash values are computed  and used to create the signature matrix
    
    Input: input_matrix, binary matrix 
           N, the number of hash functions
    Output: sig_mat, signature matrix    
    
bandrow(sig_mat):  
    The number of bands and rows are determined
    
    Input: sig_mat, the signature matrix to divide in bands and rows
    Output: selected_b_r, list of the selected b and r pairs    
    
bucket_hash(selected_b_r, sig_mat):
    First, all hash values of each band and column are determined. Then, 
    the ones with the same hash value are joined in one bucket, because these 
    will be considered candidate pairs. We want to only hash those items that 
    are identical in at least one band.    
    
    Input: sig_mat, the signature matrix to divide in bands and rows
           selected_b_r, list of the selected b and r pairs
    Output: all_bucket_hash, all hash values for the buckets
    
all_bucks(all_bucket_hash):
    The entries of the columns per row are compared. If in at least one band 
    the product has the same string, the corresponding index will be hashed to 
    a bucket. 
    
    Input: all_bucket_hash, the hash values of each band
    Output: all_buckets, list of the buckets    
    
pairs(all_buckets, N):
    Here, the candidate pairs are detemined and collected
    
    Input: all_buckets, the list of products hashed to the same bucket
           N, the total number of buckets
    Output: all_pairs, list with all candidate pairs
    
candidates(all_pairs, N):
    Matrix with all candidate pairs. The entry is a '1' if the product is a
    duplicate with another product and '0', otherwise.
    
    Input: all_pairs, list of all found pairs
           N, (int) the number of products 
    Output: all_cands, all matrices of candidate pairs    
    
jaccard_dissim(A, B):
    Determine the dissimilarity between columns, which is '1-jaccard_similarity'
    
    Input: A, column of matrix
           B, other column of matrix
    Output: 1-similarity, dissimilarity/distance  
    
dissimilarity(all_cands, input_matrix, N):
    To generate the dissimilarity matrix
    
    Input: all_cands, the candidate matrix
           input_matrix, the binary matrix
           N, the number of products 
    Output: all_dissim_mat, all dissimiliarity matrices  
    
run_bootstrap(dataset):
    Generate the required output from the above defined funcions
    
    Input: dataset, the data
    Output: training, all the lists/matrices etc. for training
            test, all the lists/matrices etc. for test
    
clustering(all_dissim_matrix, modelID, t):
    Obtaining the clusters

    Input: all_dissim_matrix, dissimilarity matrix
           modelID, ID of the TV
           t, distance threshold
    Output: all_clusts, the labels of the clusters
    
real_dups(ID):
    The total number of real duplicates in the dataset

    Input: ID, list of modelIDs
    Output: real_duplicates, the number of real duplicates in the given list
    
pair_compare_LSH(all_cands, ID):
    To compute the number of real duplicates found by LSH and the number of comparisons made in the progress

    Input:  all_cands, the candidate pairs obtained by LSH 
            ID, list of modelIDs
    Output: number_compare_LSH, number of comparisons made by LSH 
            pairs_LSH, the number of duplicates found by LSH 
    
LSH_measures(pairs, compares, real, max_comp):
    To compute the performance measures of LSH

    Input:  pairs, the number of duplicates found by LSH 
            compares, number of comparisons made by LSH 
            real, the total number of real duplicates 
            max_comp, the maximum number of comparisons
    Output: PC, number of duplicates found divided by the total number of real duplicates 
            PQ, number of duplicates found divided by the number of comparisons made 
            frac_LSH, the fraction of comparisons
            F1_star, harmonic mean between PC and PQ

clust_measures(all_clusters, real_duplicates):
    To compute the performance measures of clustering

    Input:  all_clusters, the labels of the clusters
            real_duplicates, the total number of real duplicates
    Output: pairs_clust, the number of duplicates found by clustering 
            number_compare_clust, number of comparisons made by clustering
            recall, number of duplicates found divided by the total number of real duplicates 
            precision, number of duplicates found divided by the number of comparisons made 
            frac_clust, the fraction of comparisons 
            F_1, harmonic mean between precision and recall
    
high_f1(F_1, thres_list):
    Function to obtain the highest F1-score with different distance thresholds
    
    Input: F_1, list with all the F1-scores 
           thres_list, list with all the tried distance thresholds
    Output: highest_f1, the highest F1-score
            threshold, the distance threshold that gives the highest F1-score
    
    
