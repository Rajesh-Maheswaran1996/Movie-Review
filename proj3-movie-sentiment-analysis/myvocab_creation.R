# Creating the new vocab from a combination of all trains
  j = 1 
  setwd(paste("/Users/preethaljoseph/CS_498_Project3/split_", j, sep=""))
  train1 = read.table("train.tsv", stringsAsFactors = FALSE, header = TRUE)
  j = 2 
  setwd(paste("/Users/preethaljoseph/CS_498_Project3/split_", j, sep=""))
  train2 = read.table("train.tsv", stringsAsFactors = FALSE, header = TRUE)
  j = 3 
  setwd(paste("/Users/preethaljoseph/CS_498_Project3/split_", j, sep=""))
  train3 = read.table("train.tsv", stringsAsFactors = FALSE, header = TRUE)
  j = 4 
  setwd(paste("/Users/preethaljoseph/CS_498_Project3/split_", j, sep=""))
  train4 = read.table("train.tsv", stringsAsFactors = FALSE, header = TRUE)
  j = 5 
  setwd(paste("/Users/preethaljoseph/CS_498_Project3/split_", j, sep=""))
  train5 = read.table("train.tsv", stringsAsFactors = FALSE, header = TRUE)
  
  train = rbind(train1,train2,train3,train4,train5)
  
  stop_words = c("i", "me", "my", "myself", 
                 "we", "our", "ours", "ourselves", 
                 "you", "your", "yours", 
                 "their", "they", "his", "her", 
                 "she", "he", "a", "an", "and",
                 "is", "was", "are", "were", 
                 "him", "himself", "has", "have", 
                 "it", "its", "the", "us")
  
  # creating word tokens and converting to lower case 
  it_train = itoken(train$review,
                    preprocessor = tolower, 
                    tokenizer = word_tokenizer)
  
  # creating dictionary with all tokens and removing stopwords 
  
  tmp.vocab = create_vocabulary(it_train, 
                                stopwords = stop_words, 
                                ngram = c(1L,4L))
  
  tmp.vocab = prune_vocabulary(tmp.vocab, term_count_min = 10,
                               doc_proportion_max = 0.5,
                               doc_proportion_min = 0.001)
  
  dtm_train  = create_dtm(it_train, vocab_vectorizer(tmp.vocab))
  
  
  # I use the training data from the first split. Suppose dtm_train is the document_term_matrix (with over 30K cols) which 
  # we obtained at @697. Since dtm_train is a large sparse matrix, I use commands from the R package 'slam' to efficiently 
  # compute the mean and var for each column of dtm_train. 
  
  v.size = dim(dtm_train)[2]
  ytrain = train$sentiment
  
  summ = matrix(0, nrow=v.size, ncol=4)
  summ[,1] = colapply_simple_triplet_matrix(
    as.simple_triplet_matrix(dtm_train[ytrain==1, ]), mean)
  summ[,2] = colapply_simple_triplet_matrix(
    as.simple_triplet_matrix(dtm_train[ytrain==1, ]), var)
  summ[,3] = colapply_simple_triplet_matrix(
    as.simple_triplet_matrix(dtm_train[ytrain==0, ]), mean)
  summ[,4] = colapply_simple_triplet_matrix(
    as.simple_triplet_matrix(dtm_train[ytrain==0, ]), var)
  
  n1 = sum(ytrain); 
  n = length(ytrain)
  n0 = n - n1
  
  myp = (summ[,1] - summ[,3])/
    sqrt(summ[,2]/n1 + summ[,4]/n0)
  
  
  # I ordered words by the magnitude of their t-statistics and picked the top 2000 words, 
  # which are then divided into two lists: positive words and negative words. 
  
  words = colnames(dtm_train)
  id = order(abs(myp), decreasing=TRUE)[1:2000]
  
  mynewvocab = words[id]
  
  sel_dtm_train = dtm_train[,words[id]]
  
  set.seed(12345)
  tmpfit = glmnet(x = sel_dtm_train, 
                  y = train$sentiment, 
                  alpha = 1, 
                  family='binomial')
  tmpfit$df
  
  nvalue = length(tmpfit$df[tmpfit$df<1000])
  mynewvocab = colnames(sel_dtm_train)[which(tmpfit$beta[, nvalue] != 0)]
  mynewvocab = mynewvocab[1:960]
  print(mynewvocab)
  setwd("/Users/preethaljoseph/CS_498_Project3/")
  write.table(mynewvocab, file = "myvocab.txt", row.names = FALSE, sep='\t')
  

