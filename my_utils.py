def read_data(data):
    print("Aperçu des données :")
    print(data.head(5))
    # Afficher les informations sur le DataFrame
    print("\nInformations sur les données :")
    print(data.info())
    print(" \n Taille : \n",data.shape)
#--------------------------------------------------------   
def reduced_dim(data,nb_positive,nb_negative,nb_neutre):
    num_rows = {
        0.0:nb_negative ,#570000,
        1.0:nb_positive ,#560000,
        2.0:nb_neutre, #10725  
    }
    data = data.groupby('sentiment').apply(lambda x: x.sample(num_rows[x.name]))
    # Réinitialiser l'index du DataFrame
    data = data.reset_index(drop=True)
    return data
#--------------------------------------------------------      
def str2emoji(tweet):
    for pos,ej in enumerate(tweet):
        if ej in emojis:
            tweet[pos]=emojis[ej]
    return tweet
 #--------------------------------------------------------  
def dropNan_value(data):
    index_with_nan = data.index[data.isnull().any(axis=1)]
    print("___________-Index of Nan_value : \n",index_with_nan)
    print("___________-number of Nan_value : \n",data.isnull().any(axis=1).value_counts())
    
   # print(data.shape())
    data = data.dropna()  # Drop rows with any NaN values
    print("____________After dropping nan_value :\n",data.index[data.isnull().any(axis=1)])
   # print(data.shape())
    return data
#--------------------------------------------------------      
  def preprocess(txt):
    #txt = txt.lower() # Convert to lowercase
    txt = re.sub(r"\\u2019","'",txt)
    txt = re.sub(r"\\u002c","'",txt)
    txt = ' '.join(str2emoji(unidecode(txt).lower().split()))
    txt = re.sub(r'http\S+', '', txt)
    txt = re.sub(r'\S+@\S+', '', txt)
    txt = re.sub(r'\d{1,2}/\d{1,2}/\d{4}', '', txt)
    txt = re.sub(r"\'ve","have",txt)
    txt = re.sub(r"can\'t","cannot",txt)
    txt = re.sub(r"n\'t","not",txt)
    txt = re.sub(r"\'re","are",txt)
    txt = re.sub(r"\'d","would",txt)
    txt = re.sub(r"\'ll","will",txt)
    txt = re.sub(r"\'s","",txt)
    txt = re.sub(r"\'n","",txt)
    txt = re.sub(r"\'m","am",txt)
    txt = re.sub(r'@\w+', '', txt)
    txt = re.sub(r'#\w+', '', txt)
    txt = re.sub(r'[0-9]+', ' ', txt)
    txt = re.sub('[^a-zA-Z]', ' ', txt)
    txt = [lemmatizer.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else
    lemmatizer.lemmatize(i) for i,j in pos_tag(tokenizer.tokenize(txt))]
    txt = [i for i in txt if (i not in stopwords) and (i not in punctuation)]
    txt = ' '.join(txt)
    return txt
#--------------------------------------------------------      
 def data_preprocessing(data,tweet):
    data['tweet_desc']=data[tweet].apply(lambda x:preprocess(x))
    return data['tweet_desc']
#--------------------------------------------------------     
MAX_LEN=95
def tokenize_pad_sequences(text):
    # Text tokenization
    tokenizer = Tokenizer(filters=' ')
    tokenizer.fit_on_texts(tweet_preprocessed1)
    word_index=tokenizer.word_index
    # Transforms text to a sequence of integers
    X = tokenizer.texts_to_sequences(text)
    X = pad_sequences(X,maxlen=MAX_LEN)
    # return sequences
    return X, tokenizer,word_index

def show_tokenize_pad_sequences(text):
    print('Before Tokenization & Padding \n', text[0])
    X, tokenizer,word_index = tokenize_pad_sequences(text)
    print('After Tokenization & Padding \n', X[0])   
#--------------------------------------------------------     
#--------------------------------------------------------  
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, i in word_index.items():
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector
def getModel():
    embedding_layer = tf.keras.layers.Embedding(vocab_size,
                                          EMBEDDING_DIM,
                                          weights=[embedding_matrix],
                                          input_length=MAX_LEN,
                                          trainable=False)

    model = Sequential([
        embedding_layer,
        Bidirectional(LSTM(64, dropout=0.2, return_sequences=True)),
        Bidirectional(LSTM(64, return_sequences=True)),
        # Conv1D(100, 5, activation='relu'),
        GlobalMaxPool1D(),
        Dense(3, activation='softmax'),
    ])
    return model  
#--------------------------------------------------------  
#--------------------------------------------------------      
def plot_confusion_matrix(confusion_mat,Y_test):
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_mat, cmap=plt.cm.Blues, annot=True, fmt='d', 
                xticklabels=Y1_test.unique(),
                yticklabels=Y1_test.unique())
    plt.title('Confusion matrix', fontsize=16)
    plt.xlabel('Actual label', fontsize=12)
    plt.ylabel('Predicted label', fontsize=12)
    
plot_confusion_matrix(cnf_matrix, Y1_test)  
 #--------------------------------------------------------      
#--------------------------------------------------------   
 def getTransfertModel(model):
    transfert_model = Sequential()
    for layer in model.layers[:-1]:
        transfert_model.add(layer)
    # Congeler les poids des couches pré-entraînées
    for layer in transfert_model.layers:
        layer.trainable = False
    transfert_model.add(Dense(150, activation='relu',name='dense1'))
    model.add(Dense(100, activation='relu',name='dense2'))
    transfert_model.add(Dense(64, activation='relu',name='dense3'))
    transfert_model.add(Dense(7, activation='softmax',name='dense4'))
    return transfert_model #--------------------------------------------------------      
#--------------------------------------------------------     
#--------------------------------------------------------    
    
#--------------------------------------------------------    
    
