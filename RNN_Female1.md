
# Female


```python
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.preprocessing.text import text_to_word_sequence

import numpy as np
import random
import sys
import numpy as np
import pandas as pd
```

    Using TensorFlow backend.
    

Read the data  
Mark end of message


```python
df1 = pd.read_table('C:\\Users\\DELL\\Desktop\\data_scientist\\rnn\\f.txt', header=None ,error_bad_lines=False)
df1=df1.apply(lambda x: x+' סוףהודעה')
```


```python
#concat all messages
X2=df1[0].tolist()
text= ' '.join(X2)
```

Split a sentence into a list of words.


```python
from keras.preprocessing.text import text_to_word_sequence
words1=text_to_word_sequence(text, lower=False, split=" ")
words = sorted(text_to_word_sequence(text, lower=False, split=" "))
words.append(' ')
vocab_size = len(words)
print('total words:', vocab_size)
```

    total words: 25577
    


```python
def GetUniqueWords(words):
    words_set = set()
    for word in words:                
        words_set.add(word)
    return words_set

unique_words = GetUniqueWords(words)
number_of_words = len(unique_words)
```

Build index_to_word and word_to_index vectors


```python
word_indices = dict((w, i) for i, w in enumerate(words))
indices_word = dict((i, w) for i, w in enumerate(words))
idx = [word_indices[w] for w in words]
```


```python
text_sen=df1[0].tolist()
```

Cut the text in semi-redundant sequences of maxlen words


```python
maxlen = 5
step = 3
sentences = []
next_words = []

# cut the text in semi-redundant sequences of maxlen words
for i in range(0, len(words1) - maxlen, step):
    sentences.append(words1[i: i + maxlen])
    next_words.append(words1[i + maxlen])
print('number sequences:', len(sentences))
```

    number sequences: 8524
    


```python
tmp=pd.DataFrame(sentences)
sentences=tmp[0]+' '+tmp[1]+' '+tmp[2]+' '+tmp[3]+' '+tmp[4]
```


```python
print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(words)), dtype=np.bool)
y = np.zeros((len(sentences), len(words)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(text_to_word_sequence(sentence, lower=False, split=" ")):
        X[i, t, word_indices[word]] = 1
    y[i, word_indices[next_words[i]]] = 1
```

    Vectorization...
    


```python
embedding_matrix = np.zeros((len(word_indices) + 1, 300))
for word, i in word_indices.items():        
    embedding_vector = sentences.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
```

# LSTM Model

The Long Short-Term Memory network or LSTM network is a type of recurrent neural network used in deep learning because very large architectures can be successfully trained.
Is a recurrent neural network that is trained using Backpropagation Through Time and overcomes the vanishing gradient problem

* LSTM layer
* A Dense layer with len(words) nodes
* Activation function= softmax


```python
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(words))))
#a Dense layer with len(words) nodes
model.add(Dense(len(words)))
#Activation function= softmax
model.add(Activation('softmax'))
model.summary()
```

    Build model...
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_1 (LSTM)                (None, 128)               13161472  
    _________________________________________________________________
    dense_1 (Dense)              (None, 25577)             3299433   
    _________________________________________________________________
    activation_1 (Activation)    (None, 25577)             0         
    =================================================================
    Total params: 16,460,905
    Trainable params: 16,460,905
    Non-trainable params: 0
    _________________________________________________________________
    

Compile the model</br>  
We use the RMSProp optimizer</br>  
We use the sparse_categorical_crossentropy loss that accepts sparse labels


```python
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
```

Split test, train set


```python
from array import array
import random
from random import randrange
test_x=[]
test_y=[]

index=[]
for i in range(0,900):
    random_index = randrange(0,len(X))
    index.append(random_index)
    test_x.append(X[random_index])
    test_y.append(y[random_index])
    
test_x=np.array(test_x)
test_y=np.array(test_y)

train_x = np.delete(X, index, axis=0)
train_y = np.delete(y, index, axis=0)

```


```python
train the model, fit it to the data
model.fit(train_x, train_y,
          batch_size=128,
          epochs=20,validation_split=0.2)
model.save('C:\\Users\\DELL\\Desktop\\data_scientist\\rnn\\RNN_F_train.h5')
```


```python
from keras.models import load_model
model1 = load_model('C:\\Users\\DELL\\Desktop\\data_scientist\\rnn\\RNN_F_train.h5')
```


```python
scores = model1.evaluate(test_x, test_y, verbose=1)
print("Accuracy:",scores[1]*100,"%")
```

    900/900 [==============================] - 7s     
    Accuracy: 75.5555555556 %
    

Train the model</BR>  
Fit it to the all data


```python
#train the model, fit it to the data
model.fit(X, y,
          batch_size=128,
          epochs=20,validation_split=0.2)

#save the model
model.save('C:\\Users\\DELL\\Desktop\\data_scientist\\rnn\\RNN_F.h5')
```


```python
model = load_model('C:\\Users\\DELL\\Desktop\\data_scientist\\rnn\\RNN_F.h5')
```


```python
scores = model.evaluate(X, y, verbose=1)
print("Accuracy:",scores[1]*100,"%")
```

    8524/8524 [==============================] - 75s    
    Accuracy: 99.8005631159 %
    

# Generate new sentences


```python
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def concatenate_list_data(list):
    result= ''
    for element in list:
        result += str(element)+" "
    return result[0:-1]
```


```python
import csv
#generate new sentences
to_file=[]
for i in range(0,1500):
    #select seed index- first word of the sentance
    start_index = random.randint(0, len(words1) - maxlen - 1)
    while(words1[start_index:start_index+1][0] == 'סוףהודעה'):
        start_index = random.randint(0, len(words1) - maxlen - 1)

    for diversity in [1.0]:
        generated = ''
        sentence = words1[start_index: start_index+1]
        generated += concatenate_list_data(sentence)

        sys.stdout.write(generated)
        sen_to_file=generated

        next_word=''
        z=0
        for i in range(0,12):
            x = np.zeros((1, maxlen, len(words)))
            for t, word in enumerate(sentence):
                x[0, t, word_indices[word]] = 1.
            #predict the next words
            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]
            generated += next_word+' '
            sentence = sentence[1:] + list([next_word])
            if(next_word == 'סוףהודעה'):
                break
            sen_to_file+=' '+next_word
            z+=1
            sys.stdout.write(' '+next_word)              
            sys.stdout.flush()
        print()
        if(len(sen_to_file.split(' '))>1):
            to_file.append(sen_to_file)
            
```


```python
save to file
with open('C:\\Users\\DELL\\Desktop\\data_scientist\\rnn\\ff_new.txt', 'w', newline='\n', encoding='utf-8') as txt_file:
    writer = csv.writer(txt_file, delimiter='\n')
    writer.writerow(to_file)
```

# 2 Words Model


```python
import re
#read the data
df1 = pd.read_table('C:\\Users\\DELL\\Desktop\\data_scientist\\rnn\\f.txt', header=None ,error_bad_lines=False)
#mark end of message
df1=df1.apply(lambda x: x+' סוףהודעה')

# import re
regex = re.compile('[^םןאבגדהוזחטיכלמנסעפצקרשתץףך?! ].*')
df1[0]=df1[0].apply(lambda x: regex.sub('', x))
df1=df1.dropna()
df1=df1[df1[0]!=' ']

#concat all messages
text= ' '.join(df1[0].tolist())
```


```python
from keras.preprocessing.text import text_to_word_sequence
words1=text_to_word_sequence(text, lower=False, split=" ")
words = sorted(text_to_word_sequence(text, lower=False, split=" "))
words.append(' ')
vocab_size = len(words)
print('total words:', vocab_size)

unique_words = GetUniqueWords(words)
number_of_words = len(unique_words)
print('unique_words:', number_of_words)

word_indices = dict((w, i) for i, w in enumerate(unique_words))
indices_word = dict((i, w) for i, w in enumerate(unique_words))
idx = [word_indices[w] for w in words]
```

    total words: 19494
    unique_words: 4629
    


```python
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

cs = 2

c1_dat = [idx[i] for i in range(0, len(idx)-1-cs, 1)]
c2_dat = [idx[i+1] for i in range(0, len(idx)-1-cs, 1)]
c3_dat = [idx[i+2] for i in range(0, len(idx)-1-cs, 1)] 

x1 = np.array(c1_dat)
x2 = np.array(c2_dat)
x3 = np.array(c3_dat)
    
input_ = np.stack([x1,x2],axis=1)
output_ = np.stack([x3],axis=1)

n_fac = 42
n_hidden = 256

#build the model 
model3=Sequential([
    Embedding(number_of_words, n_fac, input_length=cs),
    LSTM(n_hidden, return_sequences=False, activation='relu'),        
    Dense(number_of_words, activation='softmax'),
])    
  
print(model.summary()) 
#compile
model3.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop' ,metrics=["accuracy"])
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_1 (LSTM)                (None, 128)               13161472  
    _________________________________________________________________
    dense_1 (Dense)              (None, 25577)             3299433   
    _________________________________________________________________
    activation_1 (Activation)    (None, 25577)             0         
    =================================================================
    Total params: 16,460,905
    Trainable params: 16,460,905
    Non-trainable params: 0
    _________________________________________________________________
    None
    


```python
from array import array
import random
from random import randrange
#split to train and test set

test_x=[]
test_y=[]

index=[]
for i in range(0,900):
    random_index = randrange(0,len(input_))
    index.append(random_index)
    test_x.append(input_[random_index])
    test_y.append(output_[random_index])
    
test_x=np.array(test_x)
test_y=np.array(test_y)

train_x = np.delete(input_, index, axis=0)
train_y = np.delete(output_, index, axis=0)

```


```python
model3.fit(train_x, y=train_y, batch_size=120, epochs=30, verbose=1)
model3.save('C:\\Users\\DELL\\Desktop\\data_scientist\\rnn\\RNN_F3_train.h5')
```

    Epoch 1/30
    18610/18610 [==============================] - 27s - loss: 2.9599 - acc: 0.6384    
    Epoch 2/30
    18610/18610 [==============================] - 28s - loss: 2.8953 - acc: 0.6481    
    Epoch 3/30
    18610/18610 [==============================] - 27s - loss: 2.8324 - acc: 0.6569    
    Epoch 4/30
    18610/18610 [==============================] - 27s - loss: 2.7728 - acc: 0.6642    
    Epoch 5/30
    18610/18610 [==============================] - 28s - loss: 2.7131 - acc: 0.6715    
    Epoch 6/30
    18610/18610 [==============================] - 29s - loss: 2.6540 - acc: 0.6794    
    Epoch 7/30
    18610/18610 [==============================] - 28s - loss: 2.5974 - acc: 0.6854    
    Epoch 8/30
    18610/18610 [==============================] - 26s - loss: 2.5405 - acc: 0.6913    
    Epoch 9/30
    18610/18610 [==============================] - 27s - loss: 2.4839 - acc: 0.6983    
    Epoch 10/30
    18610/18610 [==============================] - 26s - loss: 2.4279 - acc: 0.7041    
    Epoch 11/30
    18610/18610 [==============================] - 26s - loss: 2.3750 - acc: 0.7113    
    Epoch 12/30
    18610/18610 [==============================] - 27s - loss: 2.3231 - acc: 0.7179    
    Epoch 13/30
    18610/18610 [==============================] - 27s - loss: 2.2706 - acc: 0.7237    
    Epoch 14/30
    18610/18610 [==============================] - 27s - loss: 2.2204 - acc: 0.7304    
    Epoch 15/30
    18610/18610 [==============================] - 28s - loss: 2.1713 - acc: 0.7358    
    Epoch 16/30
    18610/18610 [==============================] - 28s - loss: 2.1225 - acc: 0.7414    
    Epoch 17/30
    18610/18610 [==============================] - 29s - loss: 2.0724 - acc: 0.7472    
    Epoch 18/30
    18610/18610 [==============================] - 28s - loss: 2.0215 - acc: 0.7538    
    Epoch 19/30
    18610/18610 [==============================] - 27s - loss: 1.9710 - acc: 0.7601    
    Epoch 20/30
    18610/18610 [==============================] - 27s - loss: 1.9197 - acc: 0.7691    
    Epoch 21/30
    18610/18610 [==============================] - 28s - loss: 1.8699 - acc: 0.7749    
    Epoch 22/30
    18610/18610 [==============================] - 27s - loss: 1.8177 - acc: 0.7816    
    Epoch 23/30
    18610/18610 [==============================] - 26s - loss: 1.7671 - acc: 0.7864    
    Epoch 24/30
    18610/18610 [==============================] - 26s - loss: 1.7108 - acc: 0.7908    
    Epoch 25/30
    18610/18610 [==============================] - 26s - loss: 1.6529 - acc: 0.7945    
    Epoch 26/30
    18610/18610 [==============================] - 27s - loss: 1.6027 - acc: 0.7977    
    Epoch 27/30
    18610/18610 [==============================] - 27s - loss: 1.5283 - acc: 0.7999    
    Epoch 28/30
    18610/18610 [==============================] - 29s - loss: 1.4646 - acc: 0.8033    
    Epoch 29/30
    18610/18610 [==============================] - 29s - loss: 1.4087 - acc: 0.8066    
    Epoch 30/30
    18610/18610 [==============================] - 27s - loss: 1.3559 - acc: 0.8140    
    


```python
model4 = load_model('C:\\Users\\DELL\\Desktop\\data_scientist\\rnn\\RNN_F3_train.h5')
```


```python
scores = model4.evaluate(test_x, test_y, verbose=1)
print("Accuracy:",scores[1]*100,"%")
```

    832/900 [==========================>...] - ETA: 0sAccuracy cv: 70.1111111111 %
    

Train the model</BR>  
Fit it to the all data


```python
model3.fit(input_, y=output_, batch_size=120, epochs=30, verbose=1)
model3.save('C:\\Users\\DELL\\Desktop\\data_scientist\\rnn\\RNN_F3.h5')
```

    Epoch 1/30
    19491/19491 [==============================] - 31s - loss: 7.0879 - acc: 0.3479    
    Epoch 2/30
    19491/19491 [==============================] - 32s - loss: 4.6981 - acc: 0.4768    
    Epoch 3/30
    19491/19491 [==============================] - 31s - loss: 3.9711 - acc: 0.5406    
    Epoch 4/30
    19491/19491 [==============================] - 32s - loss: 3.5823 - acc: 0.5849    
    Epoch 5/30
    19491/19491 [==============================] - 32s - loss: 3.2815 - acc: 0.6183    
    Epoch 6/30
    19491/19491 [==============================] - 33s - loss: 3.0589 - acc: 0.6415    
    Epoch 7/30
    19491/19491 [==============================] - 29s - loss: 2.8893 - acc: 0.6572    
    Epoch 8/30
    19491/19491 [==============================] - 29s - loss: 2.7390 - acc: 0.6713    
    Epoch 9/30
    19491/19491 [==============================] - 29s - loss: 2.6015 - acc: 0.6854    
    Epoch 10/30
    19491/19491 [==============================] - 30s - loss: 2.4740 - acc: 0.6972    
    Epoch 11/30
    19491/19491 [==============================] - 31s - loss: 2.3539 - acc: 0.7080    
    Epoch 12/30
    19491/19491 [==============================] - 35s - loss: 2.2400 - acc: 0.7201    
    Epoch 13/30
    19491/19491 [==============================] - 30s - loss: 2.1254 - acc: 0.7295    
    Epoch 14/30
    19491/19491 [==============================] - 29s - loss: 2.0170 - acc: 0.7408    
    Epoch 15/30
    19491/19491 [==============================] - 29s - loss: 1.9098 - acc: 0.7522    
    Epoch 16/30
    19491/19491 [==============================] - 32s - loss: 1.8064 - acc: 0.7639    
    Epoch 17/30
    19491/19491 [==============================] - 29s - loss: 1.7105 - acc: 0.7763    
    Epoch 18/30
    19491/19491 [==============================] - 30s - loss: 1.6188 - acc: 0.7904    
    Epoch 19/30
    19491/19491 [==============================] - 29s - loss: 1.5356 - acc: 0.8023    
    Epoch 20/30
    19491/19491 [==============================] - 30s - loss: 1.4592 - acc: 0.8129    
    Epoch 21/30
    19491/19491 [==============================] - 31s - loss: 1.3898 - acc: 0.8235    
    Epoch 22/30
    19491/19491 [==============================] - 32s - loss: 1.3235 - acc: 0.8317    
    Epoch 23/30
    19491/19491 [==============================] - 33s - loss: 1.2675 - acc: 0.8416    
    Epoch 24/30
    19491/19491 [==============================] - 31s - loss: 1.2050 - acc: 0.8519    
    Epoch 25/30
    19491/19491 [==============================] - 31s - loss: 1.1472 - acc: 0.8607    
    Epoch 26/30
    19491/19491 [==============================] - 30s - loss: 1.1044 - acc: 0.8681    
    Epoch 27/30
    19491/19491 [==============================] - 32s - loss: 1.0593 - acc: 0.8736    
    Epoch 28/30
    19491/19491 [==============================] - 30s - loss: 1.0217 - acc: 0.8804    
    Epoch 29/30
    19491/19491 [==============================] - 31s - loss: 0.9883 - acc: 0.8843    
    Epoch 30/30
    19491/19491 [==============================] - 30s - loss: 0.9562 - acc: 0.8897    
    


```python
model3 = load_model('C:\\Users\\DELL\\Desktop\\data_scientist\\rnn\\RNN_F3.h5')
```


```python
scores = model3.evaluate(input_, output_, verbose=1)
print("Accuracy:",scores[1]*100,"%")
```

    19491/19491 [==============================] - 12s    
    Accuracy: 90.282694577 %
    

# generate new sentences


```python
def get_next_keras(inp):
    idxs = [word_indices[c] for c in inp] #convert characters to indices
    arrs = np.array(idxs)[np.newaxis,:] #converting to the required input format
    p = model3.predict(arrs)[0] #using the model to predict the next index
    return words[np.argmax(p)] #converting the index with max probability to a character
```


```python
#generate new sentences
to_file=[]
for i in range(0,5):
    #select seed index- first word of the sentance
    start_index = random.randint(0, len(words1) - maxlen - 1)
    sen_to_file=''
    w= words1[start_index:start_index+2]

    while(w[0] == 'סוףהודעה' or w[1] == 'סוףהודעה'):
        start_index = random.randint(0, len(words1) - maxlen - 1)
        w= words1[start_index:start_index+2]
   
    w= words1[start_index:start_index+2]
    sen_to_file+=w[0]+' '+w[1]
    for i in range(0,5):
        #predict the next words
        next_word= get_next_keras(w)
        if(next_word == 'סוףהודעה'):
            break
        sen_to_file+=' '+next_word
        w= [w[1],next_word]
    if(len(sen_to_file.split(' '))>2):
            to_file.append(sen_to_file)
            print(sen_to_file)

```

    עוגה רצינית ב אספיק בבית אוכלת בנו
    אוקיי אז אין בחמישי בבירור דבר גזר
    ערב טוב בולבולים אתן בשיעור אדמה בשקם
    שחייבים להגיע בכללי דק הוא אין אותך
    מפחיד לתלות איתי בלילה גם בשישי את
    


```python
#save to file
with open('C:\\Users\\DELL\\Desktop\\data_scientist\\rnn\\ff2_new.txt', 'w', newline='\n', encoding='utf-8') as txt_file:
    writer = csv.writer(txt_file, delimiter='\n')
    writer.writerow(to_file)
```
