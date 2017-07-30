
# RNN_LSTM-Female

The Long Short-Term Memory network or LSTM network is a type of recurrent neural network used in deep learning because very large architectures can be successfully trained.
Is a recurrent neural network that is trained using Backpropagation Through Time and overcomes the vanishing gradient problem


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
#read the data
df1 = pd.read_table('C:\\Users\\DELL\\Desktop\\data_scientist\\rnn\\f.txt', header=None ,error_bad_lines=False)

#mark end of message
df1=df1.apply(lambda x: x+' סוףהודעה')
#concat all messages
X2=df1[0].tolist()
text= ' '.join(X2)
```

Split a sentence into a list of words.


```python
from keras.preprocessing.text import text_to_word_sequence
#Split a sentence into a list of words.
words1=text_to_word_sequence(text, lower=False, split=" ")
words = sorted(text_to_word_sequence(text, lower=False, split=" "))
words.append(' ')
vocab_size = len(words)
print('total words:', vocab_size)
```

    total words: 25577
    

Build index_to_word and word_to_index vectors


```python
# build index_to_word and word_to_index vectors
word_indices = dict((w, i) for i, w in enumerate(words))
indices_word = dict((i, w) for i, w in enumerate(words))
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
    

Build the model </br>
* LSTM layer
* A Dense layer with len(words) nodes
* Activation function= softmax


```python
# build the model: LSTM
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
    

Compile the model


```python
#compile the model.
#we use the RMSProp optimizer
#we use the sparse_categorical_crossentropy loss that accepts sparse labels

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
```


```python
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
```


```python
def concatenate_list_data(list):
    result= ''
    for element in list:
        result += str(element)+" "
    return result[0:-1]
```

Model loading


```python
from keras.models import load_model
model = load_model('C:\\Users\\DELL\\Desktop\\data_scientist\\rnn\\modelrnn5f.h5')
```

Train the model, fit it to the data


```python
#train the model, fit it to the data
model.fit(X, y,
          batch_size=128,
          epochs=20)
```

    Epoch 1/20
    8524/8524 [==============================] - 186s - loss: 7.4406 - acc: 0.1844   
    Epoch 2/20
    8524/8524 [==============================] - 183s - loss: 6.3353 - acc: 0.1943   
    Epoch 3/20
    8524/8524 [==============================] - 189s - loss: 5.9810 - acc: 0.2027   
    Epoch 4/20
    8524/8524 [==============================] - 189s - loss: 5.5262 - acc: 0.2085   
    Epoch 5/20
    8524/8524 [==============================] - 192s - loss: 4.9955 - acc: 0.2344   
    Epoch 6/20
    8524/8524 [==============================] - 199s - loss: 4.3575 - acc: 0.2886   
    Epoch 7/20
    8524/8524 [==============================] - 191s - loss: 3.5789 - acc: 0.3842   
    Epoch 8/20
    8524/8524 [==============================] - 188s - loss: 2.7560 - acc: 0.5041   
    Epoch 9/20
    8524/8524 [==============================] - 192s - loss: 1.9900 - acc: 0.6561   
    Epoch 10/20
    8524/8524 [==============================] - 191s - loss: 1.3787 - acc: 0.7901   
    Epoch 11/20
    8524/8524 [==============================] - 192s - loss: 0.9278 - acc: 0.8740   
    Epoch 12/20
    8524/8524 [==============================] - 192s - loss: 0.6197 - acc: 0.9257   
    Epoch 13/20
    8524/8524 [==============================] - 189s - loss: 0.4267 - acc: 0.9532   
    Epoch 14/20
    8524/8524 [==============================] - 194s - loss: 0.3173 - acc: 0.9695   
    Epoch 15/20
    8524/8524 [==============================] - 192s - loss: 0.2529 - acc: 0.9748   
    Epoch 16/20
    8524/8524 [==============================] - 185s - loss: 0.2216 - acc: 0.9802   
    Epoch 17/20
    8524/8524 [==============================] - 193s - loss: 0.1732 - acc: 0.9863   
    Epoch 18/20
    8524/8524 [==============================] - 188s - loss: 0.1576 - acc: 0.9866   
    Epoch 19/20
    8524/8524 [==============================] - 199s - loss: 0.1275 - acc: 0.9921   
    Epoch 20/20
    8524/8524 [==============================] - 198s - loss: 0.1157 - acc: 0.9905   
    




    <keras.callbacks.History at 0x231dc25cf98>



Save the model


```python
# save the model
model.save('C:\\Users\\DELL\\Desktop\\data_scientist\\rnn\\modelrnn5ff.h5')
```

Generate new sentences


```python
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
import csv
#save to file
with open('C:\\Users\\DELL\\Desktop\\data_scientist\\rnn\\ff_new.txt', 'w', newline='\n', encoding='utf-8') as txt_file:
    writer = csv.writer(txt_file, delimiter='\n')
    writer.writerow(to_file)
```
