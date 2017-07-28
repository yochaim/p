
# RNN_LSTM

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
    
read the data
mark end of message

```python
#read the data
df1 = pd.read_table('C:\\Users\\DELL\\Desktop\\data_scientist\\rnn\\m.txt', header=None ,error_bad_lines=False)
# df1 = pd.read_table('C:\\Users\\DELL\\Desktop\\data_scientist\\rnn\\f.txt', header=None ,error_bad_lines=False)

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

    total words: 22842
    


```python
# build index_to_word and word_to_index vectors
word_indices = dict((w, i) for i, w in enumerate(words))
indices_word = dict((i, w) for i, w in enumerate(words))
```


```python
text_sen=df1[0].tolist()
```


```python
maxlen = 5
step = 3
sentences = []
next_words = []

# cut the text in semi-redundant sequences of maxlen words
for i in range(0, len(words1) - maxlen, step):
    sentences.append(words1[i: i + maxlen])
    next_words.append(words1[i + maxlen])
print('nb sequences:', len(sentences))
```

    nb sequences: 7612
    


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
    lstm_1 (LSTM)                (None, 128)               11761152  
    _________________________________________________________________
    dense_1 (Dense)              (None, 22842)             2946618   
    _________________________________________________________________
    activation_1 (Activation)    (None, 22842)             0         
    =================================================================
    Total params: 14,707,770
    Trainable params: 14,707,770
    Non-trainable params: 0
    _________________________________________________________________
    


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


```python
from keras.models import load_model
model = load_model('C:\\Users\\DELL\\Desktop\\data_scientist\\rnn\\modelrnn5m.h5')
# model = load_model('C:\\Users\\DELL\\Desktop\\data_scientist\\rnn\\modelrnn5f.h5')
```


```python
#train the model, fit it to the data
model.fit(X, y,
          batch_size=128,
          epochs=20)
```

    Epoch 1/20
    7612/7612 [==============================] - 183s - loss: 7.1848 - acc: 0.2119   
    Epoch 2/20
    7612/7612 [==============================] - 182s - loss: 6.0556 - acc: 0.2191   
    Epoch 3/20
    7612/7612 [==============================] - 178s - loss: 5.6135 - acc: 0.2275   
    Epoch 4/20
    7612/7612 [==============================] - 179s - loss: 5.1222 - acc: 0.2380   
    Epoch 5/20
    7612/7612 [==============================] - 176s - loss: 4.5456 - acc: 0.2683   
    Epoch 6/20
    7612/7612 [==============================] - 182s - loss: 3.8828 - acc: 0.3307   
    Epoch 7/20
    7612/7612 [==============================] - 174s - loss: 3.1302 - acc: 0.4251   
    Epoch 8/20
    7612/7612 [==============================] - 168s - loss: 2.3240 - acc: 0.5674   
    Epoch 9/20
    7612/7612 [==============================] - 145s - loss: 1.6223 - acc: 0.7198   
    Epoch 10/20
    7612/7612 [==============================] - 149s - loss: 1.0388 - acc: 0.8437   
    Epoch 11/20
    7612/7612 [==============================] - 149s - loss: 0.6226 - acc: 0.9200   
    Epoch 12/20
    7612/7612 [==============================] - 143s - loss: 0.3768 - acc: 0.9561   
    Epoch 13/20
    7612/7612 [==============================] - 141s - loss: 0.2447 - acc: 0.9744   
    Epoch 14/20
    7612/7612 [==============================] - 141s - loss: 0.1687 - acc: 0.9815   
    Epoch 15/20
    7612/7612 [==============================] - 146s - loss: 0.1166 - acc: 0.9891   
    Epoch 16/20
    7612/7612 [==============================] - 142s - loss: 0.0826 - acc: 0.9936   
    Epoch 17/20
    7612/7612 [==============================] - 147s - loss: 0.0639 - acc: 0.9941   
    Epoch 18/20
    7612/7612 [==============================] - 139s - loss: 0.0497 - acc: 0.9944   
    Epoch 19/20
    7612/7612 [==============================] - 140s - loss: 0.0343 - acc: 0.9967   
    Epoch 20/20
    7612/7612 [==============================] - 141s - loss: 0.0331 - acc: 0.9963   
    




    <keras.callbacks.History at 0x2797b0645f8>




```python
#save the model
model.save('C:\\Users\\DELL\\Desktop\\data_scientist\\rnn\\modelrnn5m.h5')
# model.save('C:\\Users\\DELL\\Desktop\\data_scientist\\rnn\\modelrnn5f.h5')
```


```python
#generate new sentences
to_file=[]
for i in range(0,3):
    #select seed index- first word of the sentance
    start_index = random.randint(0, len(words1) - maxlen - 1)
    while(words1[start_index:start_index+1][0] == 'סוףהודעה'):
        start_index = random.randint(0, len(words1) - maxlen - 1)

    for diversity in [1.0]:
        print()
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
with open('C:\\Users\\DELL\\Desktop\\data_scientist\\rnn\\nn_new.txt', 'w', newline='\n', encoding='utf-8') as txt_file:
    writer = csv.writer(txt_file, delimiter='\n')
    writer.writerow(to_file)
```
