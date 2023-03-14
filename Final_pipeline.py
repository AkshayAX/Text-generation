'''We can add the model inside of function,but for each call we need to load
the model and that's an unnecessary overhead'''
# models=joblib.load('model.pkl')

def predict(inputs,iter=1,models=model):
    'load model and tokenizers'
    encoder_input_tokenizer=joblib.load('encoder_input_tokenizer.pkl')
    decoder_input_tokenizer=joblib.load('decoder_input_tokenizer.pkl')
    max_sentencelen=22
    max_input_len=105
    targetArray=[] 
    target=inputs
    speaker=['A: ','B: ']
    
    '''for each iteration'''
    for j in range(iter):
        
        inputs=target
        print(speaker[j%2],inputs)
        j=j+1
        '''tokenize,pad and initilize initial state for input'''
        
        sequence=encoder_input_tokenizer.texts_to_sequences([inputs])
        sequence=tf.keras.preprocessing.sequence.pad_sequences(sequence,maxlen=max_input_len,
                                                     padding='post',
                                                    truncating='post'
                                                          )

        states=models.layers[0].initialize_states(1)
        encoder_output,lstm_h,lstm_C=models.layers[0](sequence,states)
        decoder_input=np.zeros((1,1))
        decoder_input[0,0]=decoder_input_tokenizer.word_index['<start>']
        target=''
        '''Generate output  for max output length times'''
        for i in range(max_sentencelen):
            decoder_output,lstm_h,lstm_C,attention_weights,context_vector =models.layers[1].one_step_decoder(decoder_input,encoder_output,lstm_h,lstm_C)
            decoder_input[0,0]=np.argmax(decoder_output)
            if decoder_input_tokenizer.index_word[decoder_input[0,0]]=='<end>':
                targetArray.append(target)
                break
            else:
                target=target+" "+decoder_input_tokenizer.index_word[decoder_input[0,0]]
    print('A :',target)
