from keras.models import Model
from keras.layers import Dense, Embedding, Input, Conv1D, Dropout, CuDNNLSTM, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint

def cudnnlstm_model(maxlen, max_features, embed_size, embedding_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
    x = Dropout(0.25)(x)
    x = Conv1D(2 * embed_size, kernel_size=3)(x)
    prefilt = Conv1D(2 * embed_size, kernel_size=3)(x)
    
    x = prefilt
    for strides in [1, 1, 2]:
        x = Conv1D(128 * 2 ** strides, strides=strides, kernel_size=3)(x)
    
    x_f = CuDNNLSTM(512)(x)
    x_b = CuDNNLSTM(512)(x)
    
    x = concatenate([x_f, x_b])
    x = Dropout(0.5)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    
    return model

def train_model(X_train, y_train, X_val, y_val, model, batch_size=2048, epochs=7):
    weight_path = "early_weights.hdf5"
    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    callbacks = [checkpoint, early_stopping]
    
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=callbacks)
    
    return model, history
