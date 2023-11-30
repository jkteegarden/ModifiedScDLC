import rpy2.robjects as robjects
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, RNN, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import argparse




def build_scDLC_scRNAseqClassifier(num_steps, num_classes, lstm_size, num_layers):
    inputs = Input(shape=(num_steps, 1))
    lstm_inputs = Dense(2 * num_steps, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.003))(inputs)

    
    cells = [tf.keras.layers.LSTMCell(lstm_size) for _ in range(num_layers)]
    cell = RNN(cells, return_sequences=True, return_state=True)


    lstm_outputs, _, _ = cell(lstm_inputs)

    #lstm_outputs = BatchNormalization()(lstm_outputs)


    lstm_outputs = Dropout(0.25)(lstm_outputs) #play with this number

    lstm_outputs = BatchNormalization()(lstm_outputs)

    # Flatten the LSTM output for the Dense layer
    flat_lstm_outputs = Flatten()(lstm_outputs)

    logits = Dense(num_classes, activation='sigmoid')(flat_lstm_outputs) #activation='sigmoid' for binary or 'softmax' for multi

    model = Model(inputs=inputs, outputs=logits)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy']) #sparse_categorical_crossentropy or binary_crossentropy

    return model



parser = argparse.ArgumentParser()


parser.add_argument('--num_classes', type=int, default=None, help='Number of classes')
parser.add_argument('--num_steps', type=int, default=100, help='Number of steps')
parser.add_argument('--batch_size', type=int, default=11, help='Batch size')
parser.add_argument('--lstm_size', type=int, default=64, help='LSTM size')
parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
parser.add_argument('--n_epoch', type=int, default=30, help='Number of epochs')
#parser.add_argument('--train_keep_prob', type=float, default=0.3, help='Training keep probability')


args = parser.parse_args()


num_classes = args.num_classes
num_steps = args.num_steps
batch_size = args.batch_size
lstm_size = args.lstm_size
num_layers = args.num_layers
n_epoch = args.n_epoch
#train_keep_prob = args.train_keep_prob

#Access R script
robjects.r.source('selectgene.r')

def main():
    data = pd.read_csv("data.csv", header=0, index_col=0)
    n = data.shape[0] - 1
    data = data.T
    X = data.iloc[:, 0:args.num_steps]
    X = np.array(X)
    Y = data.iloc[:, n] -1
    args.num_classes = Y.max() + 1

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    model = build_scDLC_scRNAseqClassifier(num_steps=args.num_steps, num_classes=args.num_classes,
                                           lstm_size=args.lstm_size, num_layers=args.num_layers
                                           #train_keep_prob=args.train_keep_prob
                                           )

   
    tensorboard_callback = TensorBoard(log_dir='C:/Users/tttja/modifiedscDLC/logs/', histogram_freq=1)

    
    model_checkpoint_callback = ModelCheckpoint(
        filepath='C:/Users/tttja/modifiedscDLC/logs/model_checkpoint.h5',
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        save_weights_only=True
    )

    

    test_data=list([X_test,y_test])
    #validation_data=(X_test, y_test)

    
    history = model.fit(X_train, y_train, validation_data=test_data, epochs=args.n_epoch,
                        batch_size=batch_size, callbacks=[tensorboard_callback, model_checkpoint_callback])
    
    
    #model.summary()

    X_train_df = pd.DataFrame(X_train)


    
    X_train_df.to_csv('X_train.csv', index=False)


    X_test_df = pd.DataFrame(X_test)


    
    X_test_df.to_csv('X_test.csv', index=False)
    



    
    y_train_df = pd.DataFrame(y_train)
    y_train_df.to_csv('y_train.csv', index=False)

    y_test_df = pd.DataFrame(y_test)
    y_test_df.to_csv('y_test.csv', index=False)
    
    return history

if __name__ == '__main__':
    main()
