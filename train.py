import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from time import time
from os import environ
from visualize import discPlot
environ['KERAS_BACKEND'] = 'tensorflow'

wjet_names = [
    'WJets_Ht-100To200_vlqAna',
    'WJets_Ht-200To400_vlqAna',
    'WJets_Ht-400To600_vlqAna',
    'WJets_Ht-600To800_vlqAna',
    'WJets_Ht-800To1200_vlqAna',
    'WJets_Ht-1200To2500_vlqAna',
    'WJets_Ht-2500ToInf_vlqAna',
]


def main(args):
    data = pd.HDFStore(args.input)['nominal']
    # define training variables
    training_variables = [
        'lepPt', 'leadJetPt', 'met', 'ST', 'HT', 'DPHI_Metlep',
        'DPHI_LepJet', 'DR_LepCloJet', 'bVsW_ratio', 'Ext_Jet_TransPt',
        'Angle_MuJet_Met'
    ]

    nvars = len(training_variables)

    # build the model
    model = Sequential()
    model.add(Dense(nvars*2, input_shape=(nvars,), name='input', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(nvars, name='hidden', activation='relu', kernel_initializer='normal'))
    model.add(Dense(1, name='output', activation='sigmoid', kernel_initializer='normal'))
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    # build callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=50),
        ModelCheckpoint('models/{}.hdf5'.format(args.model), monitor='val_loss',
                        verbose=0, save_best_only=True,
                        save_weights_only=False, mode='auto',
                        period=1
                        ),
        TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=200, write_grads=False, write_images=True)
    ]

    # get the data for the two-classes to discriminate
    training_processes = data[
        (data['sample_names'].str.contains('TprimeBToBW_M')) | (data['sample_names'].str.contains('WJets'))
    ]

    # apply VBF category selection
    # vbf_processes = training_processes[
    #     # selection goes here
    # ]
    vbf_processes = training_processes
    print vbf_processes, vbf_processes['sample_names'].unique()

    sig_df = vbf_processes[(vbf_processes['sample_names'].str.contains('TprimeBToBW_M'))]
    bkg_df = vbf_processes[(vbf_processes['sample_names'].str.contains('WJets'))]

    print 'No. Signal Events:     {}'.format(len(sig_df))
    print 'No. Background Events: {}'.format(len(bkg_df))

    # reweight to have equal events per class
    scaleto = max(len(sig_df), len(bkg_df))
    sig_df.loc[:, 'evtwt'] = sig_df['evtwt'].apply(lambda x: x*scaleto/len(sig_df))
    bkg_df.loc[:, 'evtwt'] = bkg_df['evtwt'].apply(lambda x: x*scaleto/len(bkg_df))
    selected_events = pd.concat([sig_df, bkg_df])

    # remove all columns except those needed for training
    training_dataframe = selected_events[training_variables + ['signal', 'evtwt']]

    training_data, testing_data, training_labels, testing_labels, training_weights, _ = train_test_split(
        training_dataframe[training_variables].values, training_dataframe['signal'].values, training_dataframe['evtwt'].values,
        test_size=0.05, random_state=7
    )

    # train the model (max 10,000 epochs, but EarlyStopping should stop it way before that).
    # Each batch is 1024 events, randomly shuffled, and 25% of the training data is used
    # for validation
    _ = model.fit(training_data, training_labels, shuffle=True,
                  epochs=10000, batch_size=1024, verbose=True,
                  callbacks=callbacks, validation_split=0.25, sample_weight=training_weights
                  )

    # make the discriminant plot if you want it
    if not args.dont_plot:
        test_sig, test_bkg = [], []
        for i in range(len(testing_labels)):
            if testing_labels[i] == 1:
                test_sig.append(testing_data[i, :])
            elif testing_labels[i] == 0:
                test_bkg.append(testing_data[i, :])

        train_sig, train_bkg = [], []
        for i in range(len(training_labels)):
            if training_labels[i] == 1:
                train_sig.append(training_data[i, :])
            elif training_labels[i] == 0:
                train_bkg.append(training_data[i, :])

        discPlot('NN_disc_{}'.format(args.model), model, np.array(train_sig),
                 np.array(train_bkg), np.array(test_sig), np.array(test_bkg))


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', action='store', dest='model',
                        default='testModel', help='name of the model to train')
    parser.add_argument('--input', '-i', action='store', dest='input', default='test', help='full name of input file')
    # parser.add_argument('--masspoint', '-m', required=True, help='mass point to train')
    parser.add_argument('--background', '-b', action='store', dest='background',
                        default='ZTT.root', help='name of background file')
    parser.add_argument('--dont-plot', action='store_true', dest='dont_plot', help='don\'t make training plots')

    main(parser.parse_args())
