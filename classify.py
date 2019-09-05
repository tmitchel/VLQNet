from keras.models import load_model
from array import array
from glob import glob
import pandas as pd
import collections
import uproot
import ROOT
import sys
from os import environ, path, mkdir, listdir
environ['KERAS_BACKEND'] = 'tensorflow'


def getGuess(df, index):
    """
    Try to grab the NN output value if there is one

    Parameters:
        df (pandas.DataFrame): all events in the current file
        index (int): event index to try and grab

    Returns:
        prob_sig (float): NN output value provided one exists for
        this index. Otherwise, returns -999
    """
    try:
        prob_sig = df.loc[index, 'prob_sig']
    except:
        prob_sig = -999
    return prob_sig


def build_filelist(input_dir):
    """
    Build list of files to process. Returns a map in case files need to be
    split into groups at a later stage of the analysis.

    Parameters:
        input_dir (string): name of input directory

    Returns:
        filelist (map): list of files that need to be processed
    """
    filelist = collections.defaultdict(list)
    filelist['all'] = [fname for fname in glob('{}/*.root'.format(input_dir))]
    return filelist


def main(args):
    model = load_model('models/{}.hdf5'.format(args.model))  # load the NN model with weights
    all_data = pd.HDFStore(args.input_name)  # load the rescaled data

    # make a place for output files if it doesn't exist
    if not path.isdir(args.output_dir):
        mkdir(args.output_dir)

    filelist = build_filelist(args.input_dir)  # get files to process
    for _, files in filelist.iteritems():
        for ifile in files:
            print 'Processing file: {}'.format(ifile)

            # now let's try and get this into the root file
            root_file = ROOT.TFile(ifile, 'READ')

            # create output file and copy things input->output
            fname = ifile.split('/')[-1].replace('.root', '')
            fout = ROOT.TFile('{}/{}.root'.format(args.output_dir, fname), 'recreate')  # make new file for output
            fout.cd()
            allEvents = root_file.Get('allEvents').Clone()
            ana = root_file.Get('allEvents').Clone()
            allEvents.Write()
            ana.Write()

            data = all_data['nominal']  # load the correct tree (only nominal for now)

            # get dataframe for this sample
            sample = data[(data['sample_names'] == fname)]

            # drop all variables not going into the network
            to_classify = sample[[
                'lepPt', 'leadJetPt', 'met', 'ST', 'HT', 'DPHI_Metlep',
                'DPHI_LepJet', 'DR_LepCloJet', 'bVsW_ratio', 'Ext_Jet_TransPt',
                'Angle_MuJet_Met'
            ]]

            # do the classification
            guesses = model.predict(to_classify.values, verbose=False)
            out = sample.copy()
            out['prob_sig'] = guesses[:, 0]
            out.set_index('idx', inplace=True)

            # copy the input tree to the new file
            itree = root_file.Get('RTree')
            ntree = itree.CloneTree(-1, 'fast')

            # create a new branch and loop through guesses to fill the new tree.
            # this will be much easier once uproot supports writing TTrees
            NN_sig = array('f', [0.])
            disc_branch_sig = ntree.Branch('NN_disc', NN_sig, 'NN_disc/F')
            evt_index = 0
            for _ in itree:
                if evt_index % 100000 == 0:
                    print 'Processing: {}% completed'.format((evt_index*100)/ntree.GetEntries())

                NN_sig[0] = getGuess(out, evt_index)

                evt_index += 1
                fout.cd()
                disc_branch_sig.Fill()
            fout.cd()
            ntree.Write()
        root_file.Close()
        fout.Close()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', action='store', dest='model', default='testModel', help='name of model to use')
    parser.add_argument('--input', '-i', action='store', dest='input_name',
                        default='test', help='name of input dataset')
    parser.add_argument('--dir', '-d', action='store', dest='input_dir',
                        default='input_files/etau_stable_Oct24', help='name of ROOT input directory')
    parser.add_argument('--out', '-o', action='store', dest='output_dir', default='output_files/example')

    main(parser.parse_args())
