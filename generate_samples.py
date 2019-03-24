import yaml
import time
import os
import logging
from argparse import ArgumentParser
import tensorflow as tf

from utils import DataUtil, AttrDict
from model import Model
from cnn_discriminator import DisCNN
from share_function import deal_generated_samples
from share_function import extend_sentence_to_maxlen
from SARI_evaluation_multi import process_evaluation_sentence,SARIsent,process_evaluation_file_multi
from nltk.translate.bleu_score import sentence_bleu
from evaluate_tok1 import Evaluator

def generate_samples(config):
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True

    default_graph = tf.Graph()
    with default_graph.as_default():
        sess = tf.Session(config=sess_config, graph=default_graph)

        logger = logging.getLogger('')
        du = DataUtil(config=config)
        du.load_vocab()

        generator = Model(config=config, graph=default_graph, sess=sess)
        generator.build_train_model()
        generator.build_generate(max_len=config.train.max_length,
                                 generate_devices=config.train.devices,
                                 optimizer=config.train.optimizer)

        generator.init_and_restore(config.train.modelFile)

        infile=config.train.src_path
        outfile=config.train.out_path
        refile=[config.train.dst_path]
        generate_batch= config.train.batch_size
        print("begin generate the data and save the negative in %s" % outfile)
        generator.generate_and_save(du, infile, generate_batch, outfile)
        print("generate the data done!")
        SARI_results, BLEU_results = process_evaluation_file_multi(infile, outfile, refile)  
        logging.info("SARI: {}, BLEU: {}".format(SARI_results, BLEU_results))
        '''
        infile=config.train.dev_src_path
        outfile=config.train.dev_out_path
        refile = ["prepare_data/test.8turkers0.clean.dst","prepare_data/test.8turkers1.clean.dst","prepare_data/test.8turkers2.clean.dst","prepare_data/test.8turkers3.clean.dst","prepare_data/test.8turkers4.clean.dst","prepare_data/test.8turkers5.clean.dst","prepare_data/test.8turkers6.clean.dst","prepare_data/test.8turkers7.clean.dst","prepare_data/test.8turkers.clean.dst"]
        generate_batch= config.train.batch_size
        print("begin generate the data and save the negative in %s" % outfile)
        generator.generate_and_save(du, infile, generate_batch, outfile)
        print("generate the data done!")
        SARI_results, BLEU_results = process_evaluation_file_multi(infile, outfile, refile)  
        logging.info("SARI: {}, BLEU: {}".format(SARI_results, BLEU_results))
        '''
        # good_train.5000  SARI: 0.3718863844751156, BLEU: 0.5174396967286163
        # good_train.10000 SARI: 0.3740540092464015, BLEU: 0.5042116225846289
        # test.8turker     SARI: 0.3446525818136698, BLEU: 0.7470631493263347
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config')
    args = parser.parse_args()
    # Read config
    config = AttrDict(yaml.load(open(args.config)))
    # Logger
    if not os.path.exists(config.train.logdir):
        os.makedirs(config.train.logdir)
    logging.basicConfig(filename=config.train.logdir+'/train.log', level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    # Train
    generate_samples(config)
