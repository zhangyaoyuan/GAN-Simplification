import codecs
import os
import path
import tensorflow as tf
import numpy as np
import yaml
import time
import logging
from tempfile import mkstemp
from argparse import ArgumentParser

from model import Model, INT_TYPE
from utils import DataUtil, AttrDict
from SARI_evaluation_multi import process_evaluation_sentence,SARIsent,process_evaluation_file_multi
from nltk.translate.bleu_score import sentence_bleu

class Evaluator(object):
    """
    Evaluate the model.
    """
    def __init__(self, config, model):
        self.config = config
        # Load model
        self.model = Model(config)
        self.model.build_test_model()

        self.du = DataUtil(config)
        self.du.load_vocab()
        
        # Create session
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True  
        sess_config.allow_soft_placement = True
        self.sess = tf.Session(config=sess_config, graph=self.model.graph)    
        # Restore model.
        with self.model.graph.as_default():
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(self.sess, model)
            self.model_name=model
            #saver.restore(self.sess, "save_model1_3000tok/pre_generator_all/model_epoch_12_step_12000")
 
    def __del__(self):
        self.sess.close()

    def greedy_search(self, X):
        """
        Greedy search.
        Args:
            X: A 2-d array with size [n, src_length], source sentence indices.
        Returns:
            A 2-d array with size [n, dst_length], destination sentence indices.
        """
        encoder_output = self.sess.run(self.model.encoder_output, feed_dict={self.model.src_pl: X})
        preds = np.ones([X.shape[0], 1], dtype=INT_TYPE) * 2 # <S>
        finish = np.zeros(X.shape[0:1], dtype=np.bool)
        for i in range(config.test.max_target_length):
            last_preds = self.sess.run(self.model.preds, feed_dict={self.model.encoder_output: encoder_output,
                                                                    self.model.decoder_input: preds})
            finish += last_preds == 3   # </S>
            if finish.all():
                break
            preds = np.concatenate((preds, last_preds[:, None]), axis=1)

        return preds[:, 1:]

    def beam_search(self, X):
        """
        Beam search with batch inputs.
        Args:
            X: A 2-d array with size [n, src_length], source sentence indices.
        Returns:
            A 2-d array with size [n, dst_length], target sentence indices.
        """

        beam_size, batch_size = self.config.test.beam_size, X.shape[0]
        inf = 1e10

        def get_bias_scores(scores, bias):
            """
            If a sequence is finished, we only allow one alive branch. This function aims to give one branch a zero score
            and the rest -inf score.
            Args:
                scores: A real value array with shape [batch_size * beam_size, beam_size].
                bias: A bool array with shape [batch_size * beam_size].
            Returns:
                A real value array with shape [batch_size * beam_size, beam_size].
            """
            b = np.array([0.0] + [-inf] * (beam_size - 1))
            b = np.repeat(b[None,:], batch_size * beam_size, axis=0)  # [batch * beam_size, beam_size]
            return scores * (1 - bias[:, None]) + b * bias[:, None]

        def get_bias_preds(preds, bias):
            """
            If a sequence is finished, all of its branch should be </S> (3).
            Args:
                preds: A int array with shape [batch_size * beam_size, beam_size].
                bias: A bool array with shape [batch_size * beam_size].
            Returns:
                A int array with shape [batch_size * beam_size].
            """
            return preds * (1 - bias[:, None]) + bias[:, None] * 3

        # Get encoder outputs.
        encoder_output = self.sess.run(self.model.encoder_output, feed_dict={self.model.src_pl: X})
        # Prepare beam search inputs.
        encoder_output = np.repeat(encoder_output, beam_size, axis=0)   # shape: [batch_size * beam_size, hidden_units]
        preds = np.ones([batch_size * beam_size, 1], dtype=INT_TYPE) * 2  # [[<S>, <S>, ..., <S>]], shape: [batch_size * beam_size, 1]
        scores = np.array(([0.0] + [-inf] * (beam_size - 1)) * batch_size)  # [0, -inf, -inf ,..., 0, -inf, -inf, ...], shape: [batch_size * beam_size]
        for i in range(self.config.test.max_target_length):
            # Whether sequences finished.
            bias = np.equal(preds[:, -1], 3)   # </S>?
            # If all sequences finished, break the loop.
            if bias.all():
                break

            # Expand the nodes.
            last_k_preds, last_k_scores = \
                self.sess.run([self.model.k_preds, self.model.k_scores],
                              feed_dict={self.model.encoder_output: encoder_output,
                                         self.model.decoder_input: preds})  # [batch_size * beam_size, beam_size]

            last_k_preds = get_bias_preds(last_k_preds, bias)
            last_k_scores = get_bias_scores(last_k_scores, bias)
            # Shrink the search range.
            scores = scores[:, None] + last_k_scores  # [batch_size * beam_size, beam_size]
            scores = scores.reshape([batch_size, beam_size**2])  # [batch_size, beam_size * beam_size]

            # Reserve beam_size nodes.
            k_indices = np.argsort(scores)[:, -beam_size:]  # [batch_size, beam_size]
            k_indices = np.repeat(np.array(list(range(0, batch_size))), beam_size) * beam_size**2 + k_indices.flatten()  # [batch_size * beam_size]
            scores = scores.flatten()[k_indices]  # [batch_size * beam_size]
            last_k_preds = last_k_preds.flatten()[k_indices]
            preds = preds[k_indices // beam_size]
            preds = np.concatenate((preds, last_k_preds[:, None]), axis=1)  # [batch_size * beam_size, i]

        scores = scores.reshape([batch_size, beam_size])
        preds = preds.reshape([batch_size, beam_size, -1])  # [batch_size, beam_size, max_length]
        lengths = np.sum(np.not_equal(preds, 3), axis=-1)   # [batch_size, beam_size]
        lp = ((5 + lengths) / (5 + 1)) ** self.config.test.lp_alpha   # Length penalty
        scores /= lp                                                  # following GNMT.
        max_indices = np.argmax(scores, axis=-1)   # [batch_size]
        max_indices += np.array(list(range(batch_size))) * beam_size
        preds = preds.reshape([batch_size * beam_size, -1])
        logging.debug(scores.flatten()[max_indices])
        return preds[max_indices][:, 1:]

    def loss(self, X, Y):
        return self.sess.run(self.model.loss_sum, feed_dict={self.model.src_pl: X, self.model.dst_pl: Y})

    def translate(self):
        logging.info('5.Translate prepare_data/test.8turkers.clean.src.big !!!!')
        output="prepare_data/test.8turkers.clean.output"+str(self.model_name[-4:-2])
        logging.info('6.The generated output was saved in {} !!!!'.format(output))
        fd = codecs.open(output,'w','utf8')
        count = 1
        start = time.time()
        for X in self.du.get_test_batches(set_src_path="prepare_data/test.8turkers.clean.src.big",set_batch=64):
            # Beam Search !!
            Y = self.beam_search(X)
            sents = self.du.indices_to_words(Y)
            for sent in sents:
                print(str(count)+". "+sent)
                if count<=359:
                    fd.write(sent+'\n')
                count +=1   
        fd.close()
        logging.info('7.Beam Search Output Generation Done !!!')
        #os.system("sed -r 's/(@@ )|(@@ ?$)//g' %s > %s" % (tmp, self.config.test.out_path))
        logging.info('8.The Result File was saved in %s.' % output)
        source = "prepare_data/test.8turkers.clean.src"
        target = ["prepare_data/test.8turkers0.clean.dst","prepare_data/test.8turkers1.clean.dst","prepare_data/test.8turkers2.clean.dst","prepare_data/test.8turkers3.clean.dst","prepare_data/test.8turkers4.clean.dst","prepare_data/test.8turkers5.clean.dst","prepare_data/test.8turkers6.clean.dst","prepare_data/test.8turkers7.clean.dst","prepare_data/test.8turkers.clean.dst"]
        SARI_results,BLEU_results = process_evaluation_file_multi(source, output, target)
        logging.info('9. Model: {}  SARI: {} BLEU: {} !!!!'.format(self.model_name,SARI_results,BLEU_results))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config')
    args = parser.parse_args()
    # Read config
    config = AttrDict(yaml.load(open(args.config)))
    # Logger
    logging.basicConfig(level=logging.INFO)
    model_list=[]
    pre = "save_model_large/pre_generator_all/"
    for i in range(20,35):
        model_i=pre+"model_epoch_"+str(i)+"_step_"+str(i)+"000"
        model_list.append(model_i)
    print (model_list)  
   
    for i in model_list:
        try:
            evaluator = Evaluator(config,i)
        except:
            logging.info("Reload this file Unsuccessfully, No such model") 
            continue
        evaluator.translate()
    logging.info("Done")
