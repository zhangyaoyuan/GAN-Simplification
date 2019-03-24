import yaml
import time
import os
import logging
from argparse import ArgumentParser
import tensorflow as tf

from utils import DataUtil, AttrDict
from model import Model
from SARI_evaluation_multi import process_evaluation_sentence,SARIsent,process_evaluation_file_multi
from nltk.translate.bleu_score import sentence_bleu
from evaluate_esc1 import Evaluator #!!!

def train(config):
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    logger = logging.getLogger('')
    du = DataUtil(config=config)
    du.load_vocab(src_vocab=config.src_vocab,
                  dst_vocab=config.dst_vocab,
                  src_vocab_size=config.src_vocab_size,
                  dst_vocab_size=config.dst_vocab_size)

    model = Model(config=config)
    model.build_train_model()
    #model.build_generate(max_len=config.train.max_length,generate_devices=config.train.devices,optimizer=config.train.optimizer)
    step = -1
    max_BLEU=0
    max_SARI=0
    #mp_best_SARI = config.train.logdir + '/best_model_SARI'
    #mp_best_BLEU = config.train.logdir + '/best_model_BLEU'
    with model.graph.as_default():
        saver = tf.train.Saver(var_list=tf.global_variables())
        sess = tf.Session(config=sess_config)
        summary_writer = tf.summary.FileWriter(config.train.logdir, graph=model.graph)
        # Initialize all variables.
        sess.run(tf.global_variables_initializer())
        logger.info("1.Generator Pretrain Start !!! Model Save Directory :"+config.train.logdir)
        
        try:
            logger.info("2.Reload model from "+config.train.logdir)
            saver.restore(sess, tf.train.latest_checkpoint(config.train.logdir))
        except:
            logger.info('2.Failed to reload model')
        
    for epoch in range(1, config.train.num_epochs+1):
        for batch in du.get_training_batches_with_buckets():
            # Evaluate the Model
            start_time = time.time()
            step = sess.run(model.global_step)
            # Summary
            if step % config.train.summary_freq == 0:
                step, lr, gnorm, loss, acc, summary, _ = sess.run([model.global_step, model.learning_rate, model.grads_norm,model.loss, model.acc, model.summary_op, model.train_op],feed_dict={model.src_pl: batch[0], model.dst_pl: batch[1]})
                summary_writer.add_summary(summary, global_step=step)
            else:
                step, lr, gnorm, loss, acc, _ = sess.run([model.global_step, model.learning_rate, model.grads_norm,model.loss, model.acc, model.train_op],feed_dict={model.src_pl: batch[0], model.dst_pl: batch[1]})
                # Information
                logger.info('Epoch: {0}  Step: {1}  lr: {2:.4f}  grads: {3:.4f}  loss: {4:.4f}  acc: {5:.4f}  time: {6:.4f}'.format(epoch, step, lr, gnorm, loss, acc, time.time() - start_time))
            # Save model
            if step % config.train.save_freq == 0:
                mp = config.train.logdir + '/model_epoch_%d_step_%d' % (step/1000, step)
                logger.info('3.Save model in %s !!!' % mp)
                saver.save(sess, mp)
                 
                logging.info("4.Evaluate on BLEU and SARI !!!")
                output_t = "prepare_data_esc/esc_test.8turkers.clean.out."+str(step/1000)
                evaluator = Evaluator(config=config, out_file=output_t)
                SARI,BLEU=evaluator.translate() #beam search
                logger.info('9.Save Model Successfully !!!\n Epoch: {0} Step: {1} SARI: {2} BLEU: {3}!!!!'.format(epoch, step, SARI, BLEU))
        logger.info("Finish training.")
                


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config')
    args = parser.parse_args()
    # Read config
    config = AttrDict(yaml.load(open(args.config)))
    # Logger
    if not os.path.exists(config.train.logdir):
        os.makedirs(config.train.logdir)
    logging.basicConfig(filename='log/train_debug.log', level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    # Train
    train(config)
