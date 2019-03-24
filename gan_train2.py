import yaml
import time
import os
import sys
import numpy as np
import logging
from argparse import ArgumentParser
import tensorflow as tf

from utils import DataUtil, AttrDict
from model import Model
from cnn_discriminator import DisCNN
from share_function import deal_generated_samples
from share_function import deal_generated_samples_to_maxlen
from share_function import extend_sentence_to_maxlen
from share_function import prepare_gan_dis_data
from share_function import FlushFile
from SARI_evaluation_multi import process_evaluation_sentence,SARIsent,process_evaluation_file_multi
from nltk.translate.bleu_score import sentence_bleu
from evaluate_tok1 import Evaluator

def gan_train(config):
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True

    default_graph=tf.Graph()
    with default_graph.as_default():
        sess = tf.Session(config=sess_config, graph=default_graph)

        logger = logging.getLogger('')
        du = DataUtil(config=config)
        du.load_vocab(src_vocab=config.generator.src_vocab,
                      dst_vocab=config.generator.dst_vocab,
                      src_vocab_size=config.src_vocab_size,
                      dst_vocab_size=config.dst_vocab_size)

        generator = Model(config=config, graph=default_graph, sess=sess)
        generator.build_train_model()
        generator.build_generate(max_len=config.generator.max_length,generate_devices=config.generator.devices,optimizer=config.generator.optimizer)
        generator.build_rollout_generate(max_len=config.generator.max_length,roll_generate_devices=config.generator.devices)
        generator.init_and_restore(modelFile=config.generator.modelFile)

        #这个变量是什么？！filters？
        dis_filter_sizes = [i for i in range(1, config.discriminator.dis_max_len, 4)]
        dis_num_filters = [(100 + i * 10) for i in range(1, config.discriminator.dis_max_len, 4)]

        discriminator = DisCNN(
            sess=sess,
            max_len=config.discriminator.dis_max_len,
            num_classes=2,
            vocab_size=config.dst_vocab_size,
            vocab_size_s=config.src_vocab_size,
            batch_size=config.discriminator.dis_batch_size,
            dim_word=config.discriminator.dis_dim_word,
            filter_sizes=dis_filter_sizes,
            num_filters=dis_num_filters,
            source_dict=config.discriminator.dis_src_vocab,
            target_dict=config.discriminator.dis_dst_vocab,
            gpu_device=config.discriminator.dis_gpu_devices,
            positive_data=config.discriminator.dis_positive_data,
            negative_data=config.discriminator.dis_negative_data,
            source_data=config.discriminator.dis_source_data,
            dev_positive_data=config.discriminator.dis_dev_positive_data,
            dev_negative_data=config.discriminator.dis_dev_negative_data,
            dev_source_data=config.discriminator.dis_dev_source_data,
            max_epoches=config.discriminator.dis_max_epoches,
            dispFreq=config.discriminator.dis_dispFreq,
            saveFreq=config.discriminator.dis_saveFreq,
            saveto=config.discriminator.dis_saveto,
            reload=config.discriminator.dis_reload,
            clip_c=config.discriminator.dis_clip_c,
            optimizer=config.discriminator.dis_optimizer,
            reshuffle=config.discriminator.dis_reshuffle,
            scope=config.discriminator.dis_scope
        )

        batch_train_iter = du.get_training_batches(
            set_train_src_path=config.generator.src_path,
            set_train_dst_path=config.generator.dst_path,
            set_batch_size=config.generator.batch_size,
            set_max_length=config.generator.max_length
        )
        
        max_SARI_results=0.32 #!!
        max_BLEU_results=0.77 #!!
        def evaluation_result(generator,config):
            nonlocal max_SARI_results 
            nonlocal max_BLEU_results

            # 在test dataset 上开始验证
            logging.info("Max_SARI_results: {}".format(max_SARI_results))
            logging.info("Max_BLEU_results: {}".format(max_BLEU_results))
            output_t = "prepare_data/test.8turkers.clean.out.gan"

            # Beam Search 8.turkers dataset
            evaluator = Evaluator(config=config, out_file=output_t)
            #logging.info("Evaluate on BLEU and SARI")
            SARI_results, BLEU_results = evaluator.translate()
            logging.info(" Current_SARI is {} \n Current_BLEU is {}".format(SARI_results,BLEU_results))
            
            if SARI_results >= max_SARI_results or BLEU_results >= max_BLEU_results:
                if SARI_results >= max_SARI_results:
                    max_SARI_results=SARI_results
                    logging.info("SARI Update Successfully !!!")
                if BLEU_results >= max_BLEU_results:
                    logging.info("BLEU Update Successfully !!!")
                    max_BLEU_results=BLEU_results
                return True
            else:
                return False
            
 
        for epoch in range(1, config.gan_iter_num + 1):#10000
            for gen_iter in range(config.gan_gen_iter_num):#1
                batch_train = next(batch_train_iter)
                x, y_ground = batch_train[0], batch_train[1]
                y_sample = generator.generate_step(x)
                logging.info("1. Policy Gradient Training !!!")
                y_sample_dealed, y_sample_mask = deal_generated_samples(y_sample, du.dst2idx)#将y_sample数字矩阵用0补齐长度
                x_to_maxlen = extend_sentence_to_maxlen(x, config.generator.max_length)#将x数字矩阵用0补齐长度
                x_str=du.indices_to_words(x, 'dst')
                ground_str=du.indices_to_words(y_ground, 'dst')
                sample_str=du.indices_to_words(y_sample, 'dst')
                
              
                # Rewards = D(Discriminator) + Q(BLEU socres)
                logging.info("2. Calculate the Reward !!!")
                rewards = generator.get_reward(x=x,
                                               x_to_maxlen=x_to_maxlen,
                                               y_sample=y_sample_dealed,
                                               y_sample_mask=y_sample_mask,
                                               rollnum=config.rollnum,
                                               disc=discriminator,
                                               max_len=config.discriminator.dis_max_len,
                                               bias_num=config.bias_num,
                                               data_util=du)
                # Police Gradient 更新Generator模型
                logging.info("3. Update the Generator Model !!!")
                loss = generator.generate_step_and_update(x, y_sample_dealed, rewards)
                #logging.info("The reward is ",rewards)
                #logging.info("The loss is ",loss)
               
                #update_or_not_update=evaluation_result(generator,config)
                #if update_or_not_update:
                # 保存Generator模型
                logging.info("4. Save the Generator model into %s" % config.generator.modelFile)
                generator.saver.save(generator.sess, config.generator.modelFile)

                if config.generator.teacher_forcing:

                    logging.info("5. Doing the Teacher Forcing begin!")
                    y_ground, y_ground_mask = deal_generated_samples_to_maxlen(
                                                y_sample=y_ground,
                                                dicts=du.dst2idx,
                                                maxlen=config.discriminator.dis_max_len)

                    rewards_ground=np.ones_like(y_ground)
                    rewards_ground=rewards_ground*y_ground_mask
                    loss = generator.generate_step_and_update(x, y_ground, rewards_ground)
                    #logging.info("The teacher forcing reward is ", rewards_ground)
                    #logging.info("The teacher forcing loss is ", loss)
            
            logging.info("5. Evaluation SARI and BLEU")
            update_or_not_update=evaluation_result(generator,config)
            if update_or_not_update:
                #保存Generator模型
                generator.saver.save(generator.sess, config.generator.modelFile)

            data_num = prepare_gan_dis_data(
                train_data_source=config.generator.src_path,
                train_data_target=config.generator.dst_path,
                gan_dis_source_data=config.discriminator.dis_source_data,
                gan_dis_positive_data=config.discriminator.dis_positive_data,
                num=config.generate_num,
                reshuf=True
            )

            logging.info("8.Generate  Negative Dataset for Discriminator !!!")
            # 生成negative数据集
            generator.generate_and_save(data_util=du,
                                        infile=config.discriminator.dis_source_data,
                                        generate_batch=config.discriminator.dis_batch_size,
                                        outfile=config.discriminator.dis_negative_data
                                      )

            logging.info("9.Negative Dataset was save in to %s." %config.discriminator.dis_negative_data)
            logging.info("10.Finetuen the discriminator begin !!!!!")

            discriminator.train(max_epoch=config.gan_dis_iter_num,
                                positive_data=config.discriminator.dis_positive_data,
                                negative_data=config.discriminator.dis_negative_data,
                                source_data=config.discriminator.dis_source_data
                                )
            discriminator.saver.save(discriminator.sess, discriminator.saveto)
            logging.info("11.Finetune the discrimiantor done !!!!")

        logging.info('Reinforcement training done!')

if __name__ == '__main__':
    sys.stdout = FlushFile(sys.stdout)
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config')
    args = parser.parse_args()
    # Read config
    config = AttrDict(yaml.load(open(args.config)))
    # Logger
    if not os.path.exists(config.logdir):
        os.makedirs(config.logdir)
    logging.basicConfig(filename=config.logdir+'/train.log', level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)
    # Train
    gan_train(config)

