
**Unpublished Results**

Sentence simplification is a task that produces a simplified sentence from any input
sentence. Sentence simplification systems can be useful tools for people whose first
language is not English, children, or the elderly to help with understanding. Initial
approaches to this task were primarily borrowed from neural machine translation
systems with transformations at the sentence level or with architectures based on
recurrent neural networks (RNNs). 
<br/>
<br/>
<br/>

**Model Training Steps**
1. sh generaotr.sh 预训练generator
2. sh generate_sample.sh 生成Negetive 数据，默认为5000
   通过修改config1/config_generate_sample.yaml可以设置Negetive的数据量 
3. sh discriminator_pretrain.sh 预训练discriminator
4. sh gan_train.sh 训练GAN
