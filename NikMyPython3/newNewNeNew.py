
    self.xData = self.inputs

    #xData = xData.view(-1, 28 * 28)
    #genFGen2 = genFGen2.view(-1, 28 * 28)
    #genFGen3 = genFGen3.squeeze()

    #self.genFgenFGen2 = self.flow_inv_model(self.z)

    #self.genFgenFGen2 = self.flow_inv_model(self.z)
    self.genFgenFGen2 = self.flow_inv_model(self.z)

    #self.genFgenFGen2 = self.flow_inv_model(self.z)
    #self.genFgenFGen2 = self.sampler_function(self.z)

    #self.genFgenFGen2 = self.flow_inv_model(self.z)
    #genFGen2 = genFgenFGen2

    self.xData = tf.reshape(self.xData, [-1, 28*28])
    self.genFGen2 = tf.reshape(self.genFgenFGen2, [-1, 28 * 28])

    #print(self.z)
    #adfasdfs

    self.genFGen3 = self.z
    self.genFGen3 = tf.reshape(self.genFGen3, [-1, 28 * 28])

    #device = args.device
    #second_term_loss2 = tf.zeros(1, device=device, requires_grad=False)
    #print(tf.pow((genFGen2[0, :] - xData), 2))
    #print(tf.reduce_sum(tf.pow((genFGen2[0, :] - xData), 2), 1))
    #asdfadsfdsaf
    #self.second_term_loss2 = tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((self.genFGen2[0, :] - self.xData), 2), 1)) ** 2)
    self.second_term_loss2 = tf.reduce_min(
      tf.sqrt(1e-17 + tf.reduce_sum(tf.pow((self.genFGen2[0, :] - self.xData), 2), 1)) ** 2)
    #for i in range(self.batch_size):
    for i in range(1, self.batch_size):
      #second_term_loss2 += torch.min(torch.sqrt((genFGen2[i, :] - xData).pow(2).sum(1)) ** 2)
      #self.second_term_loss2 += tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.pow((self.genFGen2[i, :] - self.xData), 2), 1)) ** 2)
      self.second_term_loss2 += tf.reduce_min(
        tf.sqrt(1e-17 + tf.reduce_sum(tf.pow((self.genFGen2[i, :] - self.xData), 2), 1)) ** 2)
    self.second_term_loss2 /= self.batch_size
    #second_term_loss2 = second_term_loss2.squeeze()

    #third_term_loss32 = torch.empty(self.batch_size, device=device, requires_grad=False)
    self.third_term_loss32 = tf.reduce_mean((tf.sqrt(1e-17 + tf.reduce_sum(tf.pow((self.genFGen3[0, :] - self.genFGen3), 2), 1))) / (
              1e-17 + tf.sqrt(1e-17 + tf.reduce_sum(tf.pow((self.genFGen2[0, :] - self.genFGen2), 2), 1))))
    #for i in range(self.batch_size):
    for i in range(1, self.batch_size):
      self.third_term_loss32 += tf.reduce_mean((tf.sqrt(1e-17 + tf.reduce_sum(tf.pow((self.genFGen3[i, :] - self.genFGen3), 2), 1))) / (
              1e-17 + tf.sqrt(1e-17 + tf.reduce_sum(tf.pow((self.genFGen2[i, :] - self.genFGen2), 2), 1))))
      #third_term_loss32[i] = torch.mean(third_term_loss22)
    #third_term_loss12 = torch.mean(third_term_loss32)
    self.third_term_loss12 = self.third_term_loss32 / self.batch_size

    #print(third_term_loss12)

    #print(second_term_loss2)
    #print(third_term_loss12)

    #asdfasdf

    #train_gen_para, train_jac = self.trainable_flow_model(self.flow_inv_model(self.z))
    #train_gen_para, train_jac = self.trainable_flow_model(genFgenFGen2)

    #train_gen_para, train_jac = self.trainable_flow_model(genFgenFGen2)

    #train_gen_para, train_jac = self.trainable_flow_model(genFgenFGen2)
    #train_gen_para, train_jac = self.flow_model(genFgenFGen2)



    #asdfzsfd



    #train_gen_para, train_jac = self.flow_model(genFgenFGen2)
    #train_gen_para, train_jac = self.flow_model(self.genFgenFGen2)

    #train_gen_para, train_jac = self.flow_model(self.genFgenFGen2)

    #train_gen_para, train_jac = self.flow_model(self.genFgenFGen2)
    train_gen_para, train_jac = self.trainable_flow_model(self.genFgenFGen2)

    #train_gen_para, train_jac = self.trainable_flow_model(self.flow_inv_model(self.z))
    self.train_log_likelihood = nvp_op.log_likelihood(train_gen_para, train_jac, self.prior) / self.batch_size

    #print((tf.reduce_mean(tf.exp(-self.train_log_likelihood))))
    #asdfasdfasdfs

    #self.train_log_likelihood = (tf.reduce_mean(tf.exp(-self.train_log_likelihood))) + (secondTerm) + (thirdTerm)
    #self.train_log_likelihood = (tf.reduce_mean(tf.exp(-self.train_log_likelihood))) + (self.second_term_loss2) + (self.third_term_loss12)

    #self.train_log_likelihood = (tf.reduce_mean(tf.exp(-self.train_log_likelihood))) + (self.second_term_loss2) + (self.third_term_loss12)

    #self.train_log_likelihood = (tf.reduce_mean(tf.exp(-self.train_log_likelihood))) + (self.second_term_loss2) + (self.third_term_loss12)
    self.train_log_likelihood = (tf.reduce_mean(tf.exp(-self.train_log_likelihood / 10000000))) + (self.second_term_loss2) + (
        self.third_term_loss12)

    #self.evaluate_neg_loglikelihood22(out, config)

    #self.evaluate_neg_loglikelihood22(out, config)
    #self.evaluate_neg_loglikelihood22(out, config)

