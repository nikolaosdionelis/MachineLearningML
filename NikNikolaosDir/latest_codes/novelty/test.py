import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import pickle
import numpy as np
import os 

from utils import tensor2Var

import sklearn
import sklearn.manifold
import sklearn.metrics

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torchvision as tv

def test(args, vae_checkpoint, classifier_checkpoint, test_loader):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.use_gpu

    # save_root = 'output'

    # save_folder = os.path.join(save_root, model_id)

    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder)

    feature_size = args.feature_size

    if args.dataset == 'mnist':
        import mnist_model
        classifier = mnist_model.Classifier(args, feature_size)
        if args.feature_extractor == 'ae':
            vae = mnist_model.AE(args, feature_size)
        else:
            vae = mnist_model.VAE(args, feature_size)

    elif args.dataset == 'cifar':
        import cnn_model
        classifier = cnn_model.Classifier(args, feature_size)
        if args.feature_extractor == 'ae':
            vae = cnn_model.AE(args, feature_size)
        else:
            vae = cnn_model.VAE(args, feature_size)

    # checkpoint = torch.load(classifier_checkpoint, map_location=lambda storage, loc: storage)
    # classifier.load_state_dict(checkpoint['classifier_state_dict'])

    checkpoint = torch.load(vae_checkpoint, map_location=lambda storage, loc: storage)
    vae.load_state_dict(checkpoint['vae_state_dict'])

    if torch.cuda.is_available():
        print('CUDA ensabled.')
        classifier.cuda()
        vae.cuda()

    for p in classifier.parameters():
        p.requires_grad = False

    for p in vae.parameters():
        p.requires_grad = False

    classifier.eval()
    vae.eval()


    x = []
    y = []

    emb = []

    data_list = []

    r_list = []

    label_list = []

    for i, (data, label) in enumerate(test_loader.get_iter(shuffle=False)):
        data = tensor2Var(data)

        # edge = tensor2Var(edge)

        feat = vae.get_features(data)
        # feat = vae.get_features(edge)

        emb.append(feat.squeeze().cpu().numpy())

        r, mean, _ = vae(data)

        # r, mean, _ = vae(edge)

        # noise_in = tensor2Var(torch.FloatTensor(mean.size()).uniform_(-args.beta_1, args.beta_1))
        # noise_out = tensor2Var(torch.FloatTensor(mean.size()).uniform_(-args.beta_2, args.beta_2))

        # r = vae.decode(torch.clamp(mean + noise_in, -1.0, 1.0))

        # r_out = vae.decode(torch.clamp(mean + noise_out, -1.0, 1.0))

        # pred = classifier(feat)

        data_list.append(data.cpu().numpy())
        r_list.append(r.cpu().numpy())
        label_list.append(label.cpu().numpy())

        # if i == 0:
        #     tv.utils.save_image(data * 0.5 + 0.5, 
        #         os.path.join(args.log_folder, 'test_origin.png'), nrow=10)

        #     # tv.utils.save_image(edge * 0.5 + 0.5, 
        #     #     os.path.join(args.log_folder, 'test_origin_edge.png'), nrow=10)

        #     tv.utils.save_image(r * 0.5 + 0.5, 
        #         os.path.join(args.log_folder, 'test_reconstruct.png'), nrow=10)

            # tv.utils.save_image(r_out * 0.5 + 0.5, 
            #     os.path.join(args.log_folder, 'test_reconstruct_out.png'), nrow=10)

        # pred = -((r - data) ** 2).reshape(data.shape[0], -1).mean(1)

        pred = -((r - data) ** 2).reshape(data.shape[0], -1).mean(1)

        x.append(pred.cpu().numpy())

        y.append(label.cpu().numpy())

    data_list = np.vstack(data_list)
    r_list = np.vstack(r_list)
    label_list = np.hstack(label_list)

    test_data = []
    test_r = []
    
    # print(label_list.shape)

    pos = (label_list == 0)
    # print(pos.shape)
    test_data.append(data_list[pos][:90])
    test_r.append(r_list[pos][:90])

    pos = (label_list == 1)
    # print(pos.shape)
    test_data.append(data_list[pos][:10])
    test_r.append(r_list[pos][:10])

    test_data = torch.from_numpy(np.vstack(test_data))
    test_r = torch.from_numpy(np.vstack(test_r))

    # print(test_data.shape)

    tv.utils.save_image(test_data * 0.5 + 0.5, 
        os.path.join(args.log_folder, 'test_origin.png'), nrow=10)

    # tv.utils.save_image(edge * 0.5 + 0.5, 
    #     os.path.join(args.log_folder, 'test_origin_edge.png'), nrow=10)

    tv.utils.save_image(test_r * 0.5 + 0.5, 
        os.path.join(args.log_folder, 'test_reconstruct.png'), nrow=10)




    x = np.hstack(x)
    y = np.hstack(y)

    emb = np.vstack(emb)

    print(x)
    print(y)

    print((x < 0.9).sum())

    # index = np.argsort(x)

    # x = x[index]
    # y = y[index]

    # print((y==0).sum(), (y==1).sum())

    print(x.shape, y.shape)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, x, pos_label=1)
    
    auc = sklearn.metrics.auc(fpr, tpr)
    print(auc)

    with open(os.path.join(args.log_folder, 'auc.txt'), 'w') as f:
        f.write('%.4f\n' % auc)
    # print(np.sum(y))


    ##########################################################
    # plot
    #

    tsne = sklearn.manifold.TSNE(2)

    # print(all_emb.shape)

    emb = emb[:1000]
    y = y[:1000]

    emb = tsne.fit_transform(emb)

    plt.clf()

    l_list = []
    for i in range(2):
        pos = y == i

        l_list.append(plt.scatter(emb[pos, 0], emb[pos, 1]))

    plt.legend(l_list, ['novelty data', 'train data'])

    save_path = os.path.join(args.log_folder, 'test_embedding.png')

    plt.savefig(save_path)




