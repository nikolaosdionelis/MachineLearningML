import os

import all_data
from utils import create_logger, makedirs

import argparse

def argparser():
    parser = argparse.ArgumentParser(description='Difference-Seeking GAN')

    parser.add_argument('--todo', default='baseline')
    parser.add_argument('--dataset', required=True, help='dataset type')

    parser.add_argument('--data_root', default='/home/lab530/Data', help='the place for data')
    parser.add_argument('--model_root', default='model', help='the place to save model')
    parser.add_argument('--log_root', default='log', help='the place to save logs and figs')
    parser.add_argument('--use_gpu', default='')

    parser.add_argument('--seed', default=-1, type=int, help='seed for choosing labeled data')
    parser.add_argument('--train_batch_size', default=100, type=int)
    parser.add_argument('--dev_batch_size', default=100, type=int)
    parser.add_argument('--eval_period', default=450, type=int, 
        help='how many iterations to evaluate model each time')
    parser.add_argument('--visual_period', default=5000, type=int, help='how many iterations to \
        visualize generated images')

    parser.add_argument('--classifier_lr', default=3e-3, type=float)
    parser.add_argument('--gen_lr', default=3e-3, type=float)
    parser.add_argument('--dis_lr', default=3e-3, type=float)
    parser.add_argument('--min_lr', default=0.0, type=float)
    parser.add_argument('--max_epochs', default=1000, type=int)
    parser.add_argument('--lambda_g', default=10, type=float, help='parameter for gradient penalty')
    parser.add_argument('--lambda_p', default=0.0, type=float, help='parameter for pull-away term')
    parser.add_argument('--lambda_e', default=0.0, type=float, 
        help='parameter for entropy term of unlabeled data')
    parser.add_argument('--lambda_gan', default=0, type=float)
    parser.add_argument('--early_stop', default=8, type=int)

    parser.add_argument('--iter_c', default=1, type=int, help='iteration to train discriminator \
        per each iteration to train generator')
    parser.add_argument('--alpha', default=0.8, type=float, help='parameters to exclude the true data')
    parser.add_argument('--p_d_bar_type', default='uniform', help='type of p_d_bar')
    parser.add_argument('--beta_1', default=0.0, type=float, 
        help='parameter to control the shape of p_d_bar (e.g. the std of the noise added in p_d_bar)')
    parser.add_argument('--beta_2', default=1.0, type=float, 
        help='parameter to control the shape of p_d_bar (e.g. the std of the noise added in p_d_bar)')
    parser.add_argument('--LAMBDA', default=1e-6, type=float)
    parser.add_argument('--lambda_feat', default=1.0, type=float)
    parser.add_argument('--lambda_out', default=0.3, type=float)
    parser.add_argument('--threshold', default=1.5, type=float)

    parser.add_argument('--feature_extractor', default='ae')

    parser.add_argument('--save_gan', action='store_true')
    parser.add_argument('--save_classifier', action='store_true')
    parser.add_argument('--save_vae', action='store_true')

    parser.add_argument('--gan_checkpoint', default='')
    parser.add_argument('--classifier_checkpoint', default='')
    parser.add_argument('--vae_checkpoint', default='')

    #### parameters for models
    parser.add_argument('--image_size', default=32, type=int)
    parser.add_argument('--n_channels', default=3, type=int)
    parser.add_argument('--noise_size', default=100, type=int)
    parser.add_argument('--num_label', default=10, type=int)
    parser.add_argument('--train_class', default=0, type=int)
    parser.add_argument('--feature_size', default=128, type=int)

    parser.add_argument('--spectral_norm', action='store_true')
    parser.add_argument('--dynamic_alpha', action='store_true')
    parser.add_argument('--linear', action='store_true')
    ####

    parser.add_argument('--task_id', default='baseline', type=str)
    parser.add_argument('--affix', default='')

    #### argument only for toy dataset 
    parser.add_argument('--size_valid_data', default=5000, type=int)
    parser.add_argument('--size_test_data', default=1000, type=int)
    ####
    return parser.parse_args()

def print_args(args, logger=None):
    if logger == None:
        for k, v in vars(args).items():
            print('{:<16} : {}'.format(k, v))
    else:
        for k, v in vars(args).items():
            logger.info('{:<16} : {}'.format(k, v))  

def main(args):

    ###########################################################
    ## create folders to save models and logs
    save_folder = '%s_%s' % (args.dataset, args.task_id)
    name = 'a%.2f_%s_b1_%.2f_b2_%.2f_e%.2f_p%.2f_seed%d' % (args.alpha, args.p_d_bar_type, args.beta_1, 
        args.beta_2, args.lambda_e, args.lambda_p, args.seed)
    if args.spectral_norm:
        name += '_spectral'

    name += args.affix

    log_folder = os.path.join(args.log_root, save_folder, '%d' % args.train_class); makedirs(log_folder)

    model_folder = os.path.join(args.model_root, save_folder, '%d' % args.train_class); makedirs(model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    setattr(args, 'save_folder', save_folder)
    setattr(args, 'name', name)

    logger = create_logger(log_folder, args.todo)

    print_args(args, logger)

    ###########################################################
    # preparing dataset
    ###########################################################

    if args.dataset == 'mnist':
        setattr(args, '2d', False)
        train_loader, p_d, p_d_bar, dev_loader, p_d_2 = \
            all_data.get_mnist_loaders(args)

        # from mnist_trainer import Trainer

    elif args.dataset == 'svhn':
        setattr(args, '2d', False)

        train_loader, p_d, p_d_bar, dev_loader, p_d_2 = \
            all_data.get_svhn_loaders(args)

        # from svhn_trainer import Trainer

    elif args.dataset == 'cifar':
        setattr(args, '2d', False)
        train_loader, p_d, p_d_bar, dev_loader, p_d_2 = \
            all_data.get_cifar_loaders(args)

        # from cifar_trainer import Trainer

    else:
        raise NotImplementedError    


    if args.todo in ['train', 'ae', 'vae', 'gan', 'classifier', 'finetune']:

        from trainer_feat import Trainer

        # parameters for trainer
        trainer_dict = {'args': args, 'logger': logger}

        tr_data_dict = {'train_loader': train_loader,
            'p_d': p_d,
            'p_d_bar': p_d_bar,
            'dev_loader': dev_loader,
            'p_d_2': p_d_2}

        trainer = Trainer(trainer_dict)

        if args.todo == 'train':
            trainer.train(tr_data_dict)
        elif args.todo == 'vae':
            trainer.train_vae(tr_data_dict)
        elif args.todo == 'ae':
            trainer.train_ae(tr_data_dict)
        elif args.todo == 'gan':
            trainer.train_gan(tr_data_dict)
        elif args.todo == 'classifier':
            trainer.train_classifier(tr_data_dict)
        elif args.todo == 'finetune':
            trainer.finetune(tr_data_dict)

    elif args.todo == 'test':
        from test import test
        test(args, args.vae_checkpoint, args.classifier_checkpoint, dev_loader)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    args = argparser()
    if args.use_gpu != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.use_gpu
    main(args)
    