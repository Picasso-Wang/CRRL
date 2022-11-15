import os
import argparse


def parse_option_evaluation(args_pretrain):
    parser = argparse.ArgumentParser('argument for evaluation')
    parser.add_argument('--gpu', type=int, default=4, help='GPU id to use.')   # args_pretrain.gpu
    parser.add_argument('--epochs', type=int, default=85, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--dataset', type=str, default=args_pretrain.dataset, choices=['ntu', 'ntu120', 'pku1', 'pku2', 'cmu', 'ucla'])
    parser.add_argument('--ntu_dataset', type=str, default=args_pretrain.ntu_dataset, choices=['xsub', 'xview', 'xsetup'])
    parser.add_argument('--pku_dataset', type=str, default=args_pretrain.pku_dataset, choices=['xsub', 'xview'])
    parser.add_argument('--fold', type=str, default=args_pretrain.fold, choices=['fold1', 'fold2', 'fold3', 'fold4'])
    parser.add_argument('--mode', type=str, default='ntu60_transfer_to_pku2', choices=['lin_eval', 'knn', 'fine_tune', 'pku1_transfer_to_pku2', 'ntu60_transfer_to_pku2'])
    parser.add_argument('--if_pretrain', type=bool, default=True, help='if use pretrained model or not')
    parser.add_argument('--if_resume', type=bool, default=False, help='if resume training or not')
    parser.add_argument('--resume_load_path', type=str, default='')
    parser.add_argument('--semi_percent', type=float, default=0.1)
    parser.add_argument('--semi_num', type=int, default=0)

    # model definition
    parser.add_argument('--model', type=str, default=args_pretrain.model, choices=['Bi_GRU'])   # ['lstm', 'Bi_lstm', 'GRU', 'Bi_GRU']
    parser.add_argument('--input_size', type=int, default=args_pretrain.input_size)
    parser.add_argument('--hidden_units', type=int, default=args_pretrain.hidden_units)
    parser.add_argument('--layers', type=int, default=args_pretrain.layers)
    parser.add_argument('--dropout', type=float, default=args_pretrain.dropout)

    # optimizer setting
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--learning_rate_encoder', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--nesterov', type=bool, default=True)
    parser.add_argument('--lr_decay_epochs', type=str, default=[15, 35, 50, 66, 75], help='where to decay lr')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='decay rate for learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--syncBN', action='store_true', help='enable synchronized BN')
    parser.add_argument('--decay_type', type=str, default='None', choices=['cosine', 'None'])

    opt = parser.parse_args()

    opt.reconstruct = args_pretrain.reconstruct
    opt.contrastive = args_pretrain.contrastive

    print('==> gpu:', opt.gpu)
    print('==> mode:', opt.mode)
    print('==> learning_rate:', opt.learning_rate)
    print('==> learning_rate_encoder:', opt.learning_rate_encoder)
    print('==> pretrain model name:', args_pretrain.model_name)
    print('==> if_pretrain:', opt.if_pretrain)

    # set validate dataset/label path
    if opt.dataset == 'ntu':
        if opt.ntu_dataset == 'xsub':
            opt.train_data_path = args_pretrain.train_data_path
            opt.train_label_path = r'/root/wp/dataset/ntu/ske/raw_data/xsub/train_label.npy'
            opt.train_num_frame_path = args_pretrain.train_num_frame_path
            opt.val_data_path = r'/root/wp/dataset/ntu/ske/normalized_data/xsub/val_data.npy'
            opt.val_label_path = r'/root/wp/dataset/ntu/ske/raw_data/xsub/val_label.npy'
            opt.val_num_frame_path = r'/root/wp/dataset/ntu/ske/normalized_data/xsub/val_num_frame.npy'

        if opt.ntu_dataset == 'xview':
            opt.train_data_path = args_pretrain.train_data_path
            opt.train_label_path = r'/root/wp/dataset/ntu/ske/raw_data/xview/train_label.npy'
            opt.train_num_frame_path = args_pretrain.train_num_frame_path
            opt.val_data_path = r'/root/wp/dataset/ntu/ske/normalized_data/xview/val_data.npy'
            opt.val_label_path = r'/root/wp/dataset/ntu/ske/raw_data/xview/val_label.npy'
            opt.val_num_frame_path = r'/root/wp/dataset/ntu/ske/normalized_data/xview/val_num_frame.npy'

    elif opt.dataset == 'ntu120':
        if opt.ntu_dataset == 'xsub':
            opt.train_data_path = args_pretrain.train_data_path
            opt.train_label_path = r'/root/wp/dataset/ntu120/ske/normalized_data/xsub/train_label.npy'
            opt.train_num_frame_path = args_pretrain.train_num_frame_path
            opt.val_data_path = r'/root/wp/dataset/ntu120/ske/normalized_data/xsub/val_data.npy'
            opt.val_label_path = r'/root/wp/dataset/ntu120/ske/normalized_data/xsub/val_label.npy'
            opt.val_num_frame_path = r'/root/wp/dataset/ntu120/ske/normalized_data/xsub/val_num_frame.npy'

        elif opt.ntu_dataset == 'xsetup':
            opt.train_data_path = args_pretrain.train_data_path
            opt.train_label_path = r'/root/wp/dataset/ntu120/ske/normalized_data/xsetup/train_label.npy'
            opt.train_num_frame_path = args_pretrain.train_num_frame_path
            opt.val_data_path = r'/root/wp/dataset/ntu120/ske/normalized_data/xsetup/val_data.npy'
            opt.val_label_path = r'/root/wp/dataset/ntu120/ske/raw_data/xsetup/val_label.npy'
            opt.val_num_frame_path = r'/root/wp/dataset/ntu120/ske/normalized_data/xsetup/val_num_frame.npy'


    elif opt.dataset == 'pku1':
        if opt.pku_dataset == 'xsub':
            opt.train_data_path = args_pretrain.train_data_path
            opt.train_label_path = r'/root/wp/dataset/pku_mmd_part1/ske/normalized/xsub/train_label.npy'
            opt.train_num_frame_path = args_pretrain.train_num_frame_path
            opt.val_data_path = r'/root/wp/dataset/pku_mmd_part1/ske/normalized/xsub/val_data.npy'
            opt.val_label_path = r'/root/wp/dataset/pku_mmd_part1/ske/normalized/xsub/val_label.npy'
            opt.val_num_frame_path = r'/root/wp/dataset/pku_mmd_part1/ske/normalized/xsub/val_num_frame.npy'

        if opt.pku_dataset == 'xview':
            opt.train_data_path = args_pretrain.train_data_path
            opt.train_label_path = r'/root/wp/dataset/pku_mmd_part1/ske/normalized/xview/train_label.npy'
            opt.train_num_frame_path = args_pretrain.train_num_frame_path
            opt.val_data_path = r'/root/wp/dataset/pku_mmd_part1/ske/normalized/xview/val_data.npy'
            opt.val_label_path = r'/root/wp/dataset/pku_mmd_part1/ske/normalized/xview/val_label.npy'
            opt.val_num_frame_path = r'/root/wp/dataset/pku_mmd_part1/ske/normalized/xview/val_num_frame.npy'


    elif opt.dataset == 'pku2':
        if opt.pku_dataset == 'xsub':
            opt.train_data_path = args_pretrain.train_data_path
            opt.train_label_path = r'/root/wp/dataset/pku_mmd_part2/ske/normalized/xsub/train_label.npy'
            opt.train_num_frame_path = args_pretrain.train_num_frame_path
            opt.val_data_path = r'/root/wp/dataset/pku_mmd_part2/ske/normalized/xsub/val_data.npy'
            opt.val_label_path = r'/root/wp/dataset/pku_mmd_part2/ske/normalized/xsub/val_label.npy'
            opt.val_num_frame_path = r'/root/wp/dataset/pku_mmd_part2/ske/normalized/xsub/val_num_frame.npy'

        if opt.pku_dataset == 'xview':
            opt.train_data_path = args_pretrain.train_data_path
            opt.train_label_path = r'/root/wp/dataset/pku_mmd_part2/ske/normalized/xview/train_label.npy'
            opt.train_num_frame_path = args_pretrain.train_num_frame_path
            opt.val_data_path = r'/root/wp/dataset/pku_mmd_part2/ske/normalized/xview/val_data.npy'
            opt.val_label_path = r'/root/wp/dataset/pku_mmd_part2/ske/normalized/xview/val_label.npy'
            opt.val_num_frame_path = r'/root/wp/dataset/pku_mmd_part2/ske/normalized/xview/val_num_frame.npy'

    elif opt.dataset == 'cmu':
        opt.train_data_path = args_pretrain.train_data_path
        opt.train_num_frame_path = args_pretrain.train_num_frame_path
        opt.train_label_path = r'/root/wp/dataset/cmu/fullset/downSampled100/' + opt.fold + r'/train_label.npy'
        opt.val_data_path = r'/root/wp/dataset/cmu/fullset/downSampled100/' + opt.fold + r'/val_data.npy'
        opt.val_label_path = r'/root/wp/dataset/cmu/fullset/downSampled100/' + opt.fold + r'/val_label.npy'
        opt.val_num_frame_path = r'/root/wp/dataset/cmu/fullset/downSampled100/' + opt.fold + r'/val_num_frame.npy'

    elif opt.dataset == 'ucla':
        opt.train_data_path = args_pretrain.train_data_path
        opt.train_label_path = r'/root/wp/dataset/N_UCLA/preprocessed/train_label.npy'
        opt.train_num_frame_path = args_pretrain.train_num_frame_path
        opt.val_data_path = r'/root/wp/dataset/N_UCLA/preprocessed/val_data.npy'
        opt.val_label_path = r'/root/wp/dataset/N_UCLA/preprocessed/val_label.npy'
        opt.val_num_frame_path = r'/root/wp/dataset/N_UCLA/preprocessed/val_num_frame.npy'

    print('==> val_data_path:', opt.val_data_path)

    opt.model_name = 'lr{}_lrEn{}_bsz{}_epoch{}'.format(opt.learning_rate, opt.learning_rate_encoder, opt.batch_size, opt.epochs)

    print('==> model_name: {}\n'.format(opt.model_name))
    if opt.mode == 'lin_eval':
        if opt.if_pretrain:
            opt.save_model_tb_path = os.path.join(args_pretrain.model_folder, 'lin_eval')
        else:
            opt.save_model_tb_path = os.path.join(args_pretrain.model_folder, 'lin_eval_no_pretrain')

    elif opt.mode == 'fine_tune':
        opt.save_model_tb_path = os.path.join(args_pretrain.model_folder, 'fine_tune')

    elif opt.mode == 'knn':
        opt.save_model_tb_path = os.path.join(args_pretrain.model_folder, 'knn')

    elif opt.mode == 'pku1_transfer_to_pku2':
        opt.save_model_tb_path = os.path.join(args_pretrain.model_folder, 'pku1_transfer_to_pku2')

    elif opt.mode == 'ntu60_transfer_to_pku2':
        opt.save_model_tb_path = os.path.join(args_pretrain.model_folder, 'ntu60_transfer_to_pku2')

    opt.model_folder = os.path.join(opt.save_model_tb_path, opt.model_name)
    opt.tb_folder = os.path.join(opt.model_folder, 'tensorboards')
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.n_classes = args_pretrain.n_classes   # number of classes

    # when perform across dataset learning (transfer learning), some arguments need to change as follows:
    if opt.mode == 'pku1_transfer_to_pku2' or opt.mode == 'ntu60_transfer_to_pku2':
        opt.dataset = 'pku2'
        opt.train_data_path = r'/root/wp/dataset/pku_mmd_part2/ske/normalized/xsub/train_data.npy'
        opt.train_label_path = r'/root/wp/dataset/pku_mmd_part2/ske/normalized/xsub/train_label.npy'
        opt.train_num_frame_path = r'/root/wp/dataset/pku_mmd_part2/ske/normalized/xsub/train_num_frame.npy'
        opt.val_data_path = r'/root/wp/dataset/pku_mmd_part2/ske/normalized/xsub/val_data.npy'
        opt.val_label_path = r'/root/wp/dataset/pku_mmd_part2/ske/normalized/xsub/val_label.npy'
        opt.val_num_frame_path = r'/root/wp/dataset/pku_mmd_part2/ske/normalized/xsub/val_num_frame.npy'
        opt.n_classes = 41   # number of classes

    return opt
