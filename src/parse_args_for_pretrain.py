import os
import argparse


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--gpu', default=6, type=int, help='GPU id to use.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--epochs', type=int, default=60, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=list, default=[18, 50], help='where to decay lr')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--dataset', type=str, default='ntu', choices=['ntu', 'ntu120', 'pku1', 'pku2', 'cmu', 'uwa3d', 'ucla', 'sbu'])
    parser.add_argument('--ntu_dataset', type=str, default='xsub', choices=['xsub', 'xview', 'xsetup'])
    parser.add_argument('--pku_dataset', type=str, default='xsub', choices=['xsub', 'xview'])
    parser.add_argument('--fold', type=str, default='fold3', choices=['fold1', 'fold2', 'fold3', 'fold4'], help='set fold for cmu dataset')

    parser.add_argument('--if_resume', type=bool, default=False, help='if resume training or not')
    parser.add_argument('--resume_load_path', type=str, default='')
    parser.add_argument('--if_mlp', type=bool, default=True, help='if use MLP on top of encoder?')
    parser.add_argument('--if_update_key_encoder', type=bool, default=True, help='if update the key encoder?')
    parser.add_argument('--if_update_key_mlp', type=bool, default=False, help='if update the MLP on top of key encoder?')
    parser.add_argument('--if_recon_forwardly', type=bool, default=True, help='if reconstructing the sequence forwardly?')
    parser.add_argument('--if_recon_reversely', type=bool, default=True, help='if reconstructing the sequence reversely?')

    # model setting
    parser.add_argument('--model', type=str, default='Bi_GRU', choices=['Bi_GRU'])   # ['lstm', 'Bi_GRU', 'Bi_lstm', 'GRU']
    parser.add_argument('--hidden_units', type=int, default=256)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--threshold', type=float, default=0.08, help='threshold used in the MSE loss')

    # optimizer setting
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--nesterov', type=bool, default=False)
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='decay rate for learning rate')

    # task setting
    parser.add_argument('--reconstruct', type=bool, default=False, help='whether use reconstructive learning or not?')
    parser.add_argument('--contrastive', type=bool, default=False, help='whether use contrastive learning or not?')
    parser.add_argument('--teacher_student', type=bool, default=True, help='the encoder trained by contrastive learning is used as a teacher and knowledge is transferred to reconstruction encoder in distillation manner.')

    opt = parser.parse_args()

    print('==> gpu:', opt.gpu)
    print('==> epochs:', opt.epochs)
    print('==> reconstruct:', opt.reconstruct)
    print('==> constrastive:', opt.contrastive)
    print('==> teacher_student:', opt.teacher_student)

    if opt.teacher_student:
        opt.checkpoint_path = ' '   # the path of the checkpoint of the encoder trained by contrastive learning. 

    if opt.contrastive:
        opt.temperature = 0.1
        opt.contra_momentum = 0.999
        opt.K = 64

    if opt.teacher_student:
        opt.lambda_ts = 0.1

    # setup arguments for each dataset
    if opt.dataset == 'ntu':
        if opt.ntu_dataset == 'xsub':
            opt.train_data_path = r'/root/wp/dataset/ntu/ske/normalized_data/xsub/train_data.npy'
            opt.train_data_reverse_path = r'/root/wp/dataset/ntu/ske/normalized_data/xsub/train_data_reverse.npy'
            opt.train_num_frame_path = r'/root/wp/dataset/ntu/ske/normalized_data/xsub/train_num_frame.npy'
            opt.train_speed_path = r'/root/wp/dataset/ntu/ske/normalized_data/xsub/train_speed.npy'
            opt.train_speed_reverse_path = r'/root/wp/dataset/ntu/ske/normalized_data/xsub/train_speed_reverse.npy'

        elif opt.ntu_dataset == 'xview':
            opt.train_data_path = r'/root/wp/dataset/ntu/ske/normalized_data/xview/train_data.npy'
            opt.train_data_reverse_path = r'/root/wp/dataset/ntu/ske/normalized_data/xview/train_data_reverse.npy'
            opt.train_num_frame_path = r'/root/wp/dataset/ntu/ske/normalized_data/xview/train_num_frame.npy'
            opt.train_speed_path = r'/root/wp/dataset/ntu/ske/normalized_data/xview/train_speed.npy'
            opt.train_speed_reverse_path = r'/root/wp/dataset/ntu/ske/normalized_data/xview/train_speed_reverse.npy'

    elif opt.dataset == 'ntu120':
        if opt.ntu_dataset == 'xsub':
            opt.train_data_path = r'/root/wp/dataset/ntu120/ske/normalized_data/xsub/train_data.npy'
            opt.train_data_reverse_path = r'/root/wp/dataset/ntu120/ske/normalized_data/xsub/train_data_reverse.npy'
            opt.train_num_frame_path = r'/root/wp/dataset/ntu120/ske/normalized_data/xsub/train_num_frame.npy'
            opt.train_speed_path = r'/root/wp/dataset/ntu120/ske/normalized_data/xsub/train_speed.npy'
            opt.train_speed_reverse_path = r'/root/wp/dataset/ntu120/ske/normalized_data/xsub/train_speed_reverse.npy'

        elif opt.ntu_dataset == 'xsetup':
            opt.train_data_path = r'/root/wp/dataset/ntu120/ske/normalized_data/xsetup/train_data.npy'
            opt.train_data_reverse_path = r'/root/wp/dataset/ntu120/ske/normalized_data/xsetup/train_data_reverse.npy'
            opt.train_num_frame_path = r'/root/wp/dataset/ntu120/ske/normalized_data/xsetup/train_num_frame.npy'
            opt.train_speed_path = r'/root/wp/dataset/ntu120/ske/normalized_data/xsetup/train_speed.npy'
            opt.train_speed_reverse_path = r'/root/wp/dataset/ntu120/ske/normalized_data/xsetup/train_speed_reverse.npy'

    elif opt.dataset == 'pku1':
        if opt.pku_dataset == 'xsub':
            opt.train_data_path = r'/root/wp/dataset/pku_mmd_part1/ske/normalized/xsub/train_data.npy'
            opt.train_data_reverse_path = r'/root/wp/dataset/pku_mmd_part1/ske/normalized/xsub/train_data_reverse.npy'
            opt.train_num_frame_path = r'/root/wp/dataset/pku_mmd_part1/ske/normalized/xsub/train_num_frame.npy'
            opt.train_speed_path = r'/root/wp/dataset/pku_mmd_part1/ske/normalized/xsub/train_speed.npy'
            opt.train_speed_reverse_path = r'/root/wp/dataset/pku_mmd_part1/ske/normalized/xsub/train_speed_reverse.npy'

        elif opt.pku_dataset == 'xview':
            opt.train_data_path = r'/root/wp/dataset/pku_mmd_part1/ske/normalized/xview/train_data.npy'
            opt.train_data_reverse_path = r'/root/wp/dataset/pku_mmd_part1/ske/normalized/xview/train_data_reverse.npy'
            opt.train_num_frame_path = r'/root/wp/dataset/pku_mmd_part1/ske/normalized/xview/train_num_frame.npy'
            opt.train_speed_path = r'/root/wp/dataset/pku_mmd_part1/ske/normalized/xview/train_speed.npy'
            opt.train_speed_reverse_path = r'/root/wp/dataset/pku_mmd_part1/ske/normalized/xview/train_speed_reverse.npy'

    elif opt.dataset == 'pku2':
        if opt.pku_dataset == 'xsub':
            opt.train_data_path = r'/root/wp/dataset/pku_mmd_part2/ske/normalized/xsub/train_data.npy'
            opt.train_data_reverse_path = r'/root/wp/dataset/pku_mmd_part2/ske/normalized/xsub/train_data_reverse.npy'
            opt.train_num_frame_path = r'/root/wp/dataset/pku_mmd_part2/ske/normalized/xsub/train_num_frame.npy'
            opt.train_speed_path = r'/root/wp/dataset/pku_mmd_part2/ske/normalized/xsub/train_speed.npy'
            opt.train_speed_reverse_path = r'/root/wp/dataset/pku_mmd_part2/ske/normalized/xsub/train_speed_reverse.npy'

        elif opt.pku_dataset == 'xview':
            opt.train_data_path = r'/root/wp/dataset/pku_mmd_part2/ske/normalized/xview/train_data.npy'
            opt.train_data_reverse_path = r'/root/wp/dataset/pku_mmd_part2/ske/normalized/xview/train_data_reverse.npy'
            opt.train_num_frame_path = r'/root/wp/dataset/pku_mmd_part2/ske/normalized/xview/train_num_frame.npy'
            opt.train_speed_path = r'/root/wp/dataset/pku_mmd_part2/ske/normalized/xview/train_speed.npy'
            opt.train_speed_reverse_path = r'/root/wp/dataset/pku_mmd_part2/ske/normalized/xview/train_speed_reverse.npy'

    elif opt.dataset == 'cmu':
        opt.train_data_path = r'/root/wp/dataset/cmu/fullset/downSampled100/' + opt.fold + r'/train_data.npy'
        opt.train_data_reverse_path = r'/root/wp/dataset/cmu/fullset/downSampled100/' + opt.fold + r'/train_data_reverse.npy'
        opt.train_num_frame_path = r'/root/wp/dataset/cmu/fullset/downSampled100/' + opt.fold + r'/train_num_frame.npy'
        opt.train_speed_path = r'/root/wp/dataset/cmu/fullset/downSampled100/' + opt.fold + r'/train_speed.npy'
        opt.train_speed_reverse_path = r'/root/wp/dataset/cmu/fullset/downSampled100/' + opt.fold + r'/train_speed_reverse.npy'

    elif opt.dataset == 'ucla':
        opt.train_data_path = r'/root/wp/dataset/N_UCLA/preprocessed/train_data.npy'
        opt.train_data_reverse_path = r'/root/wp/dataset/N_UCLA/preprocessed/train_data_reverse.npy'
        opt.train_num_frame_path = r'/root/wp/dataset/N_UCLA/preprocessed/train_num_frame.npy'
        opt.train_speed_path = r'/root/wp/dataset/N_UCLA/preprocessed/train_speed.npy'
        opt.train_speed_reverse_path = r'/root/wp/dataset/N_UCLA/preprocessed/train_speed_reverse.npy'

    print('==> train_data_path:', opt.train_data_path)

    # set flag
    if opt.dataset == 'ucla':
        flag = 'ucla'
    elif opt.dataset == 'ntu':
        flag = 'ntu_{}'.format(opt.ntu_dataset)
    elif opt.dataset == 'ntu120':
        flag = 'ntu120_{}'.format(opt.ntu_dataset)
    elif opt.dataset == 'cmu':
        flag = 'cmu_{}'.format(opt.fold)
    elif opt.dataset == 'pku1':
        flag = 'pku1_{}'.format(opt.pku_dataset)
    elif opt.dataset == 'pku2':
        flag = 'pku2_{}'.format(opt.pku_dataset)
    
    if opt.reconstruct:
        opt.model_name = '{}_reconstruct_hidden{}_layer{}_lr{}_bsz{}_epoch{}'.format(
            flag,
            opt.hidden_units,
            opt.layers,
            opt.learning_rate,
            opt.batch_size,
            opt.epochs,
            )

    if opt.contrastive:
        opt.model_name = '{}_contrastive_hidden{}_layer{}_bsz{}_lr{}_epoch{}_moment{}_k{}_temp{}'.format(
            flag,
            opt.hidden_units,
            opt.layers,
            opt.batch_size,
            opt.learning_rate,
            opt.epochs,
            opt.contra_momentum,
            opt.K,
            opt.temperature,
        )

    if opt.teacher_student:
        opt.model_name = '{}_teach_stud_hidden{}_layer{}_lr{}_bsz{}_epoch{}_lambda_ts{}'.format(
            flag,
            opt.hidden_units,
            opt.layers,
            opt.learning_rate,
            opt.batch_size,
            opt.epochs,
            opt.lambda_ts,
            )

    print('==> model_name: {}'.format(opt.model_name))
    opt.model_path = '/root/wp/action/results/20220607/{}_models'.format(opt.dataset)
    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.model_folder, 'tensorboard')
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    number_of_classes = {'ntu': 60, 'ntu120': 120, 'ucla': 10, 'cmu': 45, 'pku1': 51, 'pku2': 41}
    opt.n_classes = number_of_classes[opt.dataset]

    input_size = {'ntu': 150, 'ntu120': 150, 'ucla': 60, 'cmu': 93, 'pku1': 150, 'pku2': 150}
    opt.input_size = input_size[opt.dataset]

    return opt
