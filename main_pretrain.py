import os
import torch
import time
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
from tools.adjustLearningRate import adjust_learning_rate
from sklearn.cluster import KMeans
from tools.loss_function import variable_length_mse
from parse_args_for_pretrain import parse_option
from feeder import SelfDefineDataset_pretrain_speed, collate_fn_speed
from models import GRU_model, Bi_GRU, Nonlinear_2layer, Bi_GRU_packpad
import moco_builder
from thop import profile


def main_contrastive(args):
    train_dataset = SelfDefineDataset_pretrain_speed(args.train_data_path, args.train_data_reverse_path,
                                                     args.train_num_frame_path, args.train_speed_path,
                                                     args.train_speed_reverse_path, args.dataset)
    print('==> Number of training video: {}'.format(len(train_dataset)))
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn_speed, drop_last=True)
    print("==> creating MoCo model")
    moco_model = moco_builder.MoCo(Bi_GRU_packpad, args.input_size, args.hidden_units, args.layers,
                                            args.dropout, args.gpu, args.K, args.contra_momentum, args.temperature,
                                            args.if_mlp, args.if_update_key_encoder, args.if_update_key_mlp)

    moco_model = moco_model.cuda(args.gpu)
    optimizer_moco = torch.optim.SGD(moco_model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                     nesterov=args.nesterov, weight_decay=args.weight_decay)
    ctrs_loss_func = nn.CrossEntropyLoss()
    args.start_epoch = 0
    # tensorboard
    writer = SummaryWriter(log_dir=args.tb_folder, flush_secs=2)  # instantiate SummaryWriter class

    print('==> training')
    moco_model.train()
    since = time.time()
    loss_collum = 1
    loss_epochs = np.zeros(((args.epochs-args.start_epoch), loss_collum), dtype=float)  # save loss of all epochs

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(epoch, args, optimizer_moco)
        loss_batchs = np.zeros((len(train_dataset) // args.batch_size + 1, loss_collum), dtype=float)  # save loss of all batches in one epoch

        for batch_id, (num_frame, X, Y, S, SR) in enumerate(train_loader):
            X, Y, S, SR = X.cuda(args.gpu), Y.cuda(args.gpu), S.cuda(args.gpu), SR.cuda(args.gpu)
            _, output, target = moco_model(X, S, num_frame)
            contrast_loss = ctrs_loss_func(output, target)
            optimizer_moco.zero_grad()
            contrast_loss.backward()
            optimizer_moco.step()
            loss_batchs[batch_id, 0] = contrast_loss.item()

        loss_mean = np.mean(loss_batchs, 0)
        loss_epochs[epoch-args.start_epoch, :] = loss_mean
        print('epoch:{:3d},   contra_loss:{:7.3f}'.format(epoch, loss_mean[0]))
        writer.add_scalar('learning_rate', optimizer_moco.param_groups[0]['lr'], epoch-args.start_epoch)
        writer.add_scalar('contrastive_loss', loss_mean[0], epoch-args.start_epoch)

    time_elapsed = time.time() - since
    print('==> pretraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('==> saving model')
    encoder = copy.deepcopy(moco_model.encoder_fc_q.gru)
    encoder_dict = encoder.state_dict()
    for k in list(encoder_dict.keys()):
        encoder_dict['gru.'+k] = encoder_dict[k]
        del encoder_dict[k]

    state = {'opt': args,
             'epoch': epoch,
             'encoder': encoder_dict,
             'moco_model': moco_model.state_dict(),
             'optim_moco_model': optimizer_moco.state_dict()}
    save_path_state = os.path.join(args.model_folder, 'epoch{}state.pth'.format(epoch))
    torch.save(state, save_path_state)

    return state


def main_reconstruct(args):
    train_dataset = SelfDefineDataset_pretrain_speed(args.train_data_path, args.train_data_reverse_path,
                                                     args.train_num_frame_path, args.train_speed_path,
                                                     args.train_speed_reverse_path, args.dataset)
    print('==> Number of training video: {}'.format(len(train_dataset)))
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn_speed, drop_last=True)
    encoder = Bi_GRU(args.input_size, args.hidden_units, args.layers, args.dropout)
    
    # # count FLOPs and params
    # input_ = torch.randn(1, 60, 150)
    # macs, params = profile(encoder, inputs=(input_, ))
    # print(' FLOPs: ', macs*2)
    # print('params: ', params)

    encoder_parameters = sum(param.numel() for param in encoder.parameters())
    print('==> encoder parameters: {}. '.format(encoder_parameters))
    encoder = encoder.cuda(args.gpu)
    decoder = GRU_model(2 * args.hidden_units, args.input_size, args.layers, args.dropout)
    decoder_parameters = sum(param.numel() for param in decoder.parameters())
    print('==> decoder parameters: {}'.format(decoder_parameters))
    decoder = decoder.cuda(args.gpu)
    optimizer = torch.optim.SGD([{'params': encoder.parameters()}, {'params': decoder.parameters()}],
                                lr=args.learning_rate, momentum=args.momentum, nesterov=args.nesterov,
                                weight_decay=args.weight_decay)
    args.start_epoch = 0
    # tensorboard
    writer = SummaryWriter(log_dir=args.tb_folder, flush_secs=2)  # instantiate SummaryWriter class

    print('==> training')
    encoder.train()
    decoder.train()
    since = time.time()
    loss_collum = 2
    loss_epochs = np.zeros(((args.epochs-args.start_epoch), loss_collum), dtype=float)  # save loss of all epochs

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(epoch, args, optimizer)
        loss_batchs = np.zeros((len(train_dataset) // args.batch_size + 1, loss_collum), dtype=float)  # save loss of all batches in one epoch

        for batch_id, (num_frame, X, Y, S, SR) in enumerate(train_loader):
            X, Y, S, SR = X.cuda(args.gpu), Y.cuda(args.gpu), S.cuda(args.gpu), SR.cuda(args.gpu)
            indices_0 = list(range(len(num_frame)))
            indices_1 = [i - 1 for i in num_frame]
            max_len = max(num_frame)
            x_pack = rnn_utils.pack_padded_sequence(X, num_frame, batch_first=True)
            feature = encoder(x_pack)
            feature_pad, _ = rnn_utils.pad_packed_sequence(feature, batch_first=True)
            fisrtStep_hiddenStates = feature_pad[:, 0, :]
            lastStep_hiddenStates = feature_pad[indices_0, indices_1, :]
            all_loss = 0.0

            if args.if_recon_forwardly:
                tmp1 = fisrtStep_hiddenStates.repeat(max_len, 1, 1)
                tmp1 = tmp1.permute([1, 0, 2])
                tmp_pack1 = rnn_utils.pack_padded_sequence(tmp1, num_frame, batch_first=True)
                reconstruction1 = decoder(tmp_pack1)
                reconstruction_pad1, _ = rnn_utils.pad_packed_sequence(reconstruction1, batch_first=True)
                X = X[:, :max_len, :]
                reconstructLoss1 = variable_length_mse(reconstruction_pad1, X, num_frame, args.gpu)
                all_loss = all_loss + reconstructLoss1
                loss_batchs[batch_id, 0] = reconstructLoss1.item()

            if args.if_recon_reversely:
                tmp = lastStep_hiddenStates.repeat(max_len, 1, 1)
                tmp = tmp.permute([1, 0, 2])
                tmp_pack = rnn_utils.pack_padded_sequence(tmp, num_frame, batch_first=True)
                reconstruction = decoder(tmp_pack)
                reconstruction_pad, _ = rnn_utils.pad_packed_sequence(reconstruction, batch_first=True)
                Y = Y[:, :max_len, :]
                reconstructLoss = variable_length_mse(reconstruction_pad, Y, num_frame, args.gpu)  # loss is the mean error of each coordinates.
                all_loss = all_loss + reconstructLoss
                loss_batchs[batch_id, 1] = reconstructLoss.item()

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        loss_mean = np.mean(loss_batchs, 0)
        loss_epochs[epoch-args.start_epoch, :] = loss_mean
        print('epoch:{:3d},   recn_lossF:{:7.3f},   recn_lossL:{:7.3f}'.format(epoch, loss_mean[0], loss_mean[1]))

        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch-args.start_epoch)
        if args.if_recon_forwardly:
            writer.add_scalar('recn_lossF', loss_mean[0], epoch - args.start_epoch)
        if args.if_recon_reversely:
            writer.add_scalar('recn_lossL', loss_mean[1], epoch - args.start_epoch)

    time_elapsed = time.time() - since
    print('==> pretraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('==> saving model')
    state = {'opt': args,
             'epoch': epoch,
             'encoder': encoder.state_dict(),
             'optim_enc': optimizer.state_dict()}
    save_path_state = os.path.join(args.model_folder, 'epoch{}state.pth'.format(epoch))
    torch.save(state, save_path_state)

    return state


def main_teacher_student(args, encoder_pre):
    train_dataset = SelfDefineDataset_pretrain_speed(args.train_data_path, args.train_data_reverse_path,
                                                     args.train_num_frame_path, args.train_speed_path,
                                                     args.train_speed_reverse_path, args.dataset)
    print('==> Number of training video: {}'.format(len(train_dataset)))
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn_speed, drop_last=True)

    encoder = Bi_GRU(args.input_size, args.hidden_units, args.layers, args.dropout)
    encoder_parameters = sum(param.numel() for param in encoder.parameters())
    print('==> encoder parameters: {}. '.format(encoder_parameters))
    encoder = encoder.cuda(args.gpu)

    decoder = GRU_model(2 * args.hidden_units, args.input_size, args.layers, args.dropout)
    decoder_parameters = sum(param.numel() for param in decoder.parameters())
    print('==> decoder parameters: {}'.format(decoder_parameters))
    decoder = decoder.cuda(args.gpu)

    encoder_teacher = Bi_GRU(args.input_size, args.hidden_units, args.layers, args.dropout)
    encoder_teacher.load_state_dict(encoder_pre)
    print('==> loading pretrained encoder to teacher encoder successfully')
    for param in encoder_teacher.parameters():  # freeze encoder_teacher's parameters
        param.requires_grad = False
    encoder_teacher = encoder_teacher.cuda(args.gpu)
    teacher_student_loss_func = nn.MSELoss(reduction='sum')
    mlp = Nonlinear_2layer(2 * args.hidden_units, 2 * args.hidden_units)

    # # count FLOPs and params
    # input_ = torch.randn(1, 512, 512)
    # macs, params = profile(mlp, inputs=(input_, ))
    # print(' FLOPs: ', macs*2)
    # print('params: ', params)

    mlp = mlp.cuda(args.gpu)
    optimizer = torch.optim.SGD([{'params': encoder.parameters()}, {'params': decoder.parameters()},
                                 {'params': mlp.parameters()}], lr=args.learning_rate, momentum=args.momentum,
                                nesterov=args.nesterov, weight_decay=args.weight_decay)
    args.start_epoch = 0
    # tensorboard
    writer = SummaryWriter(log_dir=args.tb_folder, flush_secs=2)  # instantiate SummaryWriter class

    print('==> training')
    encoder.train()
    decoder.train()
    since = time.time()
    loss_collum = 3
    loss_epochs = np.zeros(((args.epochs - args.start_epoch), loss_collum), dtype=float)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(epoch, args, optimizer)
        loss_batchs = np.zeros((len(train_dataset) // args.batch_size + 1, loss_collum), dtype=float)

        for batch_id, (num_frame, X, Y, S, SR) in enumerate(train_loader):
            X, Y, S, SR = X.cuda(args.gpu), Y.cuda(args.gpu), S.cuda(args.gpu), SR.cuda(args.gpu)
            indices_0 = list(range(len(num_frame)))
            indices_1 = [i - 1 for i in num_frame]
            max_len = max(num_frame)
            x_pack = rnn_utils.pack_padded_sequence(X, num_frame, batch_first=True)
            feature = encoder(x_pack)
            feature_pad, _ = rnn_utils.pad_packed_sequence(feature, batch_first=True)
            fisrtStep_hiddenStates = feature_pad[:, 0, :]
            lastStep_hiddenStates = feature_pad[indices_0, indices_1, :]
            all_loss = 0.0

            if args.if_recon_forwardly:
                tmp1 = fisrtStep_hiddenStates.repeat(max_len, 1, 1)
                tmp1 = tmp1.permute([1, 0, 2])
                tmp_pack1 = rnn_utils.pack_padded_sequence(tmp1, num_frame, batch_first=True)
                reconstruction1 = decoder(tmp_pack1)
                reconstruction_pad1, _ = rnn_utils.pad_packed_sequence(reconstruction1, batch_first=True)
                X = X[:, :max_len, :]
                reconstructLoss1 = variable_length_mse(reconstruction_pad1, X, num_frame, args.gpu)
                all_loss = all_loss + reconstructLoss1
                loss_batchs[batch_id, 0] = reconstructLoss1.item()

            if args.if_recon_reversely:
                tmp = lastStep_hiddenStates.repeat(max_len, 1, 1)
                tmp = tmp.permute([1, 0, 2])
                tmp_pack = rnn_utils.pack_padded_sequence(tmp, num_frame, batch_first=True)
                reconstruction = decoder(tmp_pack)
                reconstruction_pad, _ = rnn_utils.pad_packed_sequence(reconstruction, batch_first=True)
                Y = Y[:, :max_len, :]
                reconstructLoss = variable_length_mse(reconstruction_pad, Y, num_frame, args.gpu)
                all_loss = all_loss + reconstructLoss
                loss_batchs[batch_id, 1] = reconstructLoss.item()

            feature_teacher = encoder_teacher(x_pack)
            feature_pad_teacher, _ = rnn_utils.pad_packed_sequence(feature_teacher, batch_first=True)

            length = torch.tensor(num_frame, dtype=torch.float32).cuda(args.gpu)
            length = torch.unsqueeze(length, dim=1)
            encoding_student = (1 / length).mul(torch.sum(feature_pad, dim=1))   # TAP
            encoding_teacher = (1 / length).mul(torch.sum(feature_pad_teacher, dim=1))
            encoding_student = mlp(encoding_student)

            teacher_student_loss = teacher_student_loss_func(encoding_teacher, encoding_student) / args.batch_size
            all_loss = all_loss + args.lambda_ts * teacher_student_loss
            loss_batchs[batch_id, 2] = teacher_student_loss.item()
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        loss_mean = np.mean(loss_batchs, 0)
        loss_epochs[epoch - args.start_epoch, :] = loss_mean
        print('epoch:{:3d}, recn_lossF:{:7.3f},   recn_lossL:{:7.3f},   teach_stud_loss:{:7.4f}'.format(epoch, loss_mean[0], loss_mean[1], loss_mean[2]))
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch - args.start_epoch)
        if args.if_recon_forwardly:
            writer.add_scalar('recn_lossF', loss_mean[0], epoch - args.start_epoch)
        if args.if_recon_reversely:
            writer.add_scalar('recn_lossL', loss_mean[1], epoch - args.start_epoch)
        writer.add_scalar('teach_stud_loss', loss_mean[2], epoch - args.start_epoch)

    time_elapsed = time.time() - since
    print('==> pretraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('==> saving model')
    state = {'opt': args,
             'epoch': epoch,
             'encoder': encoder.state_dict(),
             'optim_enc': optimizer.state_dict()}
    save_path_state = os.path.join(args.model_folder, 'epoch{}state.pth'.format(epoch))
    torch.save(state, save_path_state)

    return state


def main():
    arguments = parse_option()

    if arguments.reconstruct:
        state_pretrain = main_reconstruct(arguments)

    if arguments.contrastive:
        state_pretrain = main_contrastive(arguments)

    if arguments.teacher_student:
        checkpoint = torch.load(arguments.checkpoint_path, map_location='cpu')
        encoder_pretrained = checkpoint['encoder']
        state_pretrain = main_teacher_student(arguments, encoder_pretrained)

    print('==> pretraining done!\n\n\n')
    return state_pretrain


if __name__ == '__main__':
    main()
