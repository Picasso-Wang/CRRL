import os
import torch
import time
import copy
import numpy as np
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from feeder import SelfDefineDataset_linEval, collate_fn_linEval
from models import Bi_GRU_packpad
from tools.adjustLearningRate import adjust_learning_rate
from main_pretrain import main
from parse_args_for_evaluation import parse_option_evaluation
from tools.scheduler import WarmupCosineSchedule


def linear_eval(args, encoder_dict):
    train_dataset = SelfDefineDataset_linEval(args.train_data_path, args.train_num_frame_path, args.train_label_path, args.dataset)
    val_dataset = SelfDefineDataset_linEval(args.val_data_path, args.val_num_frame_path, args.val_label_path, args.dataset)
    print('==> Number of training video: {}'.format(len(train_dataset)))
    print('==> Number of validate video: {}'.format(len(val_dataset)))
    datasets = {'train': train_dataset, 'val': val_dataset}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(dataset=datasets[x], batch_size=args.batch_size, shuffle=True, 
                    num_workers=args.num_workers, collate_fn=collate_fn_linEval) for x in ['train', 'val']}

    print("==> creating encoder_fc and loading parameters")
    encoder_fc = Bi_GRU_packpad(args.input_size, args.hidden_units, args.layers, args.dropout, args.n_classes)
    encoder_dict['fc.weight'] = torch.normal(0.0, 0.01, (args.n_classes, args.hidden_units * 2))
    encoder_dict['fc.bias'] = torch.zeros(args.n_classes)
    if args.if_pretrain:
        encoder_fc.load_state_dict(encoder_dict)

    print('==> freeze all layers but the last fc')
    for name, param in encoder_fc.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    print('==> encoder_fc.gru.weight_ih_l0.requires_grad:', encoder_fc.gru.weight_ih_l0.requires_grad)
    print('==> encoder_fc.fc.weight.requires_grad:', encoder_fc.fc.weight.requires_grad)

    encoder_fc = encoder_fc.cuda(args.gpu)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, encoder_fc.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.SGD(parameters, args.learning_rate, momentum=args.momentum,
                                nesterov=args.nesterov, weight_decay=args.weight_decay)

    args.start_epoch = 0

    # tensorboard
    writer = SummaryWriter(log_dir=args.tb_folder, flush_secs=30)

    print('==> training')
    since = time.time()
    best_acc1 = 0.0
    best_gt_list_train = []   # ground truth list
    best_pl_list_train = []   # predict label list
    best_gt_list_val = []
    best_pl_list_val = []
    best_at_epoch = args.start_epoch
    best_model = copy.deepcopy(encoder_fc.state_dict())

    for epoch in range(args.start_epoch, args.epochs):

        for phase in ['train', 'val']:   # Each epoch has a training and validation phase
            running_loss = 0.0
            running_corrects = 0
            ground_truth_list = []
            pred_label_list = []

            encoder_fc.eval()

            for num_frame, inputs, labels in dataloaders[phase]:  # inputs' shape (args.batch_size, args.frames, args.input_size)
                inputs, labels = inputs.cuda(args.gpu), labels.cuda(args.gpu)

                with torch.set_grad_enabled(phase == 'train'):  # track history if only in train
                    _, _, outputs = encoder_fc(inputs, num_frame, args.gpu)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    pred_label_list += list(preds.data.cpu().numpy())
                    ground_truth_list += list(labels.data.cpu().numpy())
                    optimizer.zero_grad()
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                adjust_learning_rate(epoch, args, optimizer)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc1 = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                print('Epoch: {:3d}     train_loss: {:.4f}    train_acc: {:.2f}'.format(epoch, epoch_loss, epoch_acc1*100), end='    ')
            else:
                # print('Epoch {:3d},  Loss:       {:.4f},  {:>5s}, Acc:       {:.2f}'.format(epoch, epoch_loss, phase, epoch_acc1*100))
                print('val_loss: {:.4f}    val_acc: {:.2f}'.format(epoch_loss, epoch_acc1*100))

            if phase == 'train':
                tmp_gt_list_train = ground_truth_list
                tmp_pl_list_train = pred_label_list

            # deep copy the model
            if phase == 'val' and epoch_acc1 > best_acc1:
                best_at_epoch = epoch
                best_acc1 = epoch_acc1
                best_gt_list_val = ground_truth_list
                best_pl_list_val = pred_label_list
                best_gt_list_train = tmp_gt_list_train
                best_pl_list_train = tmp_pl_list_train
                best_model = copy.deepcopy(encoder_fc.state_dict())

            if phase == 'train':
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                tmp_train_loss = epoch_loss
                tmp_train_acc1 = epoch_acc1

        writer.add_scalars('loss', {'train': tmp_train_loss, 'val': epoch_loss}, epoch)
        writer.add_scalars('acc', {'train': tmp_train_acc1, 'val': epoch_acc1}, epoch)

    state = {
        'epoch': best_at_epoch,
        'best_acc1': best_acc1,
        'encoder_fc': best_model,
    }

    print('==> saving best model!')
    confusion_mat_train = confusion_matrix(best_gt_list_train, best_pl_list_train)
    np.save(os.path.join(args.model_folder, 'confusion_matrix_train.npy'), confusion_mat_train)
    np.save(os.path.join(args.model_folder, 'ground_truth_train.npy'), best_gt_list_train)
    np.save(os.path.join(args.model_folder, 'predict_label_train.npy'), best_pl_list_train)

    confusion_mat_val = confusion_matrix(best_gt_list_val, best_pl_list_val)
    np.save(os.path.join(args.model_folder, 'confusion_matrix_val.npy'), confusion_mat_val)
    np.save(os.path.join(args.model_folder, 'ground_truth_val.npy'), best_gt_list_val)
    np.save(os.path.join(args.model_folder, 'predict_label_val.npy'), best_pl_list_val)

    save_path = os.path.join(args.model_folder, 'state_bestAcc1_{:.3f}.pth'.format(best_acc1))
    torch.save(state, save_path)

    time_elapsed = time.time() - since
    print('==> Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('==> Best val Acc1: {:.2f}'.format(best_acc1*100))
    print('\n\n')


def fine_tune(args, encoder_dict):
    train_dataset = SelfDefineDataset_linEval(args.train_data_path, args.train_num_frame_path, args.train_label_path, args.dataset)
    val_dataset = SelfDefineDataset_linEval(args.val_data_path, args.val_num_frame_path, args.val_label_path, args.dataset)
    print('==> Number of training video: {}'.format(len(train_dataset)))
    print('==> Number of validate video: {}'.format(len(val_dataset)))
    datasets = {'train': train_dataset, 'val': val_dataset}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(dataset=datasets[x], batch_size=args.batch_size,shuffle=True, 
                    num_workers=args.num_workers, collate_fn=collate_fn_linEval) for x in ['train', 'val']}

    print("==> creating encoder_fc and loading parameters")
    encoder_fc = Bi_GRU_packpad(args.input_size, args.hidden_units, args.layers, args.dropout, args.n_classes)
    encoder_dict['fc.weight'] = torch.normal(0.0, 0.01, (args.n_classes, args.hidden_units * 2))
    encoder_dict['fc.bias'] = torch.zeros(args.n_classes)
    if args.if_pretrain:
        encoder_fc.load_state_dict(encoder_dict)
    print('==> encoder_fc.gru.weight_ih_l0.requires_grad:', encoder_fc.gru.weight_ih_l0.requires_grad)
    print('==> encoder_fc.fc.weight.requires_grad:', encoder_fc.fc.weight.requires_grad)

    encoder_fc = encoder_fc.cuda(args.gpu)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD([{'params': encoder_fc.gru.parameters(), 'lr': args.learning_rate_encoder},
                                 {'params': encoder_fc.fc.parameters()}], args.learning_rate,
                                momentum=args.momentum, nesterov=args.nesterov, weight_decay=args.weight_decay)
    if args.decay_type == 'cosine':
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=(args.epochs//2) * len(train_dataset) // args.batch_size, 
                                         t_total=args.epochs * len(train_dataset) // args.batch_size)
    
    args.start_epoch = 0

    # tensorboard
    writer = SummaryWriter(log_dir=args.tb_folder, flush_secs=30)

    print('==> training')
    since = time.time()
    best_acc1 = 0.0
    best_gt_list_train = []   # ground truth list
    best_pl_list_train = []   # predict label list
    best_gt_list_val = []
    best_pl_list_val = []
    best_at_epoch = args.start_epoch
    best_model = copy.deepcopy(encoder_fc.state_dict())

    for epoch in range(args.start_epoch, args.epochs):
        for phase in ['train', 'val']:
            running_loss = 0.0
            running_corrects = 0
            ground_truth_list = []
            pred_label_list = []

            if phase == 'train':
                encoder_fc.train()
            else:
                encoder_fc.eval()

            for num_frame, inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.cuda(args.gpu), labels.cuda(args.gpu)

                with torch.set_grad_enabled(phase == 'train'):
                    _, _, outputs = encoder_fc(inputs, num_frame, args.gpu)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    pred_label_list += list(preds.data.cpu().numpy())
                    ground_truth_list += list(labels.data.cpu().numpy())

                    optimizer.zero_grad()
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                if args.decay_type == 'cosine':
                    scheduler.step()
                else:
                    adjust_learning_rate(epoch, args, optimizer)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc1 = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                print('Epoch: {:3d}     train_loss: {:.4f}    train_acc: {:.2f}'.format(epoch, epoch_loss, epoch_acc1*100), end='    ')
            else:
                # print('Epoch {:3d},  Loss:       {:.4f},  {:>5s}, Acc:       {:.2f}'.format(epoch, epoch_loss, phase, epoch_acc1*100))
                print('val_loss: {:.4f}    val_acc: {:.2f}'.format(epoch_loss, epoch_acc1*100))

            if phase == 'train':
                tmp_gt_list_train = ground_truth_list
                tmp_pl_list_train = pred_label_list

            # deep copy the model
            if phase == 'val' and epoch_acc1 > best_acc1:
                best_at_epoch = epoch
                best_acc1 = epoch_acc1
                best_gt_list_val = ground_truth_list
                best_pl_list_val = pred_label_list
                best_gt_list_train = tmp_gt_list_train
                best_pl_list_train = tmp_pl_list_train
                best_model = copy.deepcopy(encoder_fc.state_dict())

            if phase == 'train':
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                tmp_train_loss = epoch_loss
                tmp_train_acc1 = epoch_acc1

        writer.add_scalars('loss', {'train': tmp_train_loss, 'val': epoch_loss}, epoch)
        writer.add_scalars('acc', {'train': tmp_train_acc1, 'val': epoch_acc1}, epoch)

    state = {
        'epoch': best_at_epoch,
        'best_acc1': best_acc1,
        'encoder_fc': best_model,
    }

    print('==> saving best model!')
    confusion_mat_train = confusion_matrix(best_gt_list_train, best_pl_list_train)
    np.save(os.path.join(args.model_folder, 'confusion_matrix_train.npy'), confusion_mat_train)
    np.save(os.path.join(args.model_folder, 'ground_truth_train.npy'), best_gt_list_train)
    np.save(os.path.join(args.model_folder, 'predict_label_train.npy'), best_pl_list_train)

    confusion_mat_val = confusion_matrix(best_gt_list_val, best_pl_list_val)
    np.save(os.path.join(args.model_folder, 'confusion_matrix_val.npy'), confusion_mat_val)
    np.save(os.path.join(args.model_folder, 'ground_truth_val.npy'), best_gt_list_val)
    np.save(os.path.join(args.model_folder, 'predict_label_val.npy'), best_pl_list_val)

    save_path = os.path.join(args.model_folder, 'state_bestAcc1_{:.3f}.pth'.format(best_acc1))
    torch.save(state, save_path)

    time_elapsed = time.time() - since
    print('==> Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('==> Best val Acc1: {:.2f}'.format(best_acc1*100))


def knn(args, encoder_dict):
    train_dataset = SelfDefineDataset_linEval(args.train_data_path, args.train_num_frame_path, args.train_label_path, args.dataset)
    val_dataset = SelfDefineDataset_linEval(args.val_data_path, args.val_num_frame_path, args.val_label_path, args.dataset)
    print('==> Number of training video: {}'.format(len(train_dataset)))
    print('==> Number of validate video: {}'.format(len(val_dataset)))
    datasets = {'train': train_dataset, 'val': val_dataset}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(dataset=datasets[x], batch_size=args.batch_size, shuffle=True, 
                    num_workers=args.num_workers, collate_fn=collate_fn_linEval) for x in ['train', 'val']}

    print("==> creating encoder_fc and loading parameters")
    encoder_fc = Bi_GRU_packpad(args.input_size, args.hidden_units, args.layers, args.dropout, args.n_classes)
    encoder_dict['fc.weight'] = torch.normal(0.0, 0.01, (args.n_classes, args.hidden_units * 2))
    encoder_dict['fc.bias'] = torch.zeros(args.n_classes)

    if args.if_pretrain:
        encoder_fc.load_state_dict(encoder_dict)
    print('==> freeze all layers')
    for param in encoder_fc.parameters():
        param.requires_grad = False
    print('==> encoder_fc.gru.weight_ih_l0.requires_grad:', encoder_fc.gru.weight_ih_l0.requires_grad)
    print('==> encoder_fc.fc.weight.requires_grad:', encoder_fc.fc.weight.requires_grad)

    encoder_fc = encoder_fc.cuda(args.gpu)
    args.start_epoch = 0
    since = time.time()
    knn_max_score = 0.0

    args.epochs = 1
    for epoch in range(args.start_epoch, args.epochs):

        train_knn_feature = []
        test_knn_feature = []
        train_label = []
        test_label = []

        for phase in ['train', 'val']:   # Each epoch has a training and validation phase
            encoder_fc.eval()

            for num_frame, inputs, labels in dataloaders[phase]:  # inputs' shape (args.batch_size, args.frames, args.input_size)
                inputs, labels = inputs.cuda(args.gpu), labels.cuda(args.gpu)
                _, feature, _ = encoder_fc(inputs, num_frame, args.gpu)
                if phase == 'train':
                    train_knn_feature.append(feature.cpu().data.numpy())
                    train_label.append((labels.cpu().data.numpy())[:, None])  # expand a dimension
                else:
                    test_knn_feature.append(feature.cpu().data.numpy())
                    test_label.append((labels.cpu().data.numpy())[:, None])

        train_knn_feature = np.vstack(train_knn_feature)
        train_label = np.vstack(train_label)
        train_label = np.squeeze(train_label, axis=1)
        test_knn_feature = np.vstack(test_knn_feature)
        test_label = np.vstack(test_label)
        test_label = np.squeeze(test_label, axis=1)

        # # save features for TSNE visualization
        # np.save('/root/wp/action/results/features_TSNE/CRVRL_test_features.npy', test_knn_feature)
        # np.save('/root/wp/action/results/features_TSNE/CRVRL_test_labels.npy', test_label)

        neigh = KNeighborsClassifier(n_neighbors=1, metric='cosine')
        neigh.fit(train_knn_feature, train_label)
        score = neigh.score(test_knn_feature, test_label)
        print("epoch: {}, knn test score: {:.2f}".format(epoch, score*100))
        if score > knn_max_score:
            knn_max_score = score

    state = {'knn_max_score': knn_max_score, 'encoder_fc': encoder_fc}
    save_path = os.path.join(args.model_folder, 'state_bestAcc_knn_{:.3f}.pth'.format(knn_max_score))
    torch.save(state, save_path)

    time_elapsed = time.time() - since
    print('==> Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('==> Best KNN Acc: {:.2f}'.format(knn_max_score * 100))


if __name__ == '__main__':
    if_do_pretrain = False
    if if_do_pretrain:
        pretrain_checkpoint = main()
    else:
        pretrain_checkpoint_path = '/root/wp/action/results/20220607/ntu_models/ntu_xsub_teach_stud_hidden256_layer2_lr0.05_bsz32_epoch60_lambda_ts0.05/epoch59state.pth'
        pretrain_checkpoint = torch.load(pretrain_checkpoint_path, map_location='cpu')

# '/root/wp/action/results/20220607/ntu_models/ntu_xsub_contrastive_hidden256_layer2_bsz32_lr0.05_epoch60_moment0.999_k64_temp0.1/epoch59state.pth'
# '/root/wp/action/results/20220607/ntu_models/ntu_xsub_reconstruct_hidden256_layer2_lr0.05_bsz32_epoch60/epoch59state.pth'
# '/root/wp/action/results/20220607/ntu_models/ntu_xsub_teach_stud_hidden256_layer2_lr0.05_bsz32_epoch60_lambda_ts0.05/epoch59state.pth'

    encoder_pretrain = pretrain_checkpoint['encoder']
    arguments_pretrain = pretrain_checkpoint['opt']
    arguments = parse_option_evaluation(arguments_pretrain)

    if arguments.mode == 'lin_eval':
        linear_eval(arguments, encoder_pretrain)

    if arguments.mode == 'fine_tune'  or arguments.mode == 'pku1_transfer_to_pku2' or arguments.mode == 'ntu60_transfer_to_pku2':
        fine_tune(arguments, encoder_pretrain)

    if arguments.mode == 'knn':
        knn(arguments, encoder_pretrain)
