import sys
import os
import datetime
import argparse

import numpy as np
import numpy_indexed as npi
import numpy_groupies as npg

import commentjson
import laspy
import glob

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from functools import partial


import random
random.seed(777)
np.random.seed(777)

parser = argparse.ArgumentParser(description="Parsing the main command options")
parser.add_argument("--mode", default="train", type=str, help="Choose your mode: train/test/apply/inspect")
# parser.add_argument("--mode", default="test", type=str, help="Choose your mode: train/test/apply/inspect")
# parser.add_argument("--mode", default="apply", type=str, nargs="+", help="Choose your mode: train/test/apply/inspect")
parser.add_argument("--config_dir", default="../config/", type=str, help="Path containing your configuration json file and your train/test/app_list.txt")
parser.add_argument("--config_file", default="config.json", type=str, help="Your configuration json file name")
args = parser.parse_args()


class Logger(object):
    def __init__(self,name,mode):
        self.file=open(name,mode)
        self.stdout=sys.stdout
        sys.stdout=self
    def __del__(self):
        self.close()
    def close(self):
        if self.stdout is not None:
            sys.stdout=self.stdout
            self.stdout=None
        if self.file is not None:
            self.file.close()
            self.file=None

    def write(self,data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()

def save_pcd_txt(fname,data):
    np.savetxt(fname,data,fmt=['%.3f','%.3f','%.3f']+['%d']*(data.shape[1]-3))

def rotate_point_cloud_by_angle(pcd,pcd_median,rotation_angle):
    cos_ang=np.cos(rotation_angle)
    sin_ang=np.sin(rotation_angle)
    rotation_matrix=np.array([[cos_ang,sin_ang,0.0],[-sin_ang,cos_ang,0.0],[0.0,0.0,1.0]])
    rotated_pcd=np.dot(pcd-pcd_median,rotation_matrix)+pcd_median
    return rotated_pcd

def classAccuracy(pred,ref,n_classes):
    if len(pred.shape)==2:
        pred=np.argmax(pred,axis=1)
    support = np.bincount(ref, minlength=n_classes + 1)[:n_classes]
    relevant = np.bincount(pred, minlength=n_classes + 1)[:n_classes]
    tp_mask = np.where(pred == ref, ref, n_classes)
    tp = np.bincount(tp_mask, minlength=n_classes + 1)[:n_classes]
    iou = tp / (support + relevant - tp)
    return iou

def voxelAccuracy(preds,ref):
    if len(preds.shape)==2:
        pred=np.argmax(preds,axis=1)
    else:
        pred=preds
    return np.sum(pred==ref)*1.0/np.sum(ref>=0)



class Trainer(object):
    def __init__(self, model, optimizer,train_loader, val_loader, max_iter,iterations_per_epoch,model_best_save_path,logger=None, size_average=True, iterations_display_interval=None):

        self.model = model
        self.optim = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.size_average = size_average
        self.max_miou=0.0

        self.model_best_save_path=model_best_save_path
        self.logger=logger
        self.ann_ref=[]

        self.max_mean_iou=model.best_mIoU
        self.epoch_completed = 0

        if iterations_display_interval is None:
            self.iterations_display_interval = len(self.train_loader)
        else:
            self.iterations_display_interval = iterations_display_interval
        if iterations_display_interval is None:
            self.iterations_display_interval = len(self.train_loader)
        else:
            self.iterations_per_epoch = iterations_per_epoch

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0
        self.logger = logger


    def calc_iou(self, pred, truth, label_num):
        if len(pred.shape) > 2:
            pred = pred.argmax(axis=-1)
        if len(pred.shape) > 1:
            pred = np.reshape(pred, -1).astype(np.int32)
            truth = np.reshape(truth, -1).astype(np.int32)
        support = np.bincount(truth, minlength=label_num + 1)[:label_num]
        relevant = np.bincount(pred, minlength=label_num + 1)[:label_num]
        tp_mask = np.where(pred == truth, truth, label_num)
        tp = np.bincount(tp_mask, minlength=label_num + 1)[:label_num]

        iou = tp / (support + relevant - tp)
        return iou

    def calc_accu(self, h, t):
        if len(h.shape) > 2:
            h = h.argmax(axis=-1)
        return np.sum(h == t) * 1.0 / np.sum(t >= 0)

    def validate(self):
        training = self.model.training
        self.model.eval()

        nbmat_sz=self.val_loader.nbmat_sz
        nb_tsz = int(nbmat_sz[0] * nbmat_sz[1] * nbmat_sz[2])
        attr_ct = self.val_loader.attr_ct
        num_classes=self.val_loader.num_classes

        pcd_ann_preds = []
        mious = []

        for i, (nb_idxs, nb_vals, nb_anns, nb_pcd_idxs, nb_inverse_idxs, fname) in enumerate(self.val_loader):
            pcd_ann_pred = np.zeros([len(self.val_loader.columns_all[i]), 2],dtype=np.int32) + 1

            for k, idx in enumerate(nb_idxs):
                x = torch.zeros(nb_tsz, attr_ct)
                if len(idx) > 0:
                    x[idx, :] = torch.from_numpy(nb_vals[k].astype(np.float32))

                x = torch.moveaxis(x.reshape((1, *nbmat_sz, attr_ct)).float(), -1, 1)
                x = torch.swapaxes(x, -1, 2)

                ref = torch.zeros(nb_tsz, 1)
                if len(idx) > 0:
                    ref[idx, 0] = torch.from_numpy(nb_anns[k].astype(np.float32))
                ref = torch.moveaxis(ref.reshape((1, *nbmat_sz, 1)).float(), -1, 1)
                ref = torch.swapaxes(ref, -1, 2)

                x, ref = Variable(x), Variable(ref)
                h = self.model(x.to('cuda'))

                h_nonzero = torch.moveaxis(torch.unsqueeze(torch.moveaxis(torch.swapaxes(h, -1, 2), 1, -1).reshape((nb_tsz, num_classes))[idx, :], 0), -1, 1)
                t_nonzero = torch.moveaxis(torch.unsqueeze(torch.moveaxis(torch.swapaxes(ref, -1, 2), 1, -1).reshape([nb_tsz, ])[idx], 0), -1,1)[0].int()

                h_nonzero=torch.argmax(h_nonzero[0], dim=0)

                nb_ann_pred=h_nonzero.cpu().detach().numpy()
                nb_ann=t_nonzero.cpu().detach().numpy()

                nb_ann_pred[nb_ann_pred == 0] = 1
                pcd_ann_pred[nb_pcd_idxs[k], 1] = nb_ann_pred[nb_inverse_idxs[k]]
                pcd_ann_pred[nb_pcd_idxs[k], 0] = nb_ann[nb_inverse_idxs[k]]

            accu = voxelAccuracy(pcd_ann_pred[:, -1], pcd_ann_pred[:, -2])
            ious = classAccuracy(pcd_ann_pred[:, -1], pcd_ann_pred[:, -2], num_classes)
            ious = ious[1:]  # exclude empty cell class

            miou = np.mean(ious[~np.isnan(ious)])
            mious.append(miou)
            if self.logger is not None:
                self.logger.write(f"%s ---> Start validating %s (OA, IoU[other], IoU[vegetation], mIoU):\t" % (datetime.datetime.now(), fname))
                self.logger.write("%10.5f\t" % (accu))
                for l in ious:
                    self.logger.write("\t%10.5f" % (l))
                self.logger.write("\t%10.5f\n" % (miou))
            pcd_ann_preds.append(pcd_ann_pred)

        miou = np.mean(np.array(mious))
        self.logger.write("mean IoU: %10.5f\n" % (miou))

        if miou > self.max_mean_iou:
            if self.logger is not None:
                self.logger.write("Found the highest validating mean IoU: %10.5f. Saving the best model...\n" % (miou))
            self.model.best_mIoU = miou
            paras=self.model.state_dict()
            paras.update({'best_mIoU':torch.tensor(miou)})
            torch.save(paras, self.model_best_save_path)
            self.max_mean_iou = miou

        if self.logger is not None:
            self.logger.write("%s ---> Stop validating...\n" % datetime.datetime.now())
            self.logger.write("****************** Epochs completed: %d ******************\n" % (self.epoch_completed))
            self.logger.flush()

        self.epoch_completed += 1

        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()
        nbmat_sz=self.train_loader.nbmat_sz
        nb_tsz = int(nbmat_sz[0] * nbmat_sz[1] * nbmat_sz[2])
        attr_ct = self.train_loader.attr_ct
        num_classes=self.train_loader.num_classes

        for batch_idx, (idx,val,ann) in enumerate(self.train_loader):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue
            self.iteration = iteration

            if self.iteration % self.iterations_per_epoch == 0:
                self.validate()
            assert self.model.training

            x = torch.zeros(nb_tsz, attr_ct)
            if len(idx) > 0:
                x[idx, :] = torch.from_numpy(val.astype(np.float32))

            x = torch.moveaxis(x.reshape((1, *nbmat_sz, attr_ct)).float(), -1, 1)
            x = torch.swapaxes(x, -1, 2)

            ref = torch.zeros(nb_tsz,1)
            if len(idx) > 0:
                ref[idx, 0] = torch.from_numpy(ann.astype(np.float32))
            ref = torch.moveaxis(ref.reshape((1, *nbmat_sz, 1)).float(), -1, 1)
            ref = torch.swapaxes(ref, -1, 2)

            x, ref = Variable(x), Variable(ref)

            self.optim.zero_grad()
            h = self.model(x.to('cuda'))

            h_nonzero = torch.moveaxis(torch.unsqueeze(torch.moveaxis(torch.swapaxes(h, -1, 2), 1, -1).reshape((nb_tsz, num_classes))[idx, :], 0), -1, 1)
            t_nonzero = torch.moveaxis(torch.unsqueeze(torch.moveaxis(torch.swapaxes(ref, -1, 2), 1, -1).reshape([nb_tsz, ])[idx], 0), -1,1).long()
            t_nonzero = t_nonzero.to('cuda')

            loss = self.model.cross_entropy_loss(h_nonzero, t_nonzero)

            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()

            if self.iteration % self.iterations_display_interval == 0:
                pred = np.argmax(h_nonzero.data.cpu().numpy(),axis=1)
                ref = t_nonzero.data.cpu().numpy()
                accu = self.calc_accu(pred, ref)
                ious=self.calc_iou(pred,ref,num_classes)
                miou=np.mean(ious)

                if self.logger is not None:
                    self.logger.write('%10d\t%10.5f\t%10.5f' % (self.iteration, loss_data, accu))
                    for l in ious:
                        self.logger.write('\t%10.5f' % (l))
                    self.logger.write('\t%10.5f\n' % (miou))

            if self.iteration >= self.max_iter:
                break

    def train(self):
        max_epoch = int(np.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in range(max_epoch):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break

class DataGenerator(Dataset):
    def __init__(self, data_dir, pcd_fnames, min_res,nbmat_sz, attrs, num_classes, intensity_scale,return_scale, height_scale):
        self.data_dir=data_dir
        self.min_res=min_res
        self.nbmat_sz=nbmat_sz
        self.nb_length = self.min_res * self.nbmat_sz * 0.5
        self.intensity_scale = intensity_scale
        self.return_scale = return_scale
        self.height_scale = height_scale
        self.attrs  = attrs
        self.attr_ct  = len(attrs)
        self.num_classes  = num_classes
        self.pcd_fnames = pcd_fnames

    def generate_block_input(self,columns_sp):
        sp_min = np.min(columns_sp[:, :3], axis=0)
        nb_ijk = np.floor((columns_sp[:, :3] - sp_min) / self.min_res)

        nb_within = np.all((nb_ijk < self.nbmat_sz) & (nb_ijk >= 0), axis=1)
        nb_ijk = nb_ijk[nb_within]

        nb_idx = np.ravel_multi_index(np.transpose(nb_ijk.astype(np.int32)), self.nbmat_sz)
        nb_val = np.ones([len(nb_ijk), 1])
        if 3 in self.attrs:
            nb_val = np.concatenate([nb_val, np.expand_dims((columns_sp[nb_within, 2] - np.min(columns_sp[:, 2])) / self.height_scale, -1)], axis=1)
        if 2 in self.attrs:  # intensity
            nb_val = np.concatenate([nb_val, np.expand_dims(columns_sp[nb_within, -3] / self.intensity_scale, -1)], axis=1)
        if 1 in self.attrs:  # return
            nb_val = np.concatenate([nb_val, np.expand_dims(columns_sp[nb_within, -2] / self.return_scale, -1)], axis=1)

        nb_ann = columns_sp[nb_within, -1]
        nb_idx_u, nb_idx_u_idx, nb_inverse_idxs = np.unique(nb_idx, return_index=True, return_inverse=True)
        nb_ann_u = npg.aggregate(nb_inverse_idxs, nb_ann, func='max', fill_value=0)  # vegetation(class=2) has the priority of voxel annotation when mixed with ground
        nb_val_u = nb_val[nb_idx_u_idx]

        return nb_idx_u,nb_val_u,nb_ann_u,nb_within,nb_inverse_idxs

class TrainDataGenerator(DataGenerator):
    def __init__(self,  data_dir, pcd_fnames, min_res,nbmat_sz, attrs, num_classes, intensity_scale,return_scale, height_scale, myLogger=None,data_augment=False):
        super(TrainDataGenerator, self).__init__(data_dir, pcd_fnames, min_res, nbmat_sz, attrs, num_classes, intensity_scale,return_scale, height_scale)
        self.data_augment=data_augment
        self.columns_all = self.read_all_data(myLogger)

    def __len__(self):
        return len(self.columns_all)

    def __getitem__(self, i):
        has_example = self.get_random_example(i)
        while not has_example:
            has_example=self.get_random_example(i)
        return self.nb_idx, self.nb_val, self.nb_ann
    def get_random_example(self, i):
        random_rotation_angle = np.random.uniform() * 2 * np.pi
        random_float = np.random.rand(1,2)

        columns = self.columns_all[i]
        pcd_min = np.min(columns[:, :2], axis=0)
        xy_min = pcd_min + self.nb_length[:2]  # not include points near edge
        xy_max = np.max(columns[:, :2], axis=0) - self.nb_length[:2]  # not include points near edge
        center_xy = xy_min + (random_float * (xy_max - xy_min))

        columns_sub = columns[np.all(np.abs(columns[:, :2] - center_xy) < self.nb_length[:2], axis=1), :]
        if len(columns_sub) == 0:
            return False
        center_z = np.min(columns_sub[:, 2]) + self.nb_length[2] - 0.001
        columns_sp = columns_sub[np.abs(columns_sub[:, 2] - center_z) < self.nb_length[2], :]

        if len(columns_sp) == 0:
            return False
        else:
            if self.data_augment:
                rotation_center = np.append(center_xy, center_z)
                columns_sp[:, :3] = rotate_point_cloud_by_angle(columns_sp[:, :3], rotation_center,random_rotation_angle)

            nb_idx_u, nb_val_u, nb_ann_u,_,_=self.generate_block_input(columns_sp)

            self.nb_idx = nb_idx_u
            self.nb_val = nb_val_u
            self.nb_ann = nb_ann_u
            return True

    def read_all_data(self, myLogger):
        timing = datetime.datetime.now()
        myLogger.write("{} ---> Start reading the dataset.\n".format(timing))

        columns_all = []
        for fid, pcd_fname in enumerate(self.pcd_fnames):
            las = laspy.open(glob.glob(os.path.join(self.data_dir, pcd_fname + '.la*'))[0]).read()
            assert hasattr(las,'intensity')
            assert hasattr(las,'number_of_returns')
            assert hasattr(las,'Veg'), "las file should have manually added scalar field called 'Veg' with vegetation/tree points 2 and other points 1"
            columns = np.transpose(np.array([las.x, las.y, las.z, las.intensity,las.number_of_returns,las.Veg]))  #las.point_format.dimension_names, las.Veg is the reference class number(other:1,tree:2)
            columns_all.append(columns)

        timing = datetime.datetime.now()
        myLogger.write("{} ---> Successfully read {} files in total.\n".format(timing, len(self.pcd_fnames)))
        myLogger.flush()
        # myLogger.close()

        return columns_all

class ValidDataGenerator(DataGenerator):
    def __init__(self, data_dir, pcd_fnames, min_res,nbmat_sz, attrs, num_classes, intensity_scale,return_scale, height_scale, if_apply=False,myLogger=None):
        super(ValidDataGenerator, self).__init__(data_dir, pcd_fnames, min_res, nbmat_sz, attrs, num_classes, intensity_scale, return_scale, height_scale)
        self.if_apply=if_apply
        self.columns_all,self.pcd_min_all, self.block_ij_unq_all,self.block_idx_groups_all = self.read_all_data(myLogger)

    def __len__(self):
        return len(self.columns_all)

    def __getitem__(self, i):
        nb_idxs = []
        nb_vals = []
        nb_anns = []
        nb_inverse_idxs=[]
        nb_pcd_idxs=[]

        columns = self.columns_all[i]
        block_idx_groups = self.block_idx_groups_all[i]

        for iter, idx in enumerate(block_idx_groups):
            columns_sub = columns[idx, :]
            center_z = np.min(columns_sub[:, 2]) + self.nb_length[2] - 0.001
            idx_sel=np.abs(columns_sub[:, 2] - center_z) < self.nb_length[2]
            columns_sp = columns_sub[idx_sel, :]
            nb_pcd_idx = idx[idx_sel]

            nb_idx_u, nb_val_u, nb_ann_u,nb_within,nb_inverse_idx = self.generate_block_input(columns_sp)
            nb_pcd_idx = nb_pcd_idx[nb_within]

            nb_idxs.append(nb_idx_u)
            nb_vals.append(nb_val_u)
            nb_anns.append(nb_ann_u)
            nb_inverse_idxs.append(nb_inverse_idx)
            nb_pcd_idxs.append(nb_pcd_idx)

        return nb_idxs, nb_vals, nb_anns, nb_pcd_idxs, nb_inverse_idxs,self.pcd_fnames[i]

    def read_all_data(self,myLogger=None):
        timing = datetime.datetime.now()
        myLogger.write("{} ---> Start reading the validating/testing dataset.\n".format(timing))

        columns_all=[]
        pcd_min_all=[]
        block_ij_unq_all=[]
        block_idx_groups_all=[]

        for fid, pcd_fname in enumerate(self.pcd_fnames):
            las = laspy.open(glob.glob(os.path.join(self.data_dir, pcd_fname + '.la*'))[0]).read()
            assert hasattr(las,'intensity')
            assert hasattr(las,'number_of_returns')
            if self.if_apply:  # not app_process
                columns = np.transpose(np.array([las.x, las.y, las.z,las.intensity, las.number_of_returns]))  # las.point_format.dimension_names
            else:
                assert hasattr(las, 'Veg'), "las file should have manually added scalar field called 'Veg' with vegetation/tree points 2 and other points 1"
                columns = np.transpose(np.array([las.x, las.y, las.z, las.intensity, las.number_of_returns, las.Veg]))  # las.point_format.dimension_names

            pcd_min = np.min(columns[:, :2], axis=0)
            vox_ij = np.floor((columns[:, :2] - pcd_min) / self.min_res[:2]).astype(np.int32)
            block_ij = np.floor(vox_ij / self.nbmat_sz[:2])
            block_ij_unq, block_idx_groups = npi.group_by(block_ij, np.arange(len(block_ij)))

            pcd_min_all.append(pcd_min)
            block_ij_unq_all.append(block_ij_unq)
            block_idx_groups_all.append(block_idx_groups)

            columns[:,:2]=columns[:,:2]-pcd_min
            columns_all.append(columns)

        timing = datetime.datetime.now()
        myLogger.write("{} ---> Successfully read {} files in total.\n".format(timing,len(self.pcd_fnames)))
        myLogger.flush()
        return columns_all,pcd_min_all,block_ij_unq_all,block_idx_groups_all


def train(configs):
    data_dir = configs["dataset"]["data_path"]
    logs_dir = configs["training"]["output_folder"]

    nbmat_sz = np.array(configs["model"]["voxel_number_in_block"])
    min_res = np.array(configs["model"]["voxel_resolution_in_meter"])
    attrs = configs["model"]["attribute_list"]

    num_epoches = configs["training"]["num_epoches"]
    num_iters = configs["training"]["iterations_per_epoch"]
    display_interval = configs["training"]["iterations_display_interval"]
    if_data_augment = configs["training"]["data_augment"]

    num_classes = len(np.array(configs["dataset"]["classes"])) + 1  # include a empty cell class. empty:0,ground:1,vegetation/tree:2
    height_scale = float(configs["dataset"]["normalize_height"])
    intensity_scale = float(configs["dataset"]["normalize_intensity"])
    return_scale = float(configs["dataset"]["normalize_return"])

    attr_ct = len(attrs)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)

    model_type = configs["model"]["type"]
    title = "vox3D{0}_nb{1}_res{2}_ep{3}".format(model_type, nbmat_sz[0], min_res[0], num_epoches)

    myLogger = Logger(os.path.join(logs_dir, title + "_train_logs.txt"), 'w')
    timing = datetime.datetime.now()
    myLogger.write("%s ---> Start processing.\n" % timing)
    myLogger.write("Starting to train " + title + "\n")

    train_list = configs["dataset"]["training_files"]
    with open(os.path.join(args.config_dir, train_list), 'r') as prep_file:
        train_pcd_fnames = prep_file.read().split('\n')
    train_pcd_fnames = [pcd_fname for pcd_fname in train_pcd_fnames if pcd_fname]

    train_generator = TrainDataGenerator(data_dir, train_pcd_fnames, min_res,nbmat_sz, attrs, num_classes, intensity_scale,height_scale, return_scale, myLogger=myLogger,data_augment=if_data_augment)

    valid_list = configs["dataset"]["validating_files"]
    with open(os.path.join(args.config_dir, valid_list), 'r') as prep_file:
        valid_pcd_fnames = prep_file.read().split('\n')
    valid_pcd_fnames = [pcd_fname for pcd_fname in valid_pcd_fnames if pcd_fname]
    valid_generator = ValidDataGenerator(data_dir, valid_pcd_fnames, min_res, nbmat_sz, attrs, num_classes, intensity_scale, return_scale, height_scale,if_apply=False,myLogger=myLogger)

    model_best_save_path = os.path.join(logs_dir, "best_model.pth")

    if os.path.isfile(model_best_save_path):
        pretrained=True
    else:
        pretrained=False

    if model_type == "ResNet50":
        try:
            from .vox3DResNetModel import resnet50
        except Exception:  # ImportError
            from vox3DResNetModel import resnet50
        model = resnet50(num_classes=num_classes, num_attr=attr_ct, model_best_save_path=model_best_save_path,pretrained=pretrained)

    elif model_type == "SegFormer":
        try:
            from .vox3DSegFormer import Segformer
        except Exception:  # ImportError
            from vox3DSegFormer import Segformer

        patch_size = configs["model"]["patch_size"]
        decoder_dim = configs["model"]["decoder_dim"]
        channel_dims = configs["model"]["channel_dims"]
        SR_ratios = configs["model"]["SR_ratios"]
        num_heads = configs["model"]["num_heads"]
        MLP_ratios = configs["model"]["MLP_ratios"]
        depths = configs["model"]["depths"]
        qkv_bias = configs["model"]["qkv_bias"]
        drop_rate = configs["model"]["drop_rate"]
        drop_path_rate = configs["model"]["drop_path_rate"]

        model = Segformer(
            pretrained=pretrained,
            block3d_size=nbmat_sz,
            in_chans=attr_ct,
            num_classes=num_classes,
            patch_size=patch_size,
            embed_dims=channel_dims,
            num_heads=num_heads,
            mlp_ratios=MLP_ratios,
            qkv_bias=qkv_bias,
            depths=depths,
            sr_ratios=SR_ratios,
            drop_rate=drop_rate, drop_path_rate=drop_path_rate,
            decoder_dim=decoder_dim,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            model_best_save_path=model_best_save_path
        )
    else:
        myLogger.write("Please edit the config*.json file with a correct classifier name (either ResNet50 or SegFormer){}\n".format(model_best_save_path))
        return

    if os.path.isfile(model_best_save_path):
        myLogger.write("Load model from {}\n".format(model_best_save_path))

    model = model.cuda()
    optim = torch.optim.Adam(model.parameters())
    trainer = Trainer(
        model=model,
        optimizer=optim,
        train_loader=train_generator,
        val_loader=valid_generator,
        max_iter=num_iters * num_epoches,
        iterations_per_epoch=num_iters,
        iterations_display_interval=display_interval,
        model_best_save_path=model_best_save_path,
        logger=myLogger
    )

    start_epoch = 0
    start_iteration = 0
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


def test(configs):
    data_dir = configs["dataset"]["data_path"]
    logs_dir = configs["testing"]["output_folder"]
    train_logs_dir = configs["training"]["output_folder"]

    nbmat_sz = np.array(configs["model"]["voxel_number_in_block"])
    min_res = np.array(configs["model"]["voxel_resolution_in_meter"])
    attrs = configs["model"]["attribute_list"]

    num_classes = len(np.array(configs["dataset"]["classes"])) + 1  # include a empty cell class. empty:0,ground:1,vegetation/tree:2

    height_scale = float(configs["dataset"]["normalize_height"])
    intensity_scale = float(configs["dataset"]["normalize_intensity"])
    return_scale = float(configs["dataset"]["normalize_return"])

    attr_ct = len(attrs)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)

    model_type = configs["model"]["type"]
    title = "vox3D{0}_nb{1}_res{2}".format(model_type, nbmat_sz[0], min_res[0])

    myLogger = Logger(os.path.join(logs_dir, title + "_test_logs.txt"), 'w')
    timing = datetime.datetime.now()
    myLogger.write("%s ---> Start processing.\n" % timing)
    myLogger.write("Starting to test " + title + "\n")

    test_list = configs["dataset"]["testing_files"]
    with open(os.path.join(args.config_dir, test_list), 'r') as prep_file:
        test_pcd_fnames = prep_file.read().split('\n')
    test_pcd_fnames = [pcd_fname for pcd_fname in test_pcd_fnames if pcd_fname]

    test_generator = ValidDataGenerator(data_dir, test_pcd_fnames, min_res, nbmat_sz, attrs, num_classes, intensity_scale, return_scale, height_scale, if_apply=False, myLogger=myLogger)

    model_best_save_path = os.path.join(train_logs_dir, "best_model.pth")

    if os.path.isfile(model_best_save_path):
        pretrained = True
    else:
        pretrained = False

    if model_type == "ResNet50":
        try:
            from .vox3DResNetModel import resnet50
        except Exception:  # ImportError
            from vox3DResNetModel import resnet50
        model = resnet50(num_classes=num_classes, num_attr=attr_ct, model_best_save_path=model_best_save_path,pretrained=pretrained)

    elif model_type == "SegFormer":
        try:
            from .vox3DSegFormer import Segformer
        except Exception:  # ImportError
            from vox3DSegFormer import Segformer

        patch_size = configs["model"]["patch_size"]
        decoder_dim = configs["model"]["decoder_dim"]
        channel_dims = configs["model"]["channel_dims"]
        SR_ratios = configs["model"]["SR_ratios"]
        num_heads = configs["model"]["num_heads"]
        MLP_ratios = configs["model"]["MLP_ratios"]
        depths = configs["model"]["depths"]
        qkv_bias = configs["model"]["qkv_bias"]
        drop_rate = configs["model"]["drop_rate"]
        drop_path_rate = configs["model"]["drop_path_rate"]

        model = Segformer(
            pretrained=pretrained,
            block3d_size=nbmat_sz,
            in_chans=attr_ct,
            num_classes=num_classes,
            patch_size=patch_size,
            embed_dims=channel_dims,
            num_heads=num_heads,
            mlp_ratios=MLP_ratios,
            qkv_bias=qkv_bias,
            depths=depths,
            sr_ratios=SR_ratios,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            decoder_dim=decoder_dim,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            model_best_save_path=model_best_save_path
        )
    else:
        myLogger.write("Please edit the config*.json file with a correct classifier name (either ResNet50 or SegFormer){}\n".format(model_best_save_path))
        return

    if os.path.isfile(model_best_save_path):
        myLogger.write("Load model from {}\n".format(model_best_save_path))

    model = model.cuda()
    model.eval()

    nb_tsz = int(nbmat_sz[0] * nbmat_sz[1] * nbmat_sz[2])

    mious=[]
    pcd_ann_preds=[]
    for i, (nb_idxs, nb_vals, nb_anns, nb_pcd_idxs,nb_inverse_idxs, fname) in enumerate(test_generator):
        if myLogger is not None:
            myLogger.write("%s ---> Start testing %s. " % (datetime.datetime.now(), fname))

        pcd=test_generator.columns_all[i]
        pcd_ann_pred = np.zeros(len(pcd),dtype=np.int32) + 1
        for k, idx in enumerate(nb_idxs):
            x = torch.zeros(nb_tsz, attr_ct)
            if len(idx) > 0:
                x[idx, :] = torch.from_numpy(nb_vals[k].astype(np.float32))

            x = torch.moveaxis(x.reshape((1, *nbmat_sz, attr_ct)).float(), -1, 1)
            x = torch.swapaxes(x, -1, 2)

            x = Variable(x)
            h = model(x.to('cuda'))

            h_nonzero = torch.moveaxis(torch.unsqueeze(torch.moveaxis(torch.swapaxes(h, -1, 2), 1, -1).reshape((nb_tsz, num_classes))[idx, :], 0), -1, 1)
            h_nonzero=torch.argmax(h_nonzero[0], dim=0)

            nb_ann_pred=h_nonzero.cpu().detach().numpy()
            nb_ann_pred[nb_ann_pred == 0] = 1
            pcd_ann_pred[nb_pcd_idxs[k]] = nb_ann_pred[nb_inverse_idxs[k]]

        accu = voxelAccuracy(pcd_ann_pred, pcd[:,-1].astype(np.int32))
        ious = classAccuracy(pcd_ann_pred, pcd[:,-1].astype(np.int32), num_classes)
        ious = ious[1:]  # exclude empty cell class

        miou = np.mean(ious[~np.isnan(ious)])
        mious.append(miou)

        if myLogger is not None:
            myLogger.write("%s ---> Start testing %s (OA, IoU[other], IoU[vegetation], mIoU):\t" % (datetime.datetime.now(), fname))
            myLogger.write("%10.5f\t" % (accu))
            for l in ious:
                myLogger.write("\t%10.5f" % (l))
            myLogger.write("\t%10.5f\n" % (miou))
        pcd_ann_preds.append(pcd_ann_pred)

    miou = np.mean(np.array(mious))
    myLogger.write("mean IoU: %10.5f\n" % (miou))

    for k, pcd_ann_pred in enumerate(pcd_ann_preds):
        fname = glob.glob(os.path.join(data_dir, test_generator.pcd_fnames[k] + '.la*'))[0]
        las = laspy.open(fname).read()
        las.add_extra_dim(laspy.ExtraBytesParams(name="VegCls", type="int32", description="Pred"))
        las.VegCls=pcd_ann_pred
        las.write(os.path.join(logs_dir, "test_{}_vegcls.laz".format(test_generator.pcd_fnames[k])))

    if myLogger is not None:
        myLogger.write("%s ---> Stop testing...\n" % (datetime.datetime.now()))
        myLogger.flush()

def apply(configs):
    data_dir = configs["dataset"]["data_application_path"]
    logs_dir = configs["application"]["output_folder"]
    train_logs_dir = configs["training"]["output_folder"]

    nbmat_sz = np.array(configs["model"]["voxel_number_in_block"])
    min_res = np.array(configs["model"]["voxel_resolution_in_meter"])
    attrs = configs["model"]["attribute_list"]

    num_classes = len(np.array(configs["dataset"]["classes"])) + 1  # include a empty cell class. empty:0,ground:1,nonground:2

    height_scale = float(configs["dataset"]["normalize_height"])
    intensity_scale = float(configs["dataset"]["normalize_intensity"])
    return_scale = float(configs["dataset"]["normalize_return"])

    attr_ct = len(attrs)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)

    model_type = configs["model"]["type"]
    title = "vox3D{0}_nb{1}_res{2}".format(model_type, nbmat_sz[0], min_res[0])

    myLogger = Logger(os.path.join(logs_dir, title + "_app_logs.txt"), 'w')
    timing = datetime.datetime.now()
    myLogger.write("%s ---> Start processing.\n" % timing)
    myLogger.write("Starting to apply " + title + "\n")

    app_list = configs["dataset"]["application_files"]
    with open(os.path.join(args.config_dir, app_list), 'r') as prep_file:
        app_pcd_fnames = prep_file.read().split('\n')
    app_pcd_fnames = [pcd_fname for pcd_fname in app_pcd_fnames if pcd_fname]

    app_generator = ValidDataGenerator(data_dir, app_pcd_fnames, min_res, nbmat_sz, attrs, num_classes, intensity_scale, return_scale, height_scale, if_apply=True, myLogger=myLogger)

    model_best_save_path = os.path.join(train_logs_dir, "best_model.pth")

    if os.path.isfile(model_best_save_path):
        pretrained = True
    else:
        pretrained = False

    if model_type == "ResNet50":
        try:
            from .vox3DResNetModel import resnet50
        except Exception:  # ImportError
            from vox3DResNetModel import resnet50
        model = resnet50(num_classes=num_classes, num_attr=attr_ct, model_best_save_path=model_best_save_path,pretrained=pretrained)
    elif model_type == "SegFormer":
        try:
            from .vox3DSegFormer import Segformer
        except Exception:  # ImportError
            from vox3DSegFormer import Segformer

        patch_size = configs["model"]["patch_size"]
        decoder_dim = configs["model"]["decoder_dim"]
        channel_dims = configs["model"]["channel_dims"]
        SR_ratios = configs["model"]["SR_ratios"]
        num_heads = configs["model"]["num_heads"]
        MLP_ratios = configs["model"]["MLP_ratios"]
        depths = configs["model"]["depths"]
        qkv_bias = configs["model"]["qkv_bias"]
        drop_rate = configs["model"]["drop_rate"]
        drop_path_rate = configs["model"]["drop_path_rate"]

        model = Segformer(
            pretrained=pretrained,
            block3d_size=nbmat_sz,
            in_chans=attr_ct,
            num_classes=num_classes,
            patch_size=patch_size,
            embed_dims=channel_dims,
            num_heads=num_heads,
            mlp_ratios=MLP_ratios,
            qkv_bias=qkv_bias,
            depths=depths,
            sr_ratios=SR_ratios,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            decoder_dim=decoder_dim,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            model_best_save_path=model_best_save_path
        )
    else:
        myLogger.write("Please edit the config*.json file with a correct classifier name (either ResNet50 or SegFormer){}\n".format(model_best_save_path))
        return

    if os.path.isfile(model_best_save_path):
        myLogger.write("Load model from {}\n".format(model_best_save_path))

    model = model.cuda()
    model.eval()

    nb_tsz = int(nbmat_sz[0] * nbmat_sz[1] * nbmat_sz[2])

    for i, (nb_idxs, nb_vals, nb_anns, nb_pcd_idxs,nb_inverse_idxs, fname) in enumerate(app_generator):
        if myLogger is not None:
            myLogger.write("%s ---> Start applying %s. " % (datetime.datetime.now(), fname))

        pcd=app_generator.columns_all[i]
        pcd_ann_pred = np.zeros(len(pcd),dtype=np.int32) + 1
        for k, idx in enumerate(nb_idxs):
            x = torch.zeros(nb_tsz, attr_ct)
            if len(idx) > 0:
                x[idx, :] = torch.from_numpy(nb_vals[k].astype(np.float32))

            x = torch.moveaxis(x.reshape((1, *nbmat_sz, attr_ct)).float(), -1, 1)
            x = torch.swapaxes(x, -1, 2)

            x = Variable(x)
            h = model(x.to('cuda'))

            h_nonzero = torch.moveaxis(torch.unsqueeze(
                torch.moveaxis(torch.swapaxes(h, -1, 2), 1, -1).reshape((nb_tsz, num_classes))[idx, :], 0), -1, 1)

            h_nonzero=torch.argmax(h_nonzero[0], dim=0)

            nb_ann_pred=h_nonzero.cpu().detach().numpy()
            nb_ann_pred[nb_ann_pred == 0] = 1
            pcd_ann_pred[nb_pcd_idxs[k]] = nb_ann_pred[nb_inverse_idxs[k]]

        las = laspy.open(os.path.join(data_dir,app_generator.pcd_fnames[i]+'.laz')).read()
        las.add_extra_dim(laspy.ExtraBytesParams(name="VegCls",type="int32",description="veg/ground classes"))
        las.VegCls=pcd_ann_pred
        las.write(os.path.join(logs_dir, "{}_vegcls.laz".format(app_generator.pcd_fnames[i])))
        if myLogger is not None:
            myLogger.write("%s ---> Finish applying %s.\n" % (datetime.datetime.now(), fname))
            myLogger.flush()


if __name__ == '__main__':
    with open(os.path.join(args.config_dir, args.config_file)) as json_file:
        configs = commentjson.load(json_file)

    if args.mode == "train":
        train(configs)

    if args.mode == "test":
        test(configs)

    if args.mode == "apply":
        apply(configs)