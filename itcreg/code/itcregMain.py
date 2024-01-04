import random

import tqdm
import torch
import gpytorch

import os,sys

import laspy
import datetime
import numpy_indexed as npi
import numpy_groupies as npg

import glob
import pandas as pd
import numpy as np
import argparse
import commentjson

parser = argparse.ArgumentParser(description="Parsing the main command options")
parser.add_argument("--mode", default="train", type=str, nargs="+", help="Choose your mode: train/test/application")
# parser.add_argument("--mode", default="test", type=str, nargs="+", help="Choose your mode: train/test/application")
# parser.add_argument("--mode", default="apply", type=str, nargs="+", help="Choose your mode: train/test/application/inspect")
parser.add_argument("--config_dir", default="../configs/", type=str, nargs="+", help="The folder of your configuration json file")
parser.add_argument("--config_file", default="config.json", type=str, nargs="+", help="The name of your configuration json file")
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

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,model_best_save_path,pretrained=True):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
            num_dims=2, grid_size=100
        )
        self.linear1=torch.nn.Linear(train_x.size(-1), 2048)
        self.relu1=torch.nn.ReLU()
        self.maxpool1=torch.nn.MaxPool1d(kernel_size=2,stride=2,padding=0)
        self.linear2=torch.nn.Linear(1024, 512)
        self.relu2=torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=2,stride=2,padding=0)
        self.linear3=torch.nn.Linear(256, 128)
        self.relu3=torch.nn.ReLU()
        self.maxpool3 = torch.nn.MaxPool1d(kernel_size=2,stride=2,padding=0)
        self.linear6=torch.nn.Linear(64, 2)
        self.dropout=torch.nn.Dropout()
        # self.feature_extractor = feature_extractor

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        if pretrained:
            state_dict = torch.load(model_best_save_path)
            self.best_mRMSE = state_dict['best_rmse']
            state_dict.pop('best_rmse', None)
            self.load_state_dict(state_dict)
        else:
            self.best_rmse = torch.scalar_tensor(9999.0)

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        # projected_x = self.feature_extractor(x)

        h=self.relu1(self.linear1(x))
        h=self.maxpool1(h)
        h=self.relu2(self.linear2(h))
        h=self.maxpool2(h)
        h=self.relu3(self.linear3(h))
        h=self.maxpool3(h)
        # h=self.relu4(self.linear4(h))
        # h=self.relu5(self.linear5(h))
        h=self.dropout(h)
        h=self.linear6(h)

        projected_x = self.scale_to_bounds(h)  # Make the NN values "nice"

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class DataGenerator():
    def __init__(self, data_dir, pcd_fnames, attr_ct, min_res, nbmat_sz):
        self.data_dir = data_dir
        self.pcd_fnames = pcd_fnames
        self.min_res = min_res
        self.nbmat_sz = nbmat_sz
        self.attr_ct = attr_ct
        self.nb_length = self.min_res * self.nbmat_sz
        self.nbmat_len = self.nbmat_sz[0] * self.nbmat_sz[1] * self.nbmat_sz[2]
        self.min_eres = (self.min_res[0] * self.min_res[1] * self.min_res[2]) ** (1. / 3.)

    def generate_block_input(self,cluster_sp):
        nb_ijk = np.floor((cluster_sp[:, :3] - np.min(cluster_sp[:, :3], axis=0)) / self.min_res)
        nb_within = np.all((nb_ijk < self.nbmat_sz) & (nb_ijk >= 0), axis=1)
        nb_ijk = nb_ijk[nb_within]
        nb_idx = np.ravel_multi_index(np.transpose(nb_ijk.astype(np.int32)), self.nbmat_sz)

        nb_idx_u, nb_idx_u_idx, group_idx = np.unique(nb_idx, return_index=True, return_inverse=True)

        nb_val = np.zeros(self.nbmat_len)
        # nb_val_u = npg.aggregate(group_idx, 1.0, func='sum', fill_value=0)
        # nb_val[nb_idx_u] = nb_val_u / np.sum(nb_val_u)
        # nb_val[nb_idx_u] = nb_val_u

        nb_val[nb_idx_u] = 1.0
        return nb_val
class TrainDataGenerator(DataGenerator):
    def __init__(self, data_dir,pcd_fnames, attr_ct, min_res, nbmat_sz,myLogger=None,data_augment=False):
        super(TrainDataGenerator, self).__init__(data_dir,pcd_fnames, attr_ct, min_res, nbmat_sz)
        self.data_augment = data_augment
        self.trees_all = self.read_all_data(myLogger)

    def generate_random(self, i):
        has_example=False
        while not has_example:
            has_example = self.get_random_example(i)
        return self.nb_val, self.nb_ref

    def get_random_example(self, i):
        random_rotation_angle = np.random.uniform() * 2 * np.pi
        tree_cluster = self.trees_all[i]

        center_xy = np.mean(tree_cluster[:,:2],axis=0)
        tree_clusters_sp = tree_cluster[np.all(np.abs(tree_cluster[:, :2] - center_xy[:2]) < self.nb_length[:2] * 0.5, axis=1), :]
        if len(tree_clusters_sp) == 0:
            return False
        else:
            if self.data_augment:
                rotation_center = np.mean(tree_clusters_sp[:, :3], axis=0)
                tree_clusters_sp[:, :3] = rotate_point_cloud_by_angle(tree_clusters_sp[:, :3], rotation_center,random_rotation_angle)

            self.nb_val = self.generate_block_input(tree_clusters_sp)
            self.nb_ref = tree_clusters_sp[0,-1].astype(float)

            return True
    def read_all_data(self, myLogger):
        myLogger.write(f"{datetime.datetime.now()} ---> Start reading the training dataset.\n")

        trees_all = []

        for fid, pcd_fname in enumerate(self.pcd_fnames):
            # columns = pd.read_csv(os.path.join(data_dir, pcd_fname + ".txt"), sep=' ', header=headers).values
            las = laspy.open(glob.glob(os.path.join(self.data_dir, pcd_fname + ".la*"))[0]).read()
            assert hasattr(las, 'AttrRef'), "las file should have a scalar field called 'AttrRef', indicating the reference tree attribute for regression"
            assert hasattr(las, 'itc_ref'), "las file should have a scalar field called 'itc_ref' as the reference tree ID"
            # columns = np.transpose(np.array([las.x, las.y, las.z, las.intensity,las.number_of_returns,las.VegCls]))  #las.point_format.dimension_names, las.Veg is the reference class number(other:1,tree:2)
            columns = np.transpose(np.array([las.x, las.y, las.z, las.itc_ref,las.AttrRef]))  # las.point_format.dimension_names

            abg_ind = columns[:, -2] > 0
            pcd_min = np.min(columns[:, :3], axis=0)

            columns = columns[abg_ind]
            columns[:, :3] = columns[:, :3] - pcd_min

            tree_clusters, tree_clusters_idx_groups = npi.group_by(columns[:, -2], np.arange(len(columns)))
            for tree_clusters_idx in tree_clusters_idx_groups:
                trees_all.append(columns[tree_clusters_idx, :])

        myLogger.write(f"{datetime.datetime.now()} ---> Successfully read {len(self.pcd_fnames)} files in total.\n")
        myLogger.flush()
        return trees_all


class ValidDataGenerator(DataGenerator):
    def __init__(self, data_dir, pcd_fnames, attr_ct, min_res, nbmat_sz, myLogger=None):
        super(ValidDataGenerator, self).__init__(data_dir, pcd_fnames, attr_ct, min_res, nbmat_sz)
        self.columns_all = self.read_all_data(myLogger)
    def generate_regular(self,i):
        tree_clusters = self.columns_all[i]
        nb_vals=[]
        nb_refs=[]
        for tree_cluster in tree_clusters:
            center_xy = np.mean(tree_cluster[:, :2], axis=0)
            tree_clusters_sp = tree_cluster[np.all(np.abs(tree_cluster[:, :2] - center_xy[:2]) < self.nb_length[:2] * 0.5, axis=1), :]

            nb_vals.append(self.generate_block_input(tree_clusters_sp))
            nb_refs.append((tree_clusters_sp[0,-1]).astype(float))
        nb_vals = np.vstack(nb_vals)
        nb_refs = np.array(nb_refs)
        return nb_vals,nb_refs

    def read_all_data(self,myLogger):
        myLogger.write(f"{datetime.datetime.now()} ---> Start reading the training dataset.\n")
        columns_all = []

        for fid, pcd_fname in enumerate(self.pcd_fnames):
            # columns = pd.read_csv(os.path.join(data_dir, pcd_fname + ".txt"), sep=' ', header=headers).values
            las = laspy.open(glob.glob(os.path.join(self.data_dir, pcd_fname + ".la*"))[0]).read()
            assert hasattr(las, 'AttrRef'), "las file should have a scalar field called 'AttrRef', indicating the reference tree attribute for regression"
            assert hasattr(las, 'itc_ref'), "las file should have a scalar field called 'itc_ref' as the reference tree ID"
            columns = np.transpose(np.array([las.x, las.y, las.z, las.itc_ref,las.AttrRef]))  # las.point_format.dimension_names

            abg_ind = columns[:, -2] > 0
            pcd_min = np.min(columns[:, :3], axis=0)

            columns = columns[abg_ind]
            columns[:, :3] = columns[:, :3] - pcd_min

            trees_all=[]
            tree_clusters, tree_clusters_idx_groups = npi.group_by(columns[:, -2], np.arange(len(columns)))
            for tree_clusters_idx in tree_clusters_idx_groups:
                trees_all.append(columns[tree_clusters_idx, :])
            columns_all.append(trees_all)

        myLogger.write(f"{datetime.datetime.now()} ---> Successfully read {len(self.pcd_fnames)} files in total.\n")
        myLogger.flush()
        return columns_all


class AppyDataGenerator(DataGenerator):
    def __init__(self, data_dir, pcd_fnames, attr_ct, min_res, nbmat_sz):
        super(AppyDataGenerator, self).__init__(data_dir, pcd_fnames, attr_ct, min_res, nbmat_sz)
        self.columns_all, self.columns_len_all,self.tree_clusters_idx_groups_all = self.read_all_data()
    def generate_regular(self,i):
        nb_vals=[]
        columns = self.columns_all[i]

        for tree_cluster in columns:
            center_xy = np.mean(tree_cluster[:, :2], axis=0)
            tree_clusters_sp = tree_cluster[np.all(np.abs(tree_cluster[:, :2] - center_xy[:2]) < self.nb_length[:2] * 0.5, axis=1), :]
            nb_vals.append(self.generate_block_input(tree_clusters_sp))
        nb_vals=np.vstack(nb_vals)
        return nb_vals

    def read_all_data(self):
        columns_all = []
        columns_len_all = []
        tree_clusters_idx_groups_all = []

        for fid, pcd_fname in enumerate(self.pcd_fnames):
            # columns = pd.read_csv(os.path.join(data_dir, pcd_fname + ".txt"), sep=' ', header=headers).values
            las = laspy.open(glob.glob(os.path.join(self.data_dir, pcd_fname + ".la*"))[0]).read()
            assert hasattr(las, 'segs'), "las file should have a scalar field called 'segs' as each tree ID"
            columns = np.transpose(np.array([las.x, las.y, las.z, las.segs]))  # las.point_format.dimension_names

            abg_ind = columns[:, -1] > 0
            pcd_min = np.min(columns[:, :3], axis=0)
            columns = columns[abg_ind]

            columns[:, :3] = columns[:, :3] - pcd_min
            columns_len_all.append(columns.shape[0])

            trees_all = []
            tree_clusters, tree_clusters_idx_groups = npi.group_by(columns[:, -1], np.arange(len(columns)))
            for tree_clusters_idx in tree_clusters_idx_groups:
                trees_all.append(columns[tree_clusters_idx, :])
            columns_all.append(trees_all)
            tree_clusters_idx_groups_all.append(tree_clusters_idx_groups)
        return columns_all,columns_len_all,tree_clusters_idx_groups_all

def train(configs):
    data_dir = configs["dataset"]["data_path"]
    logs_dir = configs["training"]["output_folder"]
    nbmat_sz = np.array(configs["model"]["voxel_number_in_block"])
    min_res = np.array(configs["model"]["voxel_resolution_in_meter"])

    attribute_list = configs["model"]["attribute_list"]
    attr_ct = len(attribute_list)

    num_augment_samples=configs["training"]["num_samples"]
    num_iterations = configs["training"]["iterations"]
    valid_iterations = configs["training"]["valid_iterations"]
    if_data_augment = configs["training"]["data_augment"]

    train_list = configs["dataset"]["training_files"]
    with open(os.path.join(args.config_dir, train_list), 'r') as prep_file:
        pcd_fnames = prep_file.read().split('\n')
    train_pcd_fnames = [pcd_fname for pcd_fname in pcd_fnames if pcd_fname]
    valid_list = configs["dataset"]["validating_files"]
    with open(os.path.join(args.config_dir, valid_list), 'r') as prep_file:
        pcd_fnames = prep_file.read().split('\n')
    valid_pcd_fnames = [pcd_fname for pcd_fname in pcd_fnames if pcd_fname]

    num_training_files=len(train_pcd_fnames)
    num_valid_files=len(valid_pcd_fnames)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)

    myLogger = Logger(os.path.join(logs_dir, "train_logs.txt"), 'w')
    myLogger.write(f"{datetime.datetime.now()} ---> Start training process...\n")

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    trainDataGenerator=TrainDataGenerator(data_dir,pcd_fnames,attr_ct, min_res, nbmat_sz,myLogger,data_augment=if_data_augment)
    validDataGenerator=ValidDataGenerator(data_dir,pcd_fnames,attr_ct, min_res, nbmat_sz,myLogger)

    myLogger.write(f"Training for tree attribute regression...\n")

    model_best_save_name = f'best_model.pth'
    model_best_save_path = os.path.join(logs_dir, model_best_save_name)
    if os.path.isfile(model_best_save_path):
        myLogger.write(f"Load model from {model_best_save_path}\n")
        pretrained = True
    else:
        pretrained = False

    train_x=[]
    train_y=[]

    #bootstrap sampling
    for i in range(num_augment_samples):
        each_x,each_y=trainDataGenerator.generate_random(random.randrange(0,num_training_files))
        train_x.append(each_x)
        train_y.append(each_y)

    train_x = torch.Tensor(np.vstack(train_x))
    train_y = torch.Tensor(np.stack(train_y))

    valid_x = []
    valid_y = []
    for i in range(num_valid_files):
        each_x,each_y=validDataGenerator.generate_regular(i)
        valid_x.append(each_x)
        valid_y.append(each_y)
    valid_x = torch.Tensor(np.vstack(valid_x))
    valid_y = torch.Tensor(np.concatenate(valid_y))

    if torch.cuda.is_available():
        train_x, train_y = train_x.cuda(), train_y.cuda()
        valid_x, valid_y = valid_x.cuda(), valid_y.cuda()

    model = GPRegressionModel(train_x, train_y, likelihood, model_best_save_path, pretrained)

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    model.train()
    likelihood.train()

    model.to('cuda')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    if pretrained:
        best_rmse = model.best_mRMSE
    else:
        best_rmse = 9999.0
    iterator = tqdm.tqdm(range(num_iterations))
    for i in iterator:
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(train_x)
        # Calc loss and backprop derivatives
        loss = -mll(output, train_y)
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()

        if (i % valid_iterations) == 0:
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
                valid_xx = torch.Tensor(valid_x).cuda()
                valid_yy = torch.Tensor(valid_y).cuda()
                preds = model(valid_xx)
                # rmse = torch.mean(torch.pow(predictive_means.mean(0) - test_y, 2)).sqrt()
                # print(f"RMSE: {rmse.item()}, NLL: {-test_lls.mean().item()}")
                pred = preds.mean.cpu().detach().numpy()
                ref = valid_yy.cpu().detach().numpy()
                rmse = np.sqrt(np.sum((pred - ref) * (pred - ref)) / pred.shape[0])
                rmse_rel = rmse / np.mean(ref)*100
                r2 = (np.corrcoef(ref, pred)[0, 1]) ** 2
                if rmse < best_rmse:
                    myLogger.write(f'Iteration: {i}\tBest Test mRMSE: {rmse}\tBest Test mRMSE%: {rmse_rel}\tBest Test mR2: {r2}\n')
                    myLogger.flush()
                    paras = model.state_dict()
                    paras.update({'best_rmse': torch.tensor(rmse)})
                    torch.save(paras, model_best_save_path)
                    best_rmse = rmse
            model.train()
            likelihood.train()
            # if r2 > 0.992:
            #     break
    myLogger.close()


def test(configs):
    train_data_dir = configs["dataset"]["data_path"]
    data_dir = configs["dataset"]["data_test_path"]
    out_dir = configs["testing"]["output_folder"]
    logs_dir = configs["training"]["output_folder"]

    nbmat_sz = np.array(configs["model"]["voxel_number_in_block"])
    min_res = np.array(configs["model"]["voxel_resolution_in_meter"])

    attribute_list = configs["model"]["attribute_list"]
    attr_ct = len(attribute_list)

    train_list = configs["dataset"]["training_files"]
    with open(os.path.join(args.config_dir, train_list), 'r') as prep_file:
        pcd_fnames = prep_file.read().split('\n')
    train_pcd_fnames = [pcd_fname for pcd_fname in pcd_fnames if pcd_fname]
    num_training_files = len(train_pcd_fnames)

    test_list = configs["dataset"]["testing_files"]
    with open(os.path.join(args.config_dir, test_list), 'r') as prep_file:
        pcd_fnames = prep_file.read().split('\n')
    test_pcd_fnames = [pcd_fname for pcd_fname in pcd_fnames if pcd_fname]

    if not os.path.exists(logs_dir):
        print("Not existing log folder that stores trained model")
        return

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    myLogger = Logger(os.path.join(out_dir, "test_logs.txt"), 'w')
    myLogger.write(f"{datetime.datetime.now()} ---> Start testing process...\n")

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    trainDataGenerator=TrainDataGenerator(train_data_dir,train_pcd_fnames, attr_ct, min_res, nbmat_sz,myLogger)

    num_augment_samples = configs["training"]["num_samples"]

    train_x = []
    train_y = []

    for i in range(num_augment_samples):
        each_x, each_y = trainDataGenerator.generate_random(random.randrange(0, num_training_files))
        train_x.append(each_x)
        train_y.append(each_y)

    train_x = torch.Tensor(np.vstack(train_x))
    train_y = torch.Tensor(np.array(train_y))
    if torch.cuda.is_available():
        train_x, train_y = train_x.cuda(), train_y.cuda()

    model_best_save_name = f'best_model.pth'
    model_best_save_path = os.path.join(logs_dir, model_best_save_name)
    if os.path.isfile(model_best_save_path):
        myLogger.write(f"Load model from {model_best_save_path}\n")
    else:
        return
    model=GPRegressionModel(train_x, train_y, likelihood, model_best_save_path, True)
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    model.to('cuda')
    model.eval()
    likelihood.eval()

    testDataGenerator = ValidDataGenerator(data_dir, test_pcd_fnames, attr_ct, min_res, nbmat_sz, myLogger)
    with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
        for i,test_pcd_fname in enumerate(test_pcd_fnames):
            myLogger.write(f"{datetime.datetime.now()} ---> Testing for {test_pcd_fname} >> ")

            nb_val,nb_ref=testDataGenerator.generate_regular(i)
            test_x = torch.Tensor(nb_val).cuda()
            test_y=nb_ref

            test_pred= model(test_x)
            test_pred = test_pred.mean.cpu().detach().numpy()

            rmse = np.sqrt(np.sum((test_pred - test_y) * (test_pred - test_y), axis=0) / test_pred.shape[0])
            rmse_rel = rmse / np.mean(test_y, axis=0) * 100
            r2 = (np.corrcoef(test_y, test_pred)[0, 1]) ** 2
            if myLogger is not None:
                myLogger.write("%s ---> Start testing %s (RMSE, RMSE%%, R2):\t" % (datetime.datetime.now(), test_pcd_fname))
                myLogger.write("%10.5f\t%10.5f\t%10.5f\n" % (rmse,rmse_rel,r2))
                myLogger.flush()

    myLogger.close()

def apply(configs):
    train_data_dir = configs["dataset"]["data_path"]
    data_dir = configs["dataset"]["data_application_path"]
    out_dir = configs["application"]["output_folder"]
    logs_dir = configs["training"]["output_folder"]

    nbmat_sz = np.array(configs["model"]["voxel_number_in_block"])
    min_res = np.array(configs["model"]["voxel_resolution_in_meter"])

    attribute_list = configs["model"]["attribute_list"]
    attr_ct = len(attribute_list)

    train_list = configs["dataset"]["training_files"]
    with open(os.path.join(args.config_dir, train_list), 'r') as prep_file:
        pcd_fnames = prep_file.read().split('\n')
    train_pcd_fnames = [pcd_fname for pcd_fname in pcd_fnames if pcd_fname]
    num_training_files = len(train_pcd_fnames)

    app_list = configs["dataset"]["application_files"]
    with open(os.path.join(args.config_dir, app_list), 'r') as prep_file:
        pcd_fnames = prep_file.read().split('\n')
    app_pcd_fnames = [pcd_fname for pcd_fname in pcd_fnames if pcd_fname]

    if not os.path.exists(logs_dir):
        print("Not existing log folder that stores trained model")
        return

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    myLogger = Logger(os.path.join(out_dir, "app_logs.txt"), 'w')
    myLogger.write(f"{datetime.datetime.now()} ---> Start application process...\n")

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    trainDataGenerator=TrainDataGenerator(train_data_dir, train_pcd_fnames, attr_ct, min_res, nbmat_sz, myLogger)

    num_augment_samples = configs["training"]["num_samples"]

    train_x = []
    train_y = []

    for i in range(num_augment_samples):
        each_x, each_y = trainDataGenerator.generate_random(random.randrange(0, num_training_files))
        train_x.append(each_x)
        train_y.append(each_y)

    train_x = torch.Tensor(np.vstack(train_x))
    train_y = torch.Tensor(np.array(train_y))
    if torch.cuda.is_available():
        train_x, train_y = train_x.cuda(), train_y.cuda()

        model_best_save_name = f'best_model.pth'
        model_best_save_path = os.path.join(logs_dir, model_best_save_name)
        if os.path.isfile(model_best_save_path):
            myLogger.write(f"Load model from {model_best_save_path}\n")
        else:
            return
        model=GPRegressionModel(train_x, train_y, likelihood, model_best_save_path, True)
        if torch.cuda.is_available():
            model = model.cuda()
        model.to('cuda')
        model.eval()

        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            for i,app_pcd_fname in enumerate(app_pcd_fnames):
                myLogger.write(f"{datetime.datetime.now()} ---> Applying for {app_pcd_fname} >> ")

                appDataGenerator = AppyDataGenerator(data_dir, pcd_fnames, attr_ct, min_res, nbmat_sz)
                nb_val = appDataGenerator.generate_regular(i)
                app_x = torch.Tensor(nb_val).cuda()

                pcd_ann_pred=np.zeros(appDataGenerator.columns_len_all[i])
                app_preds = model(app_x)
                app_pred = app_preds.mean.cpu().detach().numpy()
                for k in range(len(app_pred)):
                    pcd_ann_pred[appDataGenerator.tree_clusters_idx_groups_all[i][k]]=app_pred[k]

                las = laspy.open(os.path.join(data_dir, app_pcd_fname + '.laz')).read()
                las.add_extra_dim(laspy.ExtraBytesParams(name="AttrPred", type="float32", description="predicted attribute"))
                las.AttrPred[las.segs > 0] = pcd_ann_pred
                las.write(os.path.join(out_dir, "{}_attr.laz".format(app_pcd_fname)))
                if myLogger is not None:
                    myLogger.write("%s ---> Finish applying %s.\n" % (datetime.datetime.now(), app_pcd_fname))
                    myLogger.flush()


    myLogger.close()

if __name__ == '__main__':
    with open(os.path.join(args.config_dir, args.config_file)) as json_file:
        configs = commentjson.load(json_file)

    if args.mode == "train":
        train(configs)

    if args.mode == "test":
        test(configs)

    if args.mode == "apply":
        apply(configs)

