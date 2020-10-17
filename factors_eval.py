# -*- coding: utf-8 -*-
import os
import pickle
import json
import struct

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"]="1" #"0,1,2,3"

import random as rnd
from collections import Counter, defaultdict
from implicit.nearest_neighbours import bm25_weight
from scipy import sparse
from lightfm import LightFM

import numpy as np
import implicit
import tqdm
import glob
import metrics
from multiprocessing import Pool

rnd.seed(0)

def save(keys, feats, out_fname):
    ''' Method used to save a list of element ids and the corresponing vectos
    '''
    feats = np.array(feats, dtype=np.float32)
    with open(out_fname + '.tmp', 'wb') as fout:
        fout.write(b' '.join([k.encode() for k in keys]))
        fout.write(b'\n')
        R, C = feats.shape
        fout.write(struct.pack('qq', *(R, C)))
        fout.write(feats.tostring())
    os.rename(out_fname + '.tmp', out_fname)


def load_feats(feat_fname, meta_only=False, nrz=False):
    ''' Method used to load element ids and the corresponing vectos
    '''
    with open(feat_fname, 'rb') as fin:
        keys = fin.readline().strip().split()
        R, C = struct.unpack('qq', fin.read(16))
        if meta_only:
            return keys, (R, C)
        feat = np.fromstring(fin.read(), count=R * C, dtype=np.float32)
        feat = feat.reshape((R, C))
        if nrz:
            feat = feat / np.sqrt((feat ** 2).sum(-1) + 1e-8)[..., np.newaxis]
    return keys, feat

latents_files = {'1': 7, '5': 1, '8': 1}

if __name__ == '__main__':

    N = 10
    dims = "300"
    model_folder = 'models_split'
    for split in ['train', '8', '5', '1']:
        # We first load all data for the current split
        tracks_ids = json.load(open(os.path.join(model_folder, 'track_ids_{}.json'.format(split)), 'r'))
        playlists_ids = json.load(open(os.path.join(model_folder, 'playlists_ids_{}.json'.format(split)), 'r'))

        item_features_file = os.path.join(model_folder, 'cf_item_{}_{}.feats'.format(dims, split))
        test_ids, track_orig_vects = load_feats(item_features_file)


        if split != 'train':
            pred_test_ids = []
            pred_vecs  = []
            for i in range(latents_files[split]):
                curr_test_ids, curr_orig_vecs = load_feats(os.path.join(model_folder, "test_pred_{}_{}.npy".format(split, i*50000)))
                pred_test_ids +=  curr_test_ids
                pred_vecs.append(curr_orig_vecs)
            pred_vecs = np.vstack(curr_orig_vecs)

            playlists_test = json.load(open(os.path.join(model_folder, 'test_cf_playlists_{}.json'.format(split)), 'r'))
            #playlists_test = json.load(open(os.path.join(model_folder, 'test_playlists_{}.json'.format(split)), 'r'))

            # The first 81219 items are the ones used to train the model, we want to evaluate on the rest of the items
            test_ids = test_ids[81219:]
            track_orig_vects = track_orig_vects[81219:]
        else:
            playlists_test = json.load(open(os.path.join(model_folder, 'test_playlists_{}.json'.format(split)), 'r'))
            pred_test_ids, pred_vecs = load_feats(os.path.join(model_folder, 'test_pred.npy'))

        dict_test_ids = {i.decode():1 for i in pred_test_ids}
        track_orig_vects = track_orig_vects[[i for i,x in enumerate(test_ids) if x.decode() in dict_test_ids]]
        test_ids = [x for x in test_ids if x.decode() in dict_test_ids]


        # Load the latent representations of playlists to make the predictions 
        train_features_file = os.path.join(model_folder, 'cf_playlist_{}_{}.feats'.format(dims, 'train'))
        train_playlists_ids, train_playlists_vects = load_feats(train_features_file)
        # This are the latent representations that are used to compute the Upper Bound
        user_features_file = os.path.join(model_folder, 'cf_playlist_{}_{}.feats'.format(dims, split))
        playlists_ids, playlists_vects = load_feats(user_features_file)

        print ("Start eval", split)
        print (track_orig_vects.shape, len(test_ids))

        inv_dict_id = {i:k.decode() for i,k in enumerate(test_ids)}
        inv_pred_id = {i:k.decode() for i,k in enumerate(pred_test_ids)}

        def evaluate(pos):
            if playlists_ids[pos].decode() not in playlists_test:
                return [],[],[]
            rets_pred = []
            gt = playlists_test[playlists_ids[pos].decode()]
            num_vals = len(gt)
            if num_vals ==0:
                    return [],[],[]
            y_orig = playlists_vects[pos].dot(track_orig_vects.T)
            y_pred = train_playlists_vects[pos].dot(pred_vecs.T)
            topn_orig = np.argsort(-y_orig)[:N]
            topn_pred= np.argsort(-y_pred)[:N]
            rets_orig = [int(inv_dict_id[t]) for t in topn_orig]
            rets_pred = [int(inv_pred_id[t]) for t in topn_pred]
            return rets_orig, rets_pred, gt


        ndcg_pred = []
        ndcg_orig = []
        ndcg_rnd= []
        gts = []
        res_orig = []
        res_pred = []
        res_rnd= []
        pool = Pool(40)
        for i in range(int(len(playlists_ids)/1000)):
            results = pool.map(evaluate, range(i*1000, (i+1)*1000))

            for rets_orig, rets_pred, gt in results:
                if len(gt) > 0:
                    res_orig.append(rets_orig)
                    res_pred.append(rets_pred)
                    rets_rnd = [int(tr) for tr in rnd.sample(dict_test_ids.keys(), N)]
                    res_rnd.append(rets_rnd)
                    gts.append(gt)
                    ndcg_pred.append(metrics.ndcg(gt, rets_pred, N))
                    ndcg_orig.append(metrics.ndcg(gt, rets_orig, N))
                    ndcg_rnd.append(metrics.ndcg(gt, rets_rnd, N))
        print ("MAP")
        print ("PRED MAP@",N,": ", metrics.mapk(gts, res_pred, N))
        print ("ORIG MAP@",N,": ", metrics.mapk(gts, res_orig, N))
        print ("RND MAP@", N,": ", metrics.mapk(gts, res_rnd, N))
        print ("NDCG:")
        print ("PRED: ", np.mean(ndcg_pred))
        print ("ORIG: ", np.mean(ndcg_orig))
        print ("RND: ", np.mean(ndcg_rnd))

