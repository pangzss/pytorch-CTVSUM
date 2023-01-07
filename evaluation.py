import os.path as osp
import numpy as np
import torch

import h5py
import json
from scipy.stats import kendalltau, spearmanr
from scipy.stats import rankdata

import vsum_tools

def get_rc_func(metric):
    if metric == 'kendalltau':
        f = lambda x, y: kendalltau(rankdata(-x), rankdata(-y))
    elif metric == 'spearmanr':
        f = lambda x, y: spearmanr(x, y)
    else:
        raise RuntimeError
    return f

kendall = get_rc_func('kendalltau')
spearman = get_rc_func('spearmanr')

def evaluate(model, test_loader, dataset, split, log_dir, save_h5=False):
    print("==> Test")
    with torch.no_grad():
        model.eval()
        fms = []
        eval_metric = 'avg' if dataset == 'tvsum' else 'max'
        if save_h5:
            experiment = '{}_split_{}.h5'.format(dataset, split) 
            h5_res = h5py.File(osp.join(log_dir,experiment), 'w')

        tau_list = []
        r_list = []
        for data in test_loader:
            video = data
            video_name = str(data['video_name'][...].astype('U8'))
            cps = video['change_points'][...]
            video_feats = torch.Tensor(video['features'][...])
            video_feats = video_feats.to(model.device)
    
            _, sum_attns = model(video_feats.unsqueeze(0))
            scores = torch.mean(sum_attns[0],dim=0).cpu().numpy()

            if vars['dataset'] == 'tvsum':
                user_scores = video['user_scores'][...]
                num_users = user_scores.shape[0]
                tau = 0.
                r = 0.
                for us in user_scores:
                    tau += kendall(us,scores)[0]
                    r += spearman(us,scores)[0]
                tau_list.append(tau/num_users)
                r_list.append(r/num_users)
            else:
                user_scores = video['user_summary'][...]
                num_users = user_scores.shape[0]
                tau = 0.
                r = 0.
                for us in user_scores:
                    us = us[::15]
                    tau += kendall(us,scores)[0]
                    r += spearman(us,scores)[0]
                tau_list.append(tau/num_users)
                r_list.append(r/num_users)

            num_frames = video['n_frames'][()]

            nfps = video['n_frame_per_seg'][...].tolist()
            positions = video['picks'][...]
            user_summary = video['user_summary'][...]

            machine_summary = vsum_tools.generate_summary(scores, cps, num_frames, nfps, positions)
            fm, _, _,machine_summary,user_summary = vsum_tools.evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)
            
            if save_h5:
                h5_res.create_dataset(video_name + '/score', data=scores)
                h5_res.create_dataset(video_name + '/machine_summary', data=machine_summary)
                h5_res.create_dataset(video_name + '/gtscore', data=video['gtscore'][...])
                h5_res.create_dataset(video_name + '/fm', data=fm)
                h5_res.create_dataset(video_name + '/tau', data=tau_list[-1])
                h5_res.create_dataset(video_name + '/r', data=r_list[-1])
                h5_res.create_dataset(video_name + '/user_summary', data=user_summary)
                h5_res.create_dataset(video_name + '/change_points', data=cps)
    
    mean_tau = np.mean(tau_list)
    mean_r = np.mean(r_list)
    if save_h5:
        h5_res.close()
    mean_fm = np.mean(fms)
    print("Average F-score {:.1%}. Tau {:.5f}, R {:.5f}".format(mean_fm,mean_tau,mean_r))

    return mean_fm,mean_tau, mean_r
