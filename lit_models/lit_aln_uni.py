import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns

import vsum_tools

from torch.utils.data import DataLoader
from scipy.stats import kendalltau, spearmanr
from scipy.stats import rankdata
from pytorch_lightning.utilities.distributed import rank_zero_only
from scipy.ndimage import gaussian_filter1d
from einops import rearrange

from datasets import DPLDatasetRand
from utils import *
from attention import TransformerEncoder
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

class LitModel(pl.LightningModule):
    def __init__(self, cfg, hpms):
        super().__init__()
        self.is_raw = cfg.is_raw
        self.use_unq = cfg.use_unq
        self.use_unif = cfg.use_unif

        self.model_cfg = cfg.model
        self.data_cfg = cfg.data
        self.hpms = hpms 
        self.setup_cfg = cfg.setup
        self.lit_cfg = cfg.lightning
        self.model = TransformerEncoder(n_layers=self.hpms.n_layer, n_head=self.hpms.n_head, num_patches=self.hpms.num_frames)
        self.bce = nn.BCELoss()
        
    def forward(self,x):
        out, scores = self.model(x)
        return out, scores

    def get_values(self, feats, proj, scores, train=True):
        assert (len(feats.shape) == 3) and (len(proj.shape) == 3)
        with torch.no_grad():
            norm_raw = F.normalize(feats, p=2, dim=-1)
            xy_raw = torch.einsum('bmc, bnc -> bmn', norm_raw, norm_raw)
        norm_proj = F.normalize(proj, p=2, dim=-1)
        xy = torch.einsum('bmc, bnc -> bmn', norm_proj, norm_proj)
        sort_ids = torch.argsort(xy_raw, -1, descending=True)

        diff_mat = 2 - 2 * xy
        L = feats.shape[1]
        S = int(L * self.hpms.ratio_s) 
        K1 = int(L * self.hpms.ratio_k1)
    
        pos = torch.gather(diff_mat, -1, sort_ids[:,:,S:S+K1])

        laln = pos.mean(dim=-1)
        
        lunif = diff_mat.mul(-2).exp().mean(dim=-1).log()

        if train:
            s = 20
            seg = rearrange(proj, 'b (s l) c -> (b s) l c', s=s)
            seg_feats = F.normalize(seg.mean(dim=1), dim=1) # (b s) c
            fv_xy = torch.einsum("bmc, nc -> bmn", norm_proj, seg_feats) # b m (b s)
            
            mask = torch.ones_like(fv_xy)
            # for i in range(len(mask)):
            #     mask[i,:,i * s : (i+1) * s] = 0
            lunif_fv = (fv_xy.mul(4).exp() * mask).sum(dim=-1) / mask.sum(dim=-1)
            lunif_fv = lunif_fv.log()
            unq_target = (lunif_fv - lunif_fv.min(dim=-1, keepdim=True)[0]) / (lunif_fv.max(dim=-1, keepdim=True)[0] - lunif_fv.min(dim=-1, keepdim=True)[0]).add(1e-9)
            unq_target = unq_target * 0.5 + 0.25
            lunq = self.bce(scores, 1 - unq_target.detach())
            return laln, lunif, lunif_fv, lunq
        else: 
            return laln, lunif

    def training_step(self, batch, batch_idx):
        out, scores = self(batch)

        laln, lunif, lunif_fv, lunq = self.get_values(batch, out, scores)

        if self.use_unq:
            loss = laln.mean() + self.hpms.alpha * lunif.mean()  + 0.1 * lunif_fv.mean() + 0.1 * lunq
        else: 
            loss = laln.mean() + self.hpms.alpha * lunif.mean()

        return loss

    def on_test_start(self):
        self.fms = []
        self.pres = []
        self.recs = []
        self.tau_list = []
        self.r_list = []
        self.eval_metric = 'avg' if self.data_cfg.name == 'tvsum' else 'max'
        # if self.setup_cfg.save_h5:
        #     experiment = '{}_split_{}.h5'.format(self.data_cfg.name, self.data_cfg.split) 
        #     self.h5_res = h5py.File(osp.join(self.setup_cfg.logdir,experiment), 'w')
    def test_step(self,batch, batch_ids, eps=0.01):
        video = batch
        # video_name = str(video['video_name'][...].astype('U8'))
        cps = video['change_points'][...]
        video_feats = torch.Tensor(video['features'][...]).to(self.device)

        feats = video_feats.unsqueeze(0)
        # proj = self(feats.transpose(2,1))
        proj, unq_scores = self(feats)

        laln_raw, lunif_raw = self.get_values(feats, feats, unq_scores, train=False)
        laln_raw = laln_raw.flatten().cpu()
        lunif_raw = lunif_raw.flatten().cpu()

        laln_raw = (laln_raw - laln_raw.min()) / (laln_raw.max() - laln_raw.min())
        lunif_raw = (lunif_raw - lunif_raw.min()) / (lunif_raw.max() - lunif_raw.min())

        laln, lunif = self.get_values(feats, proj, unq_scores, train=False)
        laln = laln.flatten().cpu()
        lunif = lunif.flatten().cpu()

        laln = (laln - laln.min()) / (laln.max() - laln.min())
        lunif = (lunif - lunif.min()) / (lunif.max() - lunif.min())

        unq_scores = unq_scores.cpu().flatten()
        unq_scores = (unq_scores - unq_scores.min()) / (unq_scores.max() - unq_scores.min())
        if not self.is_raw:
            if self.use_unif:
                scores = laln * lunif
            else:
                scores = laln

            if self.use_unq:
                scores = scores * unq_scores
        else:
            if self.use_unif:
                scores = laln_raw * lunif_raw
            else: 
                scores = laln_raw
        if self.data_cfg.name == 'tvsum': 
            scores = np.exp(scores - 1) 
        elif self.data_cfg.name == 'summe':  
            scores = scores + 0.05
        else: 
            raise NotImplementedError

        scores = gaussian_filter1d(scores, 1)

        if self.data_cfg.name == 'tvsum':
            user_scores = video['user_scores'][...]
            num_users = user_scores.shape[0]
            tau = 0.
            r = 0.
            for us in user_scores:
                tau += kendall(us,scores)[0]
                r += spearman(us,scores)[0]
            self.tau_list.append(tau/num_users)
            self.r_list.append(r/num_users)
        else:
            user_scores = video['user_summary'][...]
            num_users = user_scores.shape[0]
            tau = 0.
            r = 0.
            for us in user_scores:
                us = us[::15]
                tau += kendall(us,scores)[0]
                r += spearman(us,scores)[0]
            self.tau_list.append(tau/num_users)
            self.r_list.append(r/num_users)

        num_frames = video['n_frames'][()]

        nfps = video['n_frame_per_seg'][...].tolist()
        positions = video['picks'][...]
        user_summary = video['user_summary'][...]

        machine_summary = vsum_tools.generate_summary(scores, cps, num_frames, nfps, positions)
        fm, pre, rec,machine_summary,user_summary = vsum_tools.evaluate_summary(machine_summary, user_summary, self.eval_metric)
        self.fms.append(fm)
        self.pres.append(pre)
        self.recs.append(rec)    

    @rank_zero_only
    def test_epoch_end(self, outputs):
        mean_tau = torch.mean(torch.tensor(self.all_gather(self.tau_list)))
        mean_r = torch.mean(torch.tensor(self.all_gather(self.r_list)))
        # if self.setup_cfg.save_h5:
        #     self.h5_res.close()
        mean_fm = torch.mean(torch.tensor(self.all_gather(self.fms)))
        mean_pre = torch.mean(torch.tensor(self.all_gather(self.pres)))
        mean_rec = torch.mean(torch.tensor(self.all_gather(self.recs)))

        self.log_dict({"F1": mean_fm,
                "tau": mean_tau,
                "rho": mean_r})

    def train_dataloader(self):
        dataset = DPLDatasetRand(self.data_cfg,mode='train')
        print(len(dataset))
        dataloader = DataLoader(dataset,
                                num_workers=self.data_cfg.num_workers,
                                batch_size=self.hpms.batch_size, 
                                shuffle=True,
                                pin_memory=True)
        self.dataloader_len = len(dataloader)
        print("Number of training videos: {}".format(len(dataset)))
        return dataloader

    def test_dataloader(self):
        dataset = DPLDatasetRand(self.data_cfg, mode='test')
        dataloader = DataLoader(dataset,
                                num_workers=0,
                                batch_size=1, 
                                shuffle=False,
                                collate_fn=test_collate,
                                pin_memory=True)
        print("Number of test videos: {}".format(len(dataset)))
        return dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.model.parameters()),
                                    lr=self.hpms.lr, 
                                    weight_decay=self.hpms.weight_decay)
        return optimizer

    @torch.no_grad()
    def log_hist(self,name,inp_):
        inp = inp_.detach().cpu().flatten()
        fig = plt.figure()
        ax = fig.gca()
        sns.histplot(inp,bins=50)
        plt.tight_layout()
        
        add_plot(self.logger.experiment, name, self.global_step)