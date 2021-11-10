import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import VideoSentencePair
from model import Model
from config import get_config
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
import os
import logging
from utils import calculate_IoU_batch, AverageMeter
import numpy as np
import collections
import random
import argparse
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8.0, 4.0)
import sys
# print('argv list:', str(sys.argv))

class Runner:
    def __init__(self, config):
        self.opt = config
        os.environ["CUDA_VISIBLE_DEVICES"] = self.opt.devices
        self.set_seed()
        self.build_loader()
        self.build_model()
        self.build_optimizer()

    def set_seed(self, seed=0):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def build_loader(self):
        train = VideoSentencePair(feature_path=self.opt.feature_path,
                                  data_path=self.opt.train_path,
                                  max_frames_num=self.opt.max_frames_num,
                                  max_words_num=self.opt.max_words_num,
                                  window_widths=self.opt.window_widths,
                                  max_srl_num=self.opt.max_srl_num)
        self.trainloader = DataLoader(dataset=train, batch_size=self.opt.batch_size, shuffle=True,
                                      num_workers=self.opt.num_workers, drop_last=True)

        val = VideoSentencePair(feature_path=self.opt.feature_path,
                                data_path=self.opt.val_path,
                                max_frames_num=self.opt.max_frames_num,
                                max_words_num=self.opt.max_words_num,
                                window_widths=self.opt.window_widths,
                                max_srl_num=self.opt.max_srl_num)
        self.valloader = DataLoader(dataset=val, batch_size=self.opt.batch_size, shuffle=False,
                                    num_workers=self.opt.num_workers, drop_last=False)

    def build_model(self):
        self.net = Model(self.opt).cuda()
        if self.opt.model_load_path is not None:
            self.load_model()
        else:
            self.epoch = 1
            self.best_IoU3 = 0
            self.best_IoU5 = 0
            self.best_IoU7 = 0
            self.best_epoch = 0

    def build_optimizer(self):
        self.optimizer = Adam(self.net.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay)

    def save_model(self, path=None):
        if path is None:
            path = os.path.join(self.opt.model_save_path, 'model-epoch%d' % self.epoch)
        else:
            path = os.path.join(self.opt.model_save_path, path)
        state = {
            'state_dict': self.net.state_dict(),
            'epoch': self.epoch,
            'best_IoU3': self.best_IoU3,
            'best_IoU5': self.best_IoU5,
            'best_IoU7': self.best_IoU7,
            'best_epoch': self.best_epoch
        }
        torch.save(state, path)
        logging.info('model saved to %s\n' % path)

    def load_model(self):
        checkpoint = torch.load(self.opt.model_load_path)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.epoch = checkpoint['epoch'] + 1
        self.best_IoU3 = checkpoint['best_IoU3']
        self.best_IoU5 = checkpoint['best_IoU5']
        self.best_IoU7 = checkpoint['best_IoU7']
        self.best_epoch = checkpoint['best_epoch']
        logging.info('model load from %s' % self.opt.model_load_path)

    def train(self):
        if not os.path.exists(self.opt.model_save_path):
            os.makedirs(self.opt.model_save_path)

        epoch_IoU = []
        epoch_loss = []
        train_loss = []
        # self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=5e-4)
        # self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_muti=2, eta_min=5e-4)
        while self.epoch <= self.opt.epochs:
            # self.scheduler.step()
            logging.info('Start Epoch {}'.format(self.epoch))
            train_loss.append(self._train_one_epoch()['loss'].avg)
            meters = self.eval()
            if meters['IoU@0.3'].avg > self.best_IoU3:
                self.best_IoU3 = meters['IoU@0.3'].avg
            if meters['IoU@0.5'].avg > self.best_IoU5:
                self.best_epoch = self.epoch
                self.best_IoU5 = meters['IoU@0.5'].avg
            if meters['IoU@0.7'].avg > self.best_IoU7:
                self.best_IoU7 = meters['IoU@0.7'].avg
            logging.info("best IoU@0.5:{}, best epoch:{}".format(self.best_IoU5, self.best_epoch))
            logging.info("best IoU@0.3:{}".format(self.best_IoU3))
            logging.info("best IoU@0.7:{}".format(self.best_IoU7))
            epoch_IoU.append(meters['IoU@0.5'].avg)
            epoch_loss.append(meters['loss'].avg)
            self.epoch += 1
        plt.figure(1)
        plt.plot(range(len(train_loss)), train_loss, "g--", label="train_loss")
        plt.plot(range(len(epoch_IoU)), epoch_IoU, "r-", label="val_IoU@0.5")
        plt.plot(range(len(epoch_loss)), epoch_loss, "b--", label="vai_loss")
        plt.legend()
        # os.path.join(self.opt.model_save_path, 'model-epoch%d' % self.epoch)
        plt.savefig(os.path.join("./", "IOU&loss.png"))
        logging.info('Done.')

    def _train_one_epoch(self):
        self.net.train()
        meters = collections.defaultdict(lambda: AverageMeter())
        batch_num = len(self.trainloader)
        for batch_id, (vfeats, frame_mask, frame_mat, word_vecs, word_mask, word_mat, label, scores, duration, timestamps, objects, actions, srl_num) in enumerate(self.trainloader, 1): #The starting position of the subscript is 1
            self.optimizer.zero_grad()

            vfeats = vfeats.float().cuda()
            frame_mask = frame_mask.int().cuda()
            frame_mat = frame_mat.float().cuda()
            word_vecs = word_vecs.float().cuda()
            word_mask = word_mask.int().cuda()
            word_mat = word_mat.float().cuda()
            label = label.int().cuda()
            scores = scores.float().cuda()
            objects = objects.int().cuda()
            actions = actions.int().cuda()
            srl_num = srl_num.int().cuda()

            pred_box, loss = self.net(vfeats, frame_mask, frame_mat, word_vecs, word_mask, word_mat, label, scores, objects, actions, srl_num)
            loss.backward()
            self.optimizer.step()
            meters['loss'].update(loss.item())
            if batch_id % opt.display_n_batches == 0:
                logging.info(
                    "epoch:{}, batch:{}/{}, loss:{}".format(self.epoch, batch_id, batch_num, meters['loss'].avg))
        return meters
    
    def eval(self):
        self.net.eval()
        meters = collections.defaultdict(lambda: AverageMeter())
        with torch.no_grad():
            for (vfeats, frame_mask, frame_mat, word_vecs, word_mask, word_mat,
                 label, scores, duration, timestamps, objects, actions, srl_num) in self.valloader:
                vfeats = vfeats.float().cuda()
                frame_mask = frame_mask.int().cuda()
                frame_mat = frame_mat.float().cuda()
                word_vecs = word_vecs.float().cuda()
                word_mask = word_mask.int().cuda()
                word_mat = word_mat.float().cuda()
                label = label.int().cuda()
                scores = scores.float().cuda()
                objects = objects.int().cuda()
                actions = actions.int().cuda()
                srl_num = srl_num.int().cuda()

                pred_box, loss = self.net(vfeats, frame_mask, frame_mat, word_vecs, word_mask, word_mat, label, scores, objects, actions, srl_num)

                duration = duration.numpy()
                gt_starts, gt_ends = timestamps[0].numpy(), timestamps[1].numpy()

                pred_box = np.round(pred_box.cpu().numpy()).astype(np.int32)
                pred_starts, pred_ends = pred_box[:, 0], pred_box[:, 1]
                frame_mask = frame_mask.cpu().numpy()
                seq_len = np.sum(frame_mask, -1)
                pred_starts[pred_starts < 0] = 0
                pred_starts = (pred_starts / seq_len) * duration
                pred_ends[pred_ends >= seq_len] = seq_len[pred_ends >= seq_len] - 1
                pred_ends = ((pred_ends + 1) / seq_len) * duration

                IoUs = calculate_IoU_batch((pred_starts, pred_ends),
                                           (gt_starts, gt_ends))
                meters['loss'].update(loss.item())
                meters['mIOU'].update(np.mean(IoUs), IoUs.shape[0])
                for i in range(1, 10, 2):
                    meters['IoU@0.%d' % i].update(np.mean(IoUs >= (i / 10)), IoUs.shape[0])

            info = ""
            for key, value in meters.items():
                info += "{}, {:.4f} | ".format(key, value.avg)
            logging.info(info)

            return meters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="User Hyper-Parameters Parser")
    parser.add_argument("--test", action="store_true", help="only test")
    parser.add_argument("--dataset", default="charades", help="dataset: charades / activitynet")
    parser.add_argument("--feature", default="i3d", help="feature: c3d / i3d / two-stream")
    parser.add_argument("--feature_path", default=None, help="video feature path")
    parser.add_argument("--train_path", default=None, help="trainset information path")
    parser.add_argument("--val_path", default=None, help="validation information path")
    parser.add_argument("--model_save_path", default="./checkpoints", help="model save path")
    parser.add_argument("--model_load_path", default=None, help="model checkpoint load path")
    parser.add_argument("--epochs", type=int, default=None, help="total train epoch number")
    parser.add_argument("--batch_size", type=int, default=None, help="batch size")
    parser.add_argument("--lr", type=float, default=None, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="weight decay")
    parser.add_argument("--dropout", type=float, default=None, help="dropout")
    parser.add_argument("--alpha", type=float, default=None, help="loss coefficient")
    parser.add_argument("--node_dim", type=int, default=None, help="node representation dimension")
    parser.add_argument("--max_frames_num", type=int, default=None, help="max frames number")
    parser.add_argument("--max_words_num", type=int, default=None, help="max words number")
    parser.add_argument("--gcn_layers_num", type=int, default=None, help="GCN layers number")
    parser.add_argument("--display_n_batches", type=int, default=200, help="loss display per n batches")
    parser.add_argument("--devices", default="1", help="cuda visible devices")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    user_args = parser.parse_args()
    opt = get_config(user_args)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    start_time = time.time()
    runner = Runner(opt)
    if user_args.test:
        runner.eval()
    else:
        runner.train()
    duration_time = time.time() - start_time
    print('duration: %f' % duration_time)