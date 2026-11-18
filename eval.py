import os
import cv2
from tqdm import tqdm
# from metrics import EvaluationMetricsV2
import logging
import py_sod_metrics as metrics
import numpy as np


class EvaluationMetricsV2:
    def __init__(self):
        self.SM = metrics.Smeasure()
        self.EM = metrics.Emeasure()
        self.FM = metrics.Fmeasure()
        self.WFM = metrics.WeightedFmeasure()
        self.MAE = metrics.MAE()

    def reset(self):
        self.__init__()

    def step(self, pred, gt):
        """
        pred: [0, 255]
        gt: [0, 255]
        """
        self.SM.step(pred=pred, gt=gt)
        self.EM.step(pred=pred, gt=gt)
        self.FM.step(pred=pred, gt=gt)
        self.WFM.step(pred=pred, gt=gt)
        self.MAE.step(pred=pred, gt=gt)

    def get_results(self):
        # S-measure, default alpha=0.5
        sm = self.SM.get_results()["sm"]
        # mean E-measure and E-measure Curve
        _em = self.EM.get_results()["em"]
        em_curve = np.flip(_em["curve"])
        emMean = _em['curve'].mean()
        emMax = _em['curve'].max()
        emAdp = _em['adp']
        # F-measure curve and PR-curve
        _fm = self.FM.get_results()
        # F-measure
        fm = _fm["fm"]
        fmMean = fm["curve"].mean()
        fmMax = fm["curve"].max()
        fmAdp = fm["adp"]
        fm_curve = np.flip(fm["curve"])
        pr = _fm["pr"]
        p = np.flip(pr["p"])
        r = np.flip(pr["r"])
        # weighted F-measure
        wfm = self.WFM.get_results()["wfm"]
        # mean Absolute Error
        mae = self.MAE.get_results()["mae"]
        # return sm, emMean, emAdp, wfm, mae
        return {
            'sm': sm,

            'emMean': emMean,
            'emAdp': emAdp,
            'emMax': emMax,
            'em_curve': em_curve,

            'fmMean': fmMean,
            'fmMax': fmMax,
            'fmAdp': fmAdp,
            'fm_curve': fm_curve,

            'wfm': wfm,
            'mae': mae,

            'p': p,
            'r': r
        }


def evaluate(pred_path):
    pred_root = f'{pred_path}'
    metric = EvaluationMetricsV2()
    # mask_root = '/root/autodl-tmp/EVIAC/Testdata/GT'
    mask_root = '/root/autodl-tmp/TVCO1K/test/GT'
    mask_name_list = sorted(os.listdir(pred_root))
    mask_name_list = [name for name in mask_name_list if name.endswith('.png') or name.endswith('.jpg')]

    for i, mask_name in tqdm(list(enumerate(mask_name_list))):
        pred_path = os.path.join(pred_root, mask_name)
        mask_path = os.path.join(mask_root, mask_name)
        pred = cv2.imread(pred_path, flags=cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        assert pred.shape == mask.shape, f'{pred.shape}: {pred_path}\n{mask.shape}: {mask_path}'
        metric.step(pred=pred, gt=mask)

    metric_dic = metric.get_results()
    return metric_dic


if __name__ == '__main__':
    model_name = 'Continet'
    pred_path = f'./snapshot_64_v3_2backbone/'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S %p',
                        filename= './TVCO1K_64_v3_2backbone.log',
                        filemode='w')
    # pred_path = f'/root/autodl-tmp/1ConTriNet/Predict_maps/TVCO1K/'
    # logging.basicConfig(level=logging.INFO,
    #                     format='%(asctime)s - %(levelname)s - %(message)s',
    #                     datefmt='%Y-%m-%d %H:%M:%S %p',
    #                     filename= './TVCO1K_Continet.log',
    #                     filemode='w')

    logging.info(f'Evaluating...')

    metric_dic = evaluate(pred_path)

    sm = metric_dic['sm']

    emMean = metric_dic['emMean']
    emAdp = metric_dic['emAdp']
    emMax = metric_dic['emMax']

    fmMean = metric_dic['fmMean']
    fmAdp = metric_dic['fmAdp']
    fmMax = metric_dic['fmMax']

    wfm = metric_dic['wfm']
    mae = metric_dic['mae']

    logging.info(f'sm: {sm}')
    logging.info(f'emMean: {emMean}')
    logging.info(f'emAdp: {emAdp}')
    logging.info(f'emMax: {emMax}')
    logging.info(f'fmMean: {fmMean}')
    logging.info(f'fmAdp: {fmAdp}')
    logging.info(f'fmMax: {fmMax}')
    logging.info(f'wfm: {wfm}')
    logging.info(f'mae: {mae}')