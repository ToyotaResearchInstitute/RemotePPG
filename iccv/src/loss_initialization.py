from src.losses.NegPearsonsCorrLoss import NegPearsonsCorrLoss
from src.losses.NegSNRLoss import NegSNRLoss
from src.losses.MultiViewTripletLoss import MultiViewTripletLoss
from src.losses.NegativeMaxCrossCorr import NegativeMaxCrossCorr
from src.losses.IrrelevantPowerRatio import IrrelevantPowerRatio


def init_loss(loss_name, device, Fs, D, args):
    if loss_name == 'L1':
        return nn.L1Loss(reduction = 'none')
    elif loss_name == 'MSE':
        return nn.MSELoss(reduction = 'none')
    elif loss_name == 'NegPC':
        return NegPearsonsCorrLoss()
    elif loss_name == 'NegSNR':
        return NegSNRLoss(Fs, args.high_pass, args.low_pass)
    elif loss_name == 'MVTL':
        return MultiViewTripletLoss(Fs, D, args.high_pass, args.low_pass, args.mvtl_distance)
    elif loss_name == 'NegMCC':
        return NegativeMaxCrossCorr(Fs, args.high_pass, args.low_pass)
    elif loss_name == 'IPR':
        return IrrelevantPowerRatio(Fs, args.high_pass, args.low_pass)
    else:
        print('\nError! No such loss function ' + loss_name)
        exit(666)
