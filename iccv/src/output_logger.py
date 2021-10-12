import os
import csv
from collections import OrderedDict
import numpy as np
import pandas as pd
import plotly.express as px
import cv2

import torch

from src.shared.heart_rate import compute_power_spectrum


class OutputLogger:
    def __init__(self, use_wandb, output_dir):
        self.use_wandb = use_wandb
        self.output_dir = output_dir
        if not self.use_wandb:
            self.output_values_log = {}


    def output_values(self, values, step):
        if self.use_wandb:
            import wandb
            wandb.log(values, step = step)
        else:
            if step not in self.output_values_log:
                self.output_values_log[step] = {}
            for key, value in values.items():
                if torch.is_tensor(value):
                    value = value.item()
                self.output_values_log[step][key] = value


    def finalize(self):
        if not self.use_wandb:
            # Determine value log headers
            output_values_headers = set()
            for epoch_values in self.output_values_log.values():
                for key in epoch_values.keys():
                    output_values_headers.add(key)
            output_values_headers = ['epoch',] + list(output_values_headers)

            # Determine epochs logged
            epochs = list(self.output_values_log.keys())
            epochs.sort()

            # Output csv
            csv_path = os.path.join(self.output_dir, "logged_values.csv")
            with open(csv_path, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=output_values_headers)
                writer.writeheader()
                for epoch in epochs:
                    self.output_values_log[epoch]['epoch'] = epoch
                    writer.writerow(self.output_values_log[epoch])


    def save(self, path):
        if self.use_wandb:
            import wandb
            wandb.save(path)


    def output_ppg(self, sample_id, plot_gt_signals, plot_pred_signals, hr_gt, hr_our, hr_pred, epoch, phase, Fs):
        if self.use_wandb: 
            import wandb

        # Determine name
        plot_sample_id = sample_id.replace('/', '_')
        title_hr = ' (Our: {:.2f}, GT: {:.2f}, Pred: {:.2f})'.format(hr_our, hr_gt, hr_pred)
        num_frames = plot_pred_signals.shape[0]

        # Print the PPG signal
        plot_gt_signals = (plot_gt_signals - np.mean(plot_gt_signals)) / np.std(plot_gt_signals)
        plot_pred_signals = (plot_pred_signals - np.mean(plot_pred_signals)) / np.std(plot_pred_signals)
        time_values = np.arange(0, num_frames) / Fs
        time_values = np.tile(time_values, 2)
        ppg_values = np.concatenate((plot_gt_signals, plot_pred_signals))
        data_labels = (['GT',]*num_frames) + (['Pred',]*num_frames)
        df = pd.DataFrame({'time': time_values, 'ppg': ppg_values, 'label': data_labels})
        title = f'{phase}: {plot_sample_id} PPG Signal' + title_hr
        fig = px.line(df, x="time", y="ppg", color='label', title=title)
        if self.use_wandb:
            wandb.log({f"{phase}_ppg": fig}, step = epoch)
        else:
            dir_path = os.path.join(self.output_dir, 'ppg', phase)
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, str(epoch) + '.png')
            fig.write_image(file_path, engine="kaleido")

        # Print power spectra
        freqs, ps_gt = compute_power_spectrum(plot_gt_signals, Fs, zero_pad=100)
        freqs, ps_pred = compute_power_spectrum(plot_pred_signals, Fs, zero_pad=100)
        data_values = np.concatenate((ps_gt, ps_pred))
        data_labels = (['GT',]*len(ps_gt)) + (['Pred',]*len(ps_pred))
        freqs = np.tile(freqs, 2)
        df = pd.DataFrame({'BPM': freqs, 'Power': data_values, 'Label': data_labels})
        title = f'{phase}: {plot_sample_id} PPG Power' + title_hr
        fig = px.line(df, x="BPM", y="Power", color='Label', 
                    range_x=[0,240], title=title)
        if self.use_wandb:
            wandb.log({f"{phase}_ppg_power": fig}, step = epoch)
        else:
            dir_path = os.path.join(self.output_dir, 'power', phase)
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, str(epoch) + '.png')
            fig.write_image(file_path, engine="kaleido")


    def output_video(self, video_data, epoch, Fs, vid_name="val"):
        video_data *= 255
        video_data = video_data.astype(np.uint8)
        if video_data.shape[0] != 3:
            # Correct for videos with less than three channels
            video_data = video_data[0]
            video_data = np.repeat(video_data[np.newaxis], 3, axis=0)
        if self.use_wandb:
            import wandb
            video_data = np.transpose(video_data, (1,0,2,3))
            sample_video = wandb.Video(video_data, fps=Fs, format="mp4")
            wandb.log({vid_name+"_video": sample_video}, step = epoch)
        else:
            dir_path = os.path.join(self.output_dir, 'video', vid_name)
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, str(epoch) + '.mp4')
            video_data = np.transpose(video_data, (1,0,2,3))
            video_writer = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                           Fs, (video_data.shape[3], video_data.shape[2]))
            for frame_on in range(video_data.shape[0]):
                img = np.moveaxis(video_data[frame_on], 0, 2)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                video_writer.write(img)
            video_writer.release()


    def output_epoch_stats(self, stats, epoch_hr, epoch, phase):
        if self.use_wandb:
            import wandb

        # Output histogram of predictions
        counts, bins = np.histogram(epoch_hr['pred'], bins=range(0, 240+1, 5))
        bins = 0.5 * (bins[:-1] + bins[1:])
        fig = px.bar(x=bins, y=counts, labels={'x':'Model Predictions (BPM)', 'y':'Count'})
        if self.use_wandb:
            wandb.log({f"{phase}_pred_histogram": fig}, step = epoch)
        else:
            dir_path = os.path.join(self.output_dir, 'histogram', phase, 'pred')
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, str(epoch) + '.png')
            fig.write_image(file_path, engine="kaleido")

        # Loop through all possible comparisons of pred and GT
        for comparison_pair in [('our', 'gt'), ('pred', 'gt'), ('pred', 'our')]:
            comparison = comparison_pair[0] + '_v_' + comparison_pair[1]
            labels = {'our': 'Our Ground Truth',
                    'gt': 'Dataset Ground Truth',
                    'pred': 'Model Prediction'}
            translated_comparison = labels[comparison_pair[0]] + ' vs. ' + labels[comparison_pair[1]]
            
            # Output ground truth and predicted heart rate
            df = pd.DataFrame({labels[comparison_pair[0]]: epoch_hr[comparison_pair[0]],
                            labels[comparison_pair[1]]: epoch_hr[comparison_pair[1]]})
            title = '{:s}: {:s} (RMSE: {:.3f}; MAE: {:.3f}; Corr {:.3f})'.format(
                phase, translated_comparison,
                stats[f'{comparison}_rmse'], stats[f'{comparison}_mae'], stats[f'{comparison}_corr'])
            fig = px.scatter(df,
                            x=labels[comparison_pair[0]],
                            y=labels[comparison_pair[1]],
                            title=title,
                            range_x=[0,240],
                            range_y=[0,240])
            if self.use_wandb:
                wandb.log({f"{phase}_{comparison}_scatter": fig}, step = epoch)
            else:
                dir_path = os.path.join(self.output_dir, 'scatter', phase, comparison)
                os.makedirs(dir_path, exist_ok=True)
                file_path = os.path.join(dir_path, str(epoch) + '.png')
                fig.write_image(file_path, engine="kaleido")

            # Output all stats to wandb
            for key, value in stats.items():
                self.output_values({f'{phase}_{key}': value}, step = epoch)
