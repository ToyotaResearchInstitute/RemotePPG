# The Way to my Heart is through Contrastive Learning:<br />Remote Photoplethysmography from Unlabelled Video

This is the official project page of our ICCV 2021 conference paper.

### Abstract

The ability to reliably estimate physiological signals from video is a powerful tool in low-cost, pre-clinical health monitoring. In this work we propose a new approach to remote photoplethysmography (rPPG) – the measurement of blood volume changes from observations of a person's face or skin. Similar to current state-of-the-art methods for rPPG, we apply neural networks to learn deep representations with invariance to nuisance image variation. In contrast to such methods, we employ a fully self-supervised training approach, which has no reliance on expensive ground truth physiological training data. Our proposed method uses contrastive learning with a weak prior over the frequency and temporal smoothness of the target signal of interest. We evaluate our approach on four rPPG datasets, showing that comparable or better results can be achieved compared to recent supervised deep learning methods but without using any annotation. In addition, we incorporate a learned saliency resampling module into both our unsupervised approach and supervised baseline. We show that by allowing the model to learn where to sample the input image, we can reduce the need for hand-engineered features while providing some interpretability into the model's behavior and possible failure modes. We release code for our complete training and evaluation pipeline to encourage reproducible progress in this exciting new direction. In addition, we used our proposed approach as the basis of our winning entry to the ICCV 2021 Vision 4 Vitals Workshop Challenge.

### ICCV main conference paper materials

Code, paper, poster and video summary links will be uploaded here prior to the conference.

### Vision 4 Vitals workshop challenge

V4V workshop paper link will be uploaded here prior to the workshop.

### Citations

```
@inproceedings{gideon2021rppg,
  title={The Way to my Heart is through Contrastive Learning: Remote Photoplethysmography from Unlabelled Video},
  author={Gideon, John and Stent, Simon},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2021}
}
@inproceedings{gideon2021rppg,
  title={Estimating Heart Rate from Unlabelled Video},
  author={Gideon, John and Stent, Simon},
  booktitle={1st Vision for Vitals Workshop & Challenge, International Conference on Computer Vision (ICCV) Workshops},
  year={2021}
}
```
