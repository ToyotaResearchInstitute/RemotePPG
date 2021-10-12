# The Way to my Heart is through Contrastive Learning:<br />Remote Photoplethysmography from Unlabelled Video

This is the official project page of our ICCV 2021 conference paper.

### Abstract

The ability to reliably estimate physiological signals from video is a powerful tool in low-cost, pre-clinical health monitoring. In this work we propose a new approach to remote photoplethysmography (rPPG) – the measurement of blood volume changes from observations of a person's face or skin. Similar to current state-of-the-art methods for rPPG, we apply neural networks to learn deep representations with invariance to nuisance image variation. In contrast to such methods, we employ a fully self-supervised training approach, which has no reliance on expensive ground truth physiological training data. Our proposed method uses contrastive learning with a weak prior over the frequency and temporal smoothness of the target signal of interest. We evaluate our approach on four rPPG datasets, showing that comparable or better results can be achieved compared to recent supervised deep learning methods but without using any annotation. In addition, we incorporate a learned saliency resampling module into both our unsupervised approach and supervised baseline. We show that by allowing the model to learn where to sample the input image, we can reduce the need for hand-engineered features while providing some interpretability into the model's behavior and possible failure modes. We release code for our complete training and evaluation pipeline to encourage reproducible progress in this exciting new direction. In addition, we used our proposed approach as the basis of our winning entry to the ICCV 2021 Vision 4 Vitals Workshop Challenge.

### ICCV main conference paper materials

- [x] [Paper + supplementary](pdf/2020-10-01-full.pdf)
- [x] [Poster](pdf/2020-10-01-poster.pdf)
- [ ] Video (5 min summary)
- [x] Code - see [iccv/README.md](./iccv/README.md)

### Vision 4 Vitals workshop challenge

[V4V workshop paper link](https://openaccess.thecvf.com/content/ICCV2021W/V4V/papers/Gideon_Estimating_Heart_Rate_From_Unlabelled_Video_ICCVW_2021_paper.pdf)

### Citations

```
@InProceedings{Gideon_2021_ICCV,
    author    = {Gideon, John and Stent, Simon},
    title     = {The Way to My Heart Is Through Contrastive Learning: Remote Photoplethysmography From Unlabelled Video},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {3995-4004}
}
@InProceedings{Gideon_2021_ICCV,
    author    = {Gideon, John and Stent, Simon},
    title     = {Estimating Heart Rate From Unlabelled Video},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2021},
    pages     = {2743-2749}
}
```
