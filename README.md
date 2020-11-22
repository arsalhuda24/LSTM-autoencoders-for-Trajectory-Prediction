# CNN, Segmentation or Semantic Embedding- Evaluating Scene Context for Trajectory Prediction


## Summary
 For autonomous vehicles (AV) and social robot’s navigation, it is important for them to completely understand their surroundings for natural and safe interactions. While it is often recognized that scene context is important for understanding pedestrian behavior, it has received less attention than modeling social-context – influence from interactions between pedestrians.  In this paper, we evaluate the effectiveness of various scene representations for deep trajectory prediction. Our work focuses on characterizing the impact of scene representations (sematic images vs. semantic embeddings) and scene quality (competing semantic segmentation networks). We leverage a hierarchical RNN autoencoder to encode historical pedestrian motion, their social interaction and scene semantics into a low dimensional subspace and then decode to generate future motion prediction. Experimental evaluation on the ETH and UCY datasets show that using full scene semantics, specifically segmented images, can improve trajectory prediction over using just embeddings. 

## Model 
### This is RNN-AE model.
![Model](https://github.com/arsalhuda24/VAE-Trajectory-Prediction/blob/master/model.png)


## Semantic segmentation comparison of Seg-Net and Psp-Net. 
![Trajectory](https://github.com/arsalhuda24/VAE-Trajectory-Prediction/blob/master/PSP-NET.png)


## Results on pedestrain trajectories 
![Trajectory](https://github.com/arsalhuda24/VAE-Trajectory-Prediction/blob/master/results.png)
#### Note 
Blue == SGAN-P
Yellow == Ours
