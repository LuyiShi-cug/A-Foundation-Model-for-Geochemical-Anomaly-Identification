The implementation of each model and the code for each experiment are in the corresponding files

The following is the format of the data used:

  SpatialInput.npy(N,C,H,W):N is the number of samples, C is the number of features, H and W are the height and width, respectively
  
  GraphInput.npy(N,C):N is the number of samples, C is the number of features
  
  adj.npy(2,M):Edge table, 2 represents start and end nodes, M represents the number of edges
  
  label.npy(N,1):N is the number of samples. 1 is the deposits,0 is non-deposits, -1 is other
  
  projection.npy(str):Coordinate system information in gdal format

The stretching operation of GFM and the functions for pre-training and obtaining output are located in train_chanel_loss() and calculate_latent(), respectively
