import math

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from numpy import var
from osgeo import gdal



def get_quantile_values(arr, q):
    """
    参数:
        arr (np.ndarray)
        q (int)

    返回:
        quantile_values (np.ndarray)
        category_indices (np.ndarray)
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError("输入必须是 numpy 数组")
    if arr.ndim != 1:
        raise ValueError("只支持一维数组")

    # 计算分位点
    quantiles = np.quantile(arr, q=np.linspace(0, 1, q + 1))

    # 使用 searchsorted 找出每个值落在哪个分位区间中
    bin_indices = np.digitize(arr, bins=quantiles) - 1  # 区间从 0 开始编号

    # 边界处理：将最大值归到最后一个区间
    bin_indices = np.clip(bin_indices, 0, q - 1)

    # 对应每个元素的分位值为该元素所在区间的下边界
    quantile_values = quantiles[bin_indices]

    return quantile_values, bin_indices

# 将预测结果写为tif文件，要求输入模型输出结果和需要保存的tif文件名
def Predict(Hidden, XY, TifName):
    # 加载投影文件
    Projection = str(np.load(r"your_project_path.npy"))
    # 制作预测结果的二维数据
    XDistance = abs(XY[0, 0] - XY[:, 0])
    YDistance = abs(XY[0, 1] - XY[:, 1])
    ColumGap = np.min(XDistance[np.where(XDistance != 0)])
    IndexGap = np.min(YDistance[np.where(YDistance != 0)])
    Column = np.arange(XY[:, 0].min(), XY[:, 0].max() + ColumGap, ColumGap)
    Index = np.arange(XY[:, 1].max(), XY[:, 1].min() - IndexGap, -IndexGap)
    result = np.zeros([len(Index), len(Column)])
    result[:, :] = -99
    for i in range(len(Hidden)):
        result[
            np.where(np.round(Index) == np.round(XY[i, 1]))[0], np.where(np.round(Column) == np.round(XY[i, 0]))[0]] = \
            Hidden[i]

    # 把数据输出成tif
    newdata = pd.DataFrame(result, index=Index, columns=Column)
    var_lon = newdata.columns.map(float)
    var_lon = var_lon.astype(np.float64)
    var_lat = newdata.index
    data_arr = np.asarray(newdata)
    LonMin, LatMax, LonMax, LatMin = [var_lon.min(), var_lat.max(), var_lon.max(), var_lat.min()]
    N_Lat = len(var_lat)
    N_Lon = len(var_lon)
    Lon_Res = (LonMax - LonMin) / (float(N_Lon) - 1)
    Lat_Res = (LatMax - LatMin) / (float(N_Lat) - 1)
    driver = gdal.GetDriverByName('GTiff')
    out_tif_name = TifName
    out_tif = driver.Create(out_tif_name, N_Lon, N_Lat, 1, gdal.GDT_Float32)  # 创建框架
    geotransform = (LonMin, Lon_Res, 0, LatMax, 0, -Lat_Res)
    out_tif.SetGeoTransform(geotransform)

    out_tif.SetProjection(Projection)
    out_tif.GetRasterBand(1).SetNoDataValue(-99)
    # 将数据写入内存，此时没有写入硬盘
    out_tif.GetRasterBand(1).WriteArray(data_arr)
    # 将数据写入硬盘
    out_tif.FlushCache()
    out_tif = None  # 注意必须关闭tif文件
    print("运行结束")