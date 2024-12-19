import os
from osgeo import gdal

def read_img(self,filename):
 
    dataset = gdal.Open(filename)  # 打开文件
 
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_bands = dataset.RasterCount  # 波段数
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率
    im_proj = dataset.GetProjection()  # 地图投影信息，字符串表示
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
 
    del dataset   #关闭对象dataset，释放内存
    return  im_proj, im_geotrans, im_data, im_width,im_height,im_bands

def write_raster(output_path, datasets):
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_path, datasets[0].RasterXSize, datasets[0].RasterYSize, len(datasets), gdal.GDT_Float32)

    for i, dataset in enumerate(datasets):
        out_dataset.GetRasterBand(i + 1).WriteArray(dataset.ReadAsArray())
    
    out_dataset.SetGeoTransform(datasets[0].GetGeoTransform())
    out_dataset.SetProjection(datasets[0].GetProjection())
    out_dataset.FlushCache()

def main():
    root_folder = r'F:\new_dataset\1 guangzhou'
    img_folder = os.path.join(root_folder, '4 caijianTIF')
    area_folder = os.path.join(root_folder, 'area')
    areaproportion_folder = os.path.join(root_folder, 'areaproportion')
    building_folder = os.path.join(root_folder, 'building')
    densities_folder = os.path.join(root_folder, 'densities')
    height_folder = os.path.join(root_folder, 'height')

    output_folder = os.path.join(root_folder, 'output')
    os.makedirs(output_folder, exist_ok=True)

    img_files = [f for f in os.listdir(img_folder) if f.endswith('.tif')]

    for img_file in img_files:
        base_name = os.path.splitext(img_file)[0]

        img_path = os.path.join(img_folder, img_file)
        area_path = os.path.join(area_folder, f'{base_name}.tif')
        areaproportion_path = os.path.join(areaproportion_folder, f'{base_name}.tif')
        building_path = os.path.join(building_folder, f'{base_name}.tif')
        densities_path = os.path.join(densities_folder, f'{base_name}.tif')
        height_path = os.path.join(height_folder, f'{base_name}.tif')

        if not os.path.exists(area_path) or not os.path.exists(densities_path):
            print(f"Missing corresponding file for {base_name}, skipping.")
            continue

        img_dataset = read_raster(img_path)
        area_dataset = read_raster(area_path)
        areaproportion_dataset = read_raster(areaproportion_path)
        building_dataset = read_raster(building_path)
        densities_dataset = read_raster(densities_path)
        height_dataset = read_raster(height_path)

        output_path = os.path.join(output_folder, f'{base_name}.tif')

        bands = [img_dataset.GetRasterBand(i+1) for i in range(3)]
        bands.append(area_dataset.GetRasterBand(1))
        bands.append(areaproportion_dataset.GetRasterBand(1))
        bands.append(building_dataset.GetRasterBand(1))
        bands.append(densities_dataset.GetRasterBand(1))
        bands.append(height_dataset.GetRasterBand(1))

        write_raster(output_path, bands)

        print(f'Created {output_path}')

if __name__ == '__main__':
    main()
