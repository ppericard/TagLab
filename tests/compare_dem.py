"""Measure GDAL default slope/unit drift through TagLab's surface-area path."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import site
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
parser=argparse.ArgumentParser()
parser.add_argument('--report',required=True)
parser.add_argument('--compare',nargs=2)
parser.add_argument('--figure')
args=parser.parse_args()
assert os.environ.get('CONDA_PREFIX') and not site.ENABLE_USER_SITE
import numpy as np
if args.compare:
    reports=[json.loads(Path(path).read_text(encoding='utf-8')) for path in args.compare]
    assert reports[0]['official_dem_sha256']==reports[1]['official_dem_sha256']
    rows=[]
    for before,after in zip(reports[0]['fixtures'],reports[1]['fixtures']):
        assert before['name']==after['name'] and before['metadata']==after['metadata']
        row={'name':before['name'],'slope_311':before['median_slope_default'],
             'slope_314':after['median_slope_default'],'surface_factor_311':before['taglab_surface_factor'],
             'surface_factor_314':after['taglab_surface_factor'],
             'surface_change_percent':100*(after['taglab_surface_factor']/before['taglab_surface_factor']-1),
             'scale1_reproduces_legacy_slope':abs(after['median_slope_explicit_scale1']-before['median_slope_default'])<1e-6,
             'expected_surface_factor':after.get('expected_surface_factor')}
        rows.append(row)
    output={'baseline':args.compare[0],'candidate':args.compare[1],'matched_dem_and_metadata':True,'comparisons':rows}
    Path(args.report).write_text(json.dumps(output,indent=2),encoding='utf-8')
    if args.figure:
        import matplotlib.pyplot as plt
        chosen=rows[:5]
        labels=['Metres / metres','WGS84 equateur / metres','WGS84 latitude 24 / metres',
                'Pieds US / metres','Metres / pieds']
        y=np.arange(len(chosen));h=.35
        fig,ax=plt.subplots(figsize=(10,5))
        ax.barh(y-h/2,[row['surface_factor_311'] for row in chosen],height=h,label='Python 3.11 / GDAL 3.9.2',color='#5b77b1')
        ax.barh(y+h/2,[row['surface_factor_314'] for row in chosen],height=h,label='Python 3.14 / GDAL 3.12.2',color='#359a88')
        ax.axvline(math.sqrt(1.25),linestyle='--',color='#333',label='Reference analytique : 1.118034')
        ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,2.15)
        ax.set_xlabel('Surface calculee / surface plane')
        ax.set_title('Pente 26.565 deg : effet de la conversion des unites GDAL')
        ax.legend(loc='lower right',fontsize=9)
        fig.tight_layout();fig.savefig(args.figure,dpi=160);plt.close(fig)
    print(json.dumps(output,indent=2))
    sys.exit(0)
from osgeo import gdal,osr
from skimage.measure import regionprops
from source.Blob import Blob
from source.RasterOps import exportSlope,calculateAreaUsingSlope

gdal.UseExceptions()
report={'python':sys.version,'versions':{name:importlib.metadata.version(name) for name in ('GDAL','rasterio','numpy')},'fixtures':[]}
with tempfile.TemporaryDirectory() as temporary, contextlib.chdir(temporary):
    temp=Path(temporary)
    y,x=np.mgrid[:64,:64]
    mask=np.zeros((64,64),dtype=np.uint8);mask[16:48,16:48]=1
    region=regionprops(mask)[0]
    area=float(region.area)
    # WGS84 local east-west metres per degree (rather than a rounded 111120).
    def longitude_m_per_degree(latitude):
        phi=math.radians(latitude)
        return 6378137*math.pi/180*math.cos(phi)/math.sqrt(1-6.6943799901413165e-3*math.sin(phi)**2)
    equator_scale=longitude_m_per_degree(0)
    latitude_scale=longitude_m_per_degree(24)
    cases=[
        ('projected_metre',3857,1.0,1.0,0,'m',.5,1.0),
        ('geographic_equator',4326,1/equator_scale,1/equator_scale,0,'m',.5,equator_scale),
        ('geographic_lat24',4326,1/latitude_scale,1/equator_scale,24,'m',.5,latitude_scale),
        ('horizontal_us_survey_foot_vertical_metre',2263,1.0,1.0,0,'m',.5*(1200/3937),1200/3937),
        ('horizontal_metre_vertical_foot',3857,1.0,1.0,0,'ft',.5/.3048,1/.3048),
    ]
    def assess(name,path,expected=None,physical_scale=None):
        ds=gdal.Open(str(path))
        srs=ds.GetSpatialRef()
        metadata={'size':[ds.RasterXSize,ds.RasterYSize],'crs':srs.GetName() if srs else None,
                  'geographic':bool(srs.IsGeographic()) if srs else None,
                  'horizontal_unit':srs.GetAttrValue('UNIT',0) if srs else None,
                  'horizontal_metre_factor':srs.GetLinearUnits() if srs and srs.IsProjected() else None,
                  'band_unit':ds.GetRasterBand(1).GetUnitType(),'band_nodata':ds.GetRasterBand(1).GetNoDataValue(),'geotransform':list(ds.GetGeoTransform())}
        slope=exportSlope(str(path),str(temp/(name+'-default.tif')))
        interior=slope[16:48,16:48]
        blob=Blob(region,0,0,1)
        calculateAreaUsingSlope(str(path),[blob])
        legacy=gdal.DEMProcessing(str(temp/(name+'-scale1.tif')),str(path),'slope',scale=1).ReadAsArray()[16:48,16:48]
        item={'name':name,'metadata':metadata,'median_slope_default':float(np.median(interior)),
              'median_slope_explicit_scale1':float(np.median(legacy)),
              'pixels_over_87_degrees':int((interior>87).sum()),
              'slope_nodata_pixels':int((interior == -9999).sum()),'planar_area_pixels':area,
              'taglab_surface_area_pixels':float(blob.surface_area),
              'taglab_surface_factor':float(blob.surface_area/area),
              'expected_slope_degrees':expected}
        if expected is not None:
            item['expected_surface_factor']=1/math.cos(math.radians(expected))
            item['expected_surface_area_pixels']=area*item['expected_surface_factor']
        if physical_scale is not None:
            corrected=gdal.DEMProcessing(str(temp/(name+'-physical.tif')),str(path),'slope',scale=physical_scale).ReadAsArray()[16:48,16:48]
            item['median_slope_explicit_physical_scale']=float(np.median(corrected))
        report['fixtures'].append(item)
        Path(args.report).write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(item),flush=True)
    for name,epsg,dx,dy,latitude,unit,dz,physical_scale in cases:
        path=temp/(name+'.tif')
        ds=gdal.GetDriverByName('GTiff').Create(str(path),64,64,1,gdal.GDT_Float32)
        srs=osr.SpatialReference();srs.ImportFromEPSG(epsg)
        ds.SetProjection(srs.ExportToWkt());ds.SetGeoTransform((0,dx,0,latitude,0,-dy))
        ds.GetRasterBand(1).SetUnitType(unit)
        ds.GetRasterBand(1).WriteArray((x*dz).astype(np.float32));ds=None
        assess(name,path,math.degrees(math.atan(.5)),physical_scale)
    nodata=temp/'projected_metre_nodata.tif'
    gdal.Translate(str(nodata),str(temp/'projected_metre.tif'))
    ds=gdal.Open(str(nodata),gdal.GA_Update)
    z=(x*.5).astype(np.float32);z[28:36,28:36]=-9999
    ds.GetRasterBand(1).SetNoDataValue(-9999);ds.GetRasterBand(1).WriteArray(z);ds=None
    assess('projected_metre_nodata',nodata,math.degrees(math.atan(.5)),1.0)
    source=ROOT/'models/.samples/Plot18_DEM.tiff'
    with source.open('rb') as stream:
        report['official_dem_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
    original=gdal.Open(str(source))
    report['official_dem_size']=[original.RasterXSize,original.RasterYSize]
    crop=temp/'Plot18_crop.tif'
    gdal.Translate(str(crop),str(source),srcWin=[original.RasterXSize//2-32,original.RasterYSize//2-32,64,64])
    assess('official_Plot18_center64',crop)
