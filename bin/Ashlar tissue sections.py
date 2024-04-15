from ashlar import thumbnail, reg
from ashlar.scripts.ashlar import process_axis_flip

import numpy as np
import sys

from PIL import Image
from tifffile import imsave
import shutil
import os
import pandas as pd
import time
import random
from tqdm import tqdm

import gc

#for export to ome.zarr
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image

#for export to ome.tif
from ashlar.reg import PyramidWriter


from sparcstools._custom_ashlar_funcs import  plot_edge_scatter, plot_edge_quality
from sparcstools.filereaders import FilePatternReaderRescale

#define custom FilePatternReaderRescale to use with Ashlar to allow for custom modifications to images before performing stitching

from yattag import Doc, indent

def _write_xml(path, 
              channels, 
              slidename, 
              cropped = False):
    """ Helper function to generate an XML for import of stitched .tifs into BIAS.

    Parameters
    ----------
    path : str
        path to where the exported images are written
    channels : [str]
        list of the channel names written out
    slidename : str
        string indicating the name underwhich the files were written out
    cropped : bool
        boolean value indicating if the stitched images were written out cropped or not.
    """

    if cropped:
        image_paths = [slidename + "_"+x+'_cropped.tif' for x in channels]
    else:
        image_paths = [slidename + "_"+x+'.tif' for x in channels]

    doc, tag, text = Doc().tagtext()
    
    xml_header = '<?xml version="1.0" encoding="UTF-8"?>'
    doc.asis(xml_header)
    with tag("BIAS", version = "1.0"):
        with tag("channels"):
            for i, channel in enumerate(channels):
                with tag("channel", id = str(i+1)):
                    with tag("name"):
                        text(channel)
        with tag("images"):
            for i, image_path in enumerate(image_paths):
                with tag("image", url=str(image_path)):
                    with tag("channel"):
                        text(str(i+1))

    result = indent(
        doc.getvalue(),
        indentation = ' '*4,
        newline = '\r\n'
    )

    #write to file
    write_path = os.path.join(path, slidename + ".XML")
    with open(write_path, mode ="w") as f:
        f.write(result)

def generate_stitched(input_dir, 
                      slidename,
                      pattern,
                      outdir,
                      overlap = 0.1,
                      max_shift = 30, 
                      stitching_channel = "Alexa488",
                      crop = {'top':0, 'bottom':0, 'left':0, 'right':0},
                      plot_QC = True,
                      filetype = [".tif"],
                      WGAchannel = None,
                      do_intensity_rescale = True,
                      rescale_range = (1, 99),
                      no_rescale_channel = None,
                      export_XML = True,
                      return_tile_positions = True,
                      channel_order = None):
    
    """
    Function to generate a stitched image.

    Parameters
    ----------
    input_dir : str
        Path to the folder containing exported TIF files named with the following nameing convention: "Row{#}_Well{#}_{channel}_zstack{#}_r{#}_c{#}.tif". 
        These images can be generated for example by running the sparcstools.parse.parse_phenix() function.
    slidename : str
        string indicating the slidename that is added to the stitched images generated
    pattern : str
        Regex string to identify the naming pattern of the TIFs that should be stitched together. 
        For example: "Row1_Well2_{channel}_zstack3_r{row:03}_c{col:03}.tif". 
        All values in {} indicate those which are matched by regex to find all matching tifs.
    outdir : str
        path indicating where the stitched images should be written out
    overlap : float between 0 and 1
        value between 0 and 1 indicating the degree of overlap that was used while recording data at the microscope.
    max_shift: int
        value indicating the maximum threshold for tile shifts. Default value in ashlar is 15. In general this parameter does not need to be adjusted but it is provided
        to give more control.
    stitching_channel : str
        string indicating the channel name on which the stitching should be calculated. the positions for each tile calculated in this channel will be 
        passed to the other channels. 
    crop
        dictionary of the form {'top':0, 'bottom':0, 'left':0, 'right':0} indicating how many pixels (based on a generated thumbnail, 
        see sparcstools.stitch.generate_thumbnail) should be cropped from the final image in each indicated dimension. Leave this set to default 
        if no cropping should be performed.
    plot_QC : bool
        boolean value indicating if QC plots should be generated
    filetype : [str]
        list containing any of [".tif", ".ome.zarr", ".ome.tif"] defining to which type of file the stiched results should be written. If more than one 
        element is present in the list all export types will be generated in the same output directory.
    WGAchannel : str
        string indicating the name of the WGA channel in case an illumination correction should be performed on this channel
    do_intensity_rescale : bool
        boolean value indicating if the rescale_p1_P99 function should be applied before stitching or not. Alternatively partial then those channels listed in no_rescale_channel will 
        not be rescaled.
    no_rescale_channel : None | [str]
        either None or a list of channel strings on which no rescaling before stitching should be performed.
    export_XML
        boolean value. If true then an xml is exported when writing to .tif which allows for the import into BIAS.
    return_tile_positions : bool | default = True
        boolean value. If true and return_type != "return_array" the tile positions are written out to csv.
    channel_order : None | [int]
        if None do nothing, if list of ints is supplied remap channel order
    """
    start_time = time.time()

    #convert relativ paths into absolute paths
    outdir = os.path.abspath(outdir)

    #read data 
    print("performing stitching with ", str(overlap), " overlap.")
    slide = FilePatternReaderRescale(path = input_dir, pattern = pattern, overlap = overlap, rescale_range=rescale_range)
    
    # Turn on the rescaling
    slide.do_rescale = do_intensity_rescale
    slide.WGAchannel = WGAchannel

    if do_intensity_rescale == "partial":
        if no_rescale_channel != None:
            no_rescale_channel_id = []
            for _channel in no_rescale_channel:
                no_rescale_channel_id.append(slide.metadata.channel_map.values().index(_channel))
            slide.no_rescale_channel = no_rescale_channel_id
        else:
            sys.exit("do_intensity_rescale set to partial but not channel passed for which no rescaling should be done.")
    
    #flip y-axis to comply with labeling generated by opera phenix
    process_axis_flip(slide, flip_x=False, flip_y=True)

    #get dictionary position of channel
    channel_id = list(slide.metadata.channel_map.values()).index(stitching_channel)

    #generate aligner to use specificed channel for stitching
    print("performing stitching on channel ", stitching_channel, "with id number ", str(channel_id))
    aligner = reg.EdgeAligner(slide, channel=channel_id, filter_sigma=0, verbose=True, do_make_thumbnail=False, max_shift = max_shift)
    aligner.run() 

    #generate some QC plots
    if plot_QC:
        plot_edge_scatter(aligner, outdir)
        plot_edge_quality(aligner, outdir)
        #reg.plot_edge_scatter(aligner)
        print("need to implement this here. TODO")

    aligner.reader._cache = {} #need to empty cache for some reason

    #generate stitched file
    mosaic_args = {}
    mosaic_args['verbose'] = True
    mosaic_args['channels'] = list(slide.metadata.channel_map.keys())

    mosaic = reg.Mosaic(aligner, 
                        aligner.mosaic_shape, 
                        **mosaic_args
                        )

    mosaic.dtype = np.uint16

    if channel_order is None:
        _channels = mosaic.channels
    else:
        print(mosaic.channels)
        print("new channel order", channel_order)

        _channels = []
        for channel in channel_order:
            _channels.append(list(slide.metadata.channel_map.values()).index(channel))
            
        print("new channel order", _channels)

    if "return_array" in filetype:
        print("not saving positions")
    else:
        if return_tile_positions:
            #write out positions to csv
            positions = aligner.positions
            np.savetxt(os.path.join(outdir, slidename + "_tile_positions.tsv"), positions, delimiter="\t")
        else:
            print("not saving positions")
    
    if "return_array" in filetype:

        print("Returning array instead of saving to file.")
        mosaics = []
        
        for channel in tqdm(_channels):
            mosaics.append(mosaic.assemble_channel(channel = channel))

        merged_array = np.array(mosaics)
        merged_array = merged_array.astype("uint16")

        end_time = time.time() - start_time
        print('Merging Pipeline completed in ', str(end_time/60) , "minutes.")
        
        #get channel names
        channels = []
        for channel in  slide.metadata.channel_map.values():
            channels.append(channel)

        return(merged_array, channels)

    elif ".tif" in filetype:
        
        print("writing results to one large tif.")
        #define shape of output image

        n_channels = len(mosaic.channels)
        x, y = mosaic.shape

        from alphabase.io import tempmmap
        TEMP_DIR_NAME = tempmmap.redefine_temp_location(outdir)

        # initialize tempmmap array to save segmentation results to
        mosaics = tempmmap.array((n_channels, x, y), dtype=np.uint16)
        for i, channel in tqdm(enumerate(_channels), total = n_channels):
            mosaics[i, :, :] = mosaic.assemble_channel(channel = channel)
            
        #actually perform cropping
        if np.sum(list(crop.values())) > 0:
            print('Merged image will be cropped to the specified cropping parameters: ', crop)
            merged_array = np.array(mosaics)
            
            #manual garbage collection tp reduce memory footprint
            del mosaics
            gc.collect()

            cropping_factor = 20.00   #this is based on the scale that was used in the thumbnail generation
            _, x, y = merged_array.shape
            top = int(crop['top'] * cropping_factor)
            bottom = int(crop['bottom'] * cropping_factor)
            left = int(crop['left'] * cropping_factor)
            right = int(crop['right'] * cropping_factor)
            cropped = merged_array[:, slice(top, x-bottom), slice(left, y-right)]

            #manual garbage collection tp reduce memory footprint
            del merged_array
            gc.collect()

            #write to tif for each channel
            for i, channel in enumerate(slide.metadata.channel_map.values()):
                (print('writing to file: ', channel))
                im = Image.fromarray(cropped[i].astype('uint16'))#ensure that type is uint16
                im.save(os.path.join(outdir, slidename + "_"+channel+'_cropped.tif'))
            
            if export_XML:
                _write_xml(outdir, slide.metadata.channel_map.values(), slidename, cropped = True)

            #manual garbage collection tp reduce memory footprint
            del cropped
            gc.collect()

        else:
            merged_array = np.array(mosaics)
            
            #manual garbage collection tp reduce memory footprint
            del mosaics
            gc.collect()

            for i, channel in enumerate(slide.metadata.channel_map.values()):
                im = Image.fromarray(merged_array[i].astype('uint16'))#ensure that type is uint16
                im.save(os.path.join(outdir, slidename + "_"+channel+'.tif'))

            if export_XML:
                _write_xml(outdir, slide.metadata.channel_map.values(), slidename, cropped = False)

            del merged_array
            
    elif "ome.tif" in filetype:
        print("writing results to ome.tif")
        path = os.path.join(outdir, slidename + ".ome.tiff")
        writer = PyramidWriter([mosaic], path, scale=5, tile_size=1024, peak_size=1024, verbose=True)
        writer.run()

    elif "ome.zarr" in filetype:
        print("writing results to ome.zarr")

        if 'mosaics' not in locals():
            #define shape of output image

            n_channels = len(_channels)
            x, y = mosaic.shape

            from alphabase.io import tempmmap
            TEMP_DIR_NAME = tempmmap.redefine_temp_location(outdir)

            # initialize tempmmap array to save segmentation results to
            mosaics = tempmmap.array((n_channels, x, y), dtype=np.uint16)
            for i, channel in tqdm(enumerate(_channels), total = n_channels):
                mosaics[i, :, :] = mosaic.assemble_channel(channel = channel)

        path = os.path.join(outdir, slidename + ".ome.zarr")

        #delete file if it already exists
        if os.path.isdir(path):
            shutil.rmtree(path)
            print("Outfile already existed, deleted.")

        loc = parse_url(path, mode="w").store
        group = zarr.group(store = loc)
        axes = "cyx"

        channel_colors = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"]
        #chek if length of colors is enough for all channels in slide otherwise loop through n times
        while len(slide.metadata.channel_map.values()) > len(channel_colors):
            channel_colors = channel_colors + channel_colors

        group.attrs["omero"] = {
            "name":slidename + ".ome.zarr",
            "channels": [{"label":channel, "color":channel_colors[i], "active":True} for i, channel in enumerate(slide.metadata.channel_map.values())]
        }  
        write_image(mosaics, group = group, axes = axes, storage_options=dict(chunks=(1, 1024, 1024)))
   
    #perform garbage collection manually to free up memory as quickly as possible
    print("deleting old variables")
    if "mosaic" in locals():
        del mosaic
        gc.collect()

    if "mosaics" in locals():
        del mosaics
        gc.collect()

    if "TEMP_DIR_NAME" in locals():
        shutil.rmtree(TEMP_DIR_NAME, ignore_errors=True)
        del TEMP_DIR_NAME
        gc.collect()

    end_time = time.time() - start_time
    print('Merging Pipeline completed in ', str(end_time/60) , "minutes.")


#get parameters according to slide name
well = str(1)
row = str(1)
overlap = 0.1
max_shift = 30
stitching_channel = "Alexa647"
zstack_value = str(1)
timepoint = str(1)
output_filetype = ["ome.zarr"]
channel_order = ["Alexa488", "Alexa647", "Alexa568"]
print(channel_order)

#generate directory paths for output
input_dir = os.path.join(input_stitching_dir, phenix_dir, "parsed_images")
print("input_dir:", input_dir)
outdir_merged = os.path.join(outdir, slidename, 'stitched')
print("outdir_merged:", outdir_merged)

#create stitching directory
if not os.path.exists(outdir_merged):
    os.makedirs(outdir_merged)

#define pattern to recognize which slide should be stitched
pattern = f"Timepoint{timepoint.zfill(3)}_Row{row.zfill(2)}_Well{well.zfill(2)}_" + "{channel}"+f"_zstack{zstack_value.zfill(3)}_" + "r{row:03}_c{col:03}.tif"
print("pattern:", pattern)

generate_stitched(input_dir, 
                slidename, 
                pattern, 
                outdir_merged, 
                overlap=overlap, 
                max_shift=max_shift,
                do_intensity_rescale = True, 
                no_rescale_channel = "Alexa568",
                rescale_range = (10, 70),
                stitching_channel=stitching_channel, 
                filetype = output_filetype,
                channel_order = channel_order)
