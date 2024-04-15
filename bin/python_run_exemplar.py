from ashlar import fileseries, thumbnail,reg
from ashlar.reg import plot_edge_shifts, plot_edge_quality
from ashlar.reg import BioformatsReader
from ashlar.reg import PyramidWriter
from ashlar.viewer import view_edges
# from ashlar.scripts.ashlar import process_axis_flip
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import numpy as np
import pandas as pd
# import pims
import napari
import os
from plotly import express as px
import plotly.io as pio
import kaleido
import skimage
import time
from os.path import abspath
def get_datetime():
    return time.strftime("%Y%m%d_%H%M%S")

def parse_args():
    parser = argparse.ArgumentParser(description='Ashlar')
    parser.add_argument('-i', '--path-to-images', type=str, required=True, dest='path_to_images',
                        help='Path files to process, must be bioformats compatible')
    parser.add_argument('-c', '--channel-to-align', type=int, default=0,
                        help='Channel to align')
    parser.add_argument('-m', '--max-shift', type=int, default=30,
                        help='Maximum shift')
    parser.add_argument('--filter-sigma', type=float, default=1,
                        help='Filter sigma')
    parser.add_argument('--output-image-path', type=str, default=None,
                        help='Output image path')
    parser.add_argument('--quiet', action='store_true',
                        help='Quiet')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose')
    parser.add_argument('--output-path', type=str, default=None,
                        help='Output path')

    args = parser.parse_args()
    args.path_to_images = abspath(args.path_to_images)
    return args



def ashlar (files:list, 
        channel_to_align:int, 
        max_shift:int, 
        filter_sigma:int, 
        quiet: bool=False,
        verbose:bool=True,
        output_image_path:str=None
        ):
    
    reader = BioformatsReader(files[0])
    
    if not quiet:
        print("Stitching and registering input images")
        print('Cycle 0:')
        print('    reading %s' % files[0])
    
    aligner = reg.EdgeAligner(reader=reader, channel=channel_to_align, max_shift=max_shift, filter_sigma=filter_sigma, verbose=verbose)
    aligner.run()

    df = pd.DataFrame(
        data = {
            "edge_id": [f"edge_{str(i).zfill(4)}" for i in range(0, aligner.all_errors.shape[0])],
            "cycle_id": "cycle_01",
            'X_Error': np.clip(aligner.all_errors, 0, 10),
            'Y_Shift': np.clip([np.linalg.norm(v[0]) for v in aligner._cache.values()], 0.01, np.inf)
        }
    )

    mosaic_args = {}
    mosaic_args['verbose'] = verbose
    mosaic_args['channels'] = list(range(0, reader.metadata.num_channels))

    mosaics = []
    mosaics.append(reg.Mosaic(aligner, aligner.mosaic_shape, **mosaic_args))

    layers = []

    for i, file in enumerate(files[1:]):
        if not quiet:
            print('    reading %s' % file)
        
        tmp_reader = BioformatsReader(file)
        tmp_aligner = reg.EdgeAligner(reader=tmp_reader, channel=channel_to_align, max_shift=max_shift, filter_sigma=filter_sigma, verbose=verbose)
        tmp_aligner.run()
        tmp_df = pd.DataFrame(
            data = {
                "edge_id": [f"edge_{str(i).zfill(4)}" for i in range(0, tmp_aligner.all_errors.shape[0])],
                "cycle_id": f"cycle_{str(i+2).zfill(2)}",
                'X_Error': np.clip(tmp_aligner.all_errors, 0, 10),
                'Y_Shift': np.clip([np.linalg.norm(v[0]) for v in tmp_aligner._cache.values()], 0.01, np.inf)
            }
        )

        df = pd.concat([df, tmp_df], ignore_index=True)
        
        layer_aligner = reg.LayerAligner(tmp_reader, aligner)
        layer_aligner.run()
        layers.append(layer_aligner)
        mosaics.append(reg.Mosaic(layer_aligner, aligner.mosaic_shape))

    print(f"Writing output image to {output_image_path}")
    writer = PyramidWriter(mosaics, output_image_path, verbose=verbose)
    writer.run()

    return df,aligner,layers

def main(files, channel_to_align, max_shift, filter_sigma, output_image_path, QC_folder):
    
    datetime = get_datetime()

    df, aligner, layers =  ashlar (files=files, 
        channel_to_align=channel_to_align, 
        max_shift=max_shift, 
        filter_sigma=filter_sigma, 
        quiet=False,
        verbose=True,
        output_image_path= output_image_path
        )
    
    export_path= os.path.join(QC_folder, f"{datetime}_ashlar_QC_plots.pdf")

    fig = plot_scatter_shift_error(df)
    plt.suptitle(f"Error vs Shift \n -m {aligner.max_shift} --filter-sigma {aligner.filter_sigma}", color='salmon')
    pio.write_image(fig, os.path.join(QC_folder, f"{datetime}_Error_vs_Shift_ScatterPlot.pdf"), scale=6, width=600, height=500)

    filename = export_path
    pdf_pages = PdfPages(filename)

    plot_edge_shifts(aligner)
    #brighter is more distance
    plt.suptitle(f"Edge Shifts : Brighter is more shift \n -m {aligner.max_shift} --filter-sigma {aligner.filter_sigma}", color='salmon')
    plt.tight_layout()
    plt.gcf().set_size_inches(16, 9)
    fig = plt.gcf()
    pdf_pages.savefig(fig)
    plt.close(fig)

    plot_edge_quality(aligner, img=aligner.reader.thumbnail)
    plt.suptitle(f"Edge Quality : Brighter is better \n -m {aligner.max_shift} --filter-sigma {aligner.filter_sigma}", color='salmon')
    plt.tight_layout()
    plt.gcf().set_size_inches(16, 9)
    fig = plt.gcf()
    pdf_pages.savefig(fig)
    plt.close(fig)


    for i, layer in enumerate(layers):
        layer.mosaic_shape = aligner.mosaic_shape
        reg.plot_layer_quality(layer, img=aligner.reader.thumbnail)
        plt.gcf().suptitle(f"Cycle {i+1} -m {aligner.max_shift} --filter-sigma {aligner.filter_sigma}", color='salmon')
        plt.gcf().tight_layout()
        plt.gcf().set_size_inches(16, 9)
        fig = plt.gcf()
        pdf_pages.savefig(fig)
        plt.close(fig)

    # Close the PdfPages object
    pdf_pages.close()

def plot_scatter_shift_error(df:pd.DataFrame):

    fig = px.scatter(
        df, x='X_Error', y='Y_Shift', 
        color='cycle_id', 
        hover_data=['edge_id'], 
        text='edge_id', )

    fig.update_layout(
        title='Error vs Shift', 
        height=800,
        width=1200,
        paper_bgcolor="white",
        plot_bgcolor="white",
        uniformtext_minsize=12, 
        uniformtext_mode='hide'
    )
    fig.update_yaxes(
        showgrid=False,

        gridcolor='LightGrey',
        showline=True,
        linecolor="black",
        mirror=True,

        nticks=7,
        title=dict(
            text='Shift distance (pixels)',
            font=dict(
                family='Arial',
                size=18,
                color='black'
                )   
        )
    )
    fig.update_xaxes(
        showgrid=False,

        gridcolor='LightGrey',
        showline=True,
        linecolor="black",
        mirror=True,

        title=dict(
            text='Error (NCC)', 
            font=dict(
                family='Arial',
                size=18,
                color='black'
                )   
        )
    )

    fig.update_traces(
        textposition='top center',
        textfont_size=10,
        texttemplate='',
        marker=dict(
            size=14, 
            opacity=0.9, 
            line=dict(
                width=1, 
                color='DarkSlateGrey'
            )
        )
    )

    fig.update_annotations(
        font=dict(
            family="Arial",
            size=10,
            color="black"
        )
    )

    return fig

if __name__ == "__main__":
    args = parse_args()
    files = [os.path.join(args.path_to_images, file) for file in os.listdir(args.path_to_images)]
    main(files, args.channel_to_align, args.max_shift, args.filter_sigma, args.output_image_path, args.output_path)
