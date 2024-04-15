

def plot_scatter_shift_error(df:pd.DataFrame):

    fig = px.scatter(df, x='X_Error', y='Y_Shift', color='cycle_id')

    fig.update_layout(
        title='Error vs Shift', 
        height=800,width=1200,
        paper_bgcolor="white", plot_bgcolor="white",
        uniformtext_minsize=12, uniformtext_mode='hide'
    )

    fig.update_yaxes(
        type="log", nticks=7, 
        showline=True, linecolor="black", mirror=True,
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
        showline=True, linecolor="black", mirror=True,
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
        marker=dict(
            size=8, 
            opacity=0.6, 
            line=dict(
                width=1, 
                color='DarkSlateGrey'
            )
        )
    )

    return fig


def ashlar (files:list, 
        channel_to_align:int, 
        max_shift:int, 
        filter_sigma:int, 
        verbose:bool=True,
        output_image_path:str=None
        ):
    
    reader = BioformatsReader(files[0])
    
    if verbose:
        print("------------------------------------")
        print("Stitching and registering input images")
        print('Cycle 0:')
        print(f'    reading {files[0]}')
    
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
        if verbose:
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