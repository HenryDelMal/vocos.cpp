"""Convert the released Vocos EnCodec 24 kHz checkpoint to VOCOSN2 format."""
import argparse
from pathlib import Path
import numpy as np
import torch
import yaml
from export_model import export


def convert(config_path, checkpoint_path):
    config=yaml.safe_load(Path(config_path).read_text())
    backbone=config['backbone']['init_args']; head=config['head']['init_args']
    feature=config['feature_extractor']
    expected=(backbone.get('input_channels'),backbone.get('dim'),backbone.get('intermediate_dim'),
              backbone.get('num_layers'),backbone.get('adanorm_num_embeddings'),head.get('dim'),
              head.get('n_fft'),head.get('hop_length'),head.get('padding'))
    if (expected!=(128,384,1152,8,4,384,1280,320,'same') or
        config['head']['class_path']!='vocos.heads.ISTFTHead' or
        feature['class_path']!='vocos.feature_extractors.EncodecFeatures' or
        feature['init_args'].get('encodec_model')!='encodec_24khz' or
        feature['init_args'].get('bandwidths')!=[1.5,3.0,6.0,12.0]):
        raise ValueError(f'Checkpoint config is not the supported Vocos EnCodec 24 kHz model: {expected}')
    state=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    weights={}
    fixed={'backbone.embed.':'embed.','backbone.norm.scale.weight':'norm.scale',
           'backbone.norm.shift.weight':'norm.shift','backbone.final_layer_norm.':'final_norm.',
           'head.out.':'head.'}
    for key,value in state.items():
        name=None
        if key.startswith('backbone.convnext.'):
            name='blocks.'+key.removeprefix('backbone.convnext.')
        else:
            for source,target in fixed.items():
                if key.startswith(source):
                    name=target+key.removeprefix(source);break
        if name is not None:
            name=name.replace('.norm.scale.weight','.norm.scale').replace('.norm.shift.weight','.norm.shift')
            weights[name]=value.detach().cpu().numpy()
        elif key=='feature_extractor.codebook_weights':
            codebooks=value.detach().cpu().numpy()
            if codebooks.shape!=(16384,128): raise ValueError(f'Unexpected EnCodec codebook table {codebooks.shape}')
            weights['rvq']=codebooks.reshape(16,1024,128)
        elif key=='head.istft.window':
            window=value.detach().cpu().numpy()
            expected_window=torch.hann_window(1280).numpy().astype(window.dtype)
            if not np.allclose(window,expected_window,atol=1e-7,rtol=0):
                raise ValueError('Unexpected Vocos inverse-FFT window')
    required={'embed.weight','embed.bias','norm.scale','norm.shift','final_norm.weight','final_norm.bias',
              'head.weight','head.bias','rvq'}
    for i in range(8):
        p=f'blocks.{i}.'
        required|={p+'dwconv.weight',p+'dwconv.bias',p+'norm.scale',p+'norm.shift',
                   p+'pwconv1.weight',p+'pwconv1.bias',p+'pwconv2.weight',p+'pwconv2.bias',p+'gamma'}
    missing=required-weights.keys()
    if missing: raise ValueError(f'Checkpoint missing decoder tensors: {sorted(missing)}')
    metadata=dict(sample_rate=24000,channels=1,hop_length=320,n_fft=1280,latent_dim=128,
                  hidden_dim=384,intermediate_dim=1152,layers=8,codebooks=16,entries=1024,bandwidths=4)
    return metadata,weights


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path);p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();meta,weights=convert(a.config,a.checkpoint);export(a.output,meta,weights)
    print(f"Exported pretrained Vocos 24 kHz mono decoder ({len(weights)} tensors).")


if __name__=='__main__':main()
