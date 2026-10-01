"""Numerical C++/PyTorch parity for actual pretrained Vocos codes/features."""
import argparse
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import torch
from vocos.models import VocosBackbone
from vocos.heads import ISTFTHead
from export_vocos import convert
from export_model import export


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('executable',type=Path);p.add_argument('config',type=Path);p.add_argument('checkpoint',type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    meta,weights=convert(a.config,a.checkpoint)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    config=__import__('yaml').safe_load(Path(a.config).read_text())
    b=config['backbone']['init_args'];h=config['head']['init_args']
    backbone=VocosBackbone(b['input_channels'],b['dim'],b['intermediate_dim'],b['num_layers'],
                           adanorm_num_embeddings=b['adanorm_num_embeddings']).eval()
    head=ISTFTHead(h['dim'],h['n_fft'],h['hop_length'],h['padding']).eval()
    backbone.load_state_dict({k.removeprefix('backbone.'):v for k,v in state.items() if k.startswith('backbone.')},strict=True)
    head.load_state_dict({k.removeprefix('head.'):v for k,v in state.items() if k.startswith('head.')},strict=True)
    table=state['feature_extractor.codebook_weights'].reshape(16,1024,128)
    rng=np.random.default_rng(24)
    def compare(actual,reference,label):
        if actual.shape!=reference.shape or not np.isfinite(actual).all():
            raise AssertionError(f'{label}: invalid output shape or nonfinite samples')
        error=float(np.max(np.abs(actual-reference)))
        relative=float(np.linalg.norm(actual-reference)/max(np.linalg.norm(reference),1e-9))
        print(f'{label}: max_abs={error:.3g} relative_l2={relative:.3g}',flush=True)
        if error>=3e-4 or relative>=3e-3:
            raise AssertionError(f'{label}: parity tolerance exceeded')
    with tempfile.TemporaryDirectory(prefix='vocos-native-parity-') as d:
        root=Path(d);model_path=root/'pretrained.vocos';export(model_path,meta,weights)
        for bandwidth,q in enumerate((2,4,8,16)):
            def decode_codes(codes):
                codes.tofile(root/'codes.u16')
                subprocess.run([str(a.executable.resolve()),str(model_path),str(root/'codes.u16'),
                                str(root/'native.f32'),'--codes',str(bandwidth)],check=True,
                               stdout=subprocess.DEVNULL)
                return np.fromfile(root/'native.f32',dtype=np.float32)
            for frames in (1,11,337):
                codes=rng.integers(0,1024,size=(frames,q),dtype=np.uint16)
                with torch.inference_mode():
                    code_tensor=torch.from_numpy(codes.astype(np.int64).T.copy())
                    feature=sum(table[book,code_tensor[book]] for book in range(q)).T[None]
                    reference=head(backbone(feature,bandwidth_id=torch.tensor([bandwidth])))[0].numpy()
                actual=decode_codes(codes)
                compare(actual,reference,f'codes bandwidth_id={bandwidth} frames={frames}')
                if frames==11:
                    features=feature[0].T.contiguous().numpy().astype(np.float32)
                    features.tofile(root/'features.f32')
                    subprocess.run([str(a.executable.resolve()),str(model_path),str(root/'features.f32'),
                                    str(root/'native-feature.f32'),'--features',str(bandwidth)],check=True,
                                   stdout=subprocess.DEVNULL)
                    native_feature=np.fromfile(root/'native-feature.f32',dtype=np.float32)
                    compare(native_feature,reference,f'features bandwidth_id={bandwidth}')
                if frames==337:
                    left=decode_codes(codes[:332])[:300*320]
                    right=decode_codes(codes[225:])[75*320:]
                    compare(left,actual[:300*320],f'left context bandwidth_id={bandwidth}')
                    compare(right,actual[300*320:],f'right context bandwidth_id={bandwidth}')
    print('Pretrained Vocos 24 kHz mono parity passed for all four rates.')


if __name__=='__main__':main()
