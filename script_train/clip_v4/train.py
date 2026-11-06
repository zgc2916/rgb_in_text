"""Train v4 joint model with explicit resource paths using torchrun."""
import argparse
import json
from pathlib import Path
from script_train.clip_m3fd_joint.runtime import build


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=('rgbtvg_flir','rgbtvg_m3fd','rgbtvg_mfad'),required=True)
    for arg in ('data-root','split-root','clip-path','bert-path','output-dir'):
        p.add_argument('--'+arg,required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--public-checkpoint');group.add_argument('--resume')
    p.add_argument('--epochs',type=int,default=120);p.add_argument('--num-workers',type=int,default=4)
    c=p.parse_args()
    a=argparse.Namespace(**json.loads((Path(__file__).parent/'recipe.json').read_text()))
    a.dataset=c.dataset;a.full_match_dataset=c.dataset.removeprefix('rgbtvg_')
    a.data_root=str(Path(c.data_root).resolve());a.split_root=str(Path(c.split_root).resolve())
    a.pretrained_model_path=str(Path(c.clip_path).resolve());a.dataset_bert_model_path=str(Path(c.bert_path).resolve())
    a.output_dir=str(Path(c.output_dir).resolve());a.epochs=c.epochs;a.num_workers=c.num_workers
    a.retrain=str(Path(c.public_checkpoint).resolve()) if c.public_checkpoint else ''
    a.resume=str(Path(c.resume).resolve()) if c.resume else ''
    if a.resume:
        import torch
        last=torch.load(a.resume,map_location='cpu',weights_only=False)['epoch']
        a.hi_lora_stage=3 if last>=100 else 2 if last>=80 else 1 if last>=60 else 0
    elif (Path(a.output_dir)/'log.txt').exists():
        raise FileExistsError('Output already contains a training log; choose a new directory or use --resume')
    Path(a.output_dir).mkdir(parents=True,exist_ok=True)
    import train_val.mmvg_train as trainer
    trainer.build_model=lambda args:build(args,public_migration=not bool(args.resume))
    trainer.main(a)

if __name__=='__main__':main()
