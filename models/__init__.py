def build_model(args):
    """Construct only the selected model and avoid unrelated optional deps."""
    if args.model_name=='MMVG':
        from .mmvg import MMVG
        print('Building MMVG model...')
        return MMVG(args)
    if args.model_name=='MMVGFusion':
        from .mmvg_fusion import MMVGFusion
        print('Building MMVGFusion model...')
        return MMVGFusion(args)
    if args.model_name=='MMVG_te':
        from .mmvg_twoenc import MMVG
        print('Building MMVG two encoder model...')
        return MMVG(args)
    elif args.model_name=='HiVG':
        from .HiVG import HiVG
        print('Building HiVG model...')
        return HiVG(args)
    elif args.model_name=='FSVG':
        print('Building FSVG model...')
        from .fsvg_clip import FSVG
        return FSVG(args)
    elif args.model_name=='CLIP_VG':
        from .clip_vg import CLIP_VG
        # if hasattr(args, 'eval_model') and args.eval_model:
        #     print('Building ML_CLIP_VG model...')
        #     return ML_CLIP_VG(args)
        # else: 
        print('Building CLIP_VG model...')
        return CLIP_VG(args)
    elif args.model_name=='TransVG':
        from .trans_vg import TransVG
        print('Building TransVG model...')
        return TransVG(args)
    elif args.model_name=='AttBalance':
        from .attbalance import AttBalance
        print('Building AttBalance model...')
        return AttBalance(args)
    elif args.model_name=='QRNet':
        from .trans_vg import TransVGSwin
        print('Building QRNet model...')
        return TransVGSwin(args)
    elif args.model_name=='MMCA':
        from .mmca_vg import MMCA
        print('Building MMCA model...')
        return MMCA(args)
    elif args.model_name=='MDETR':
        print('Building MDETR model...')
        if args.model_type == 'ResNet':
            from .dynamic_mdetr_resnet import DynamicMDETR as DynamicMDETR_ResNet
            return DynamicMDETR_ResNet(args)
        elif args.model_type == 'CLIP':
            from .dynamic_mdetr_clip import DynamicMDETR as DynamicMDETR_CLIP
            return DynamicMDETR_CLIP(args)
    elif args.model_name=='OneRef':
        print('Building OnRef model...')
        if args.model == 'beit3_base_patch16_224':
            from .OneRef_model import beit3_base_patch16_224_grounding
            return beit3_base_patch16_224_grounding(args)
        elif args.model == 'beit3_large_patch16_384':
            from .OneRef_model import beit3_large_patch16_384_grounding
            return beit3_large_patch16_384_grounding(args)
