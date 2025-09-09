import os
import numpy as np
import torch
import tqdm
import json
import evaluation
import lib.utils as utils
import datasets_.data_loader as data_loader
from lib.config import cfg


class Evaler(object):
    def __init__(self, eval_ids, gv_feat, att_feats, eval_annfile, max_samples=None):
        super(Evaler, self).__init__()
        self.vocab = utils.load_vocab(cfg.INFERENCE.VOCAB)
        self.max_samples = max_samples

        # Build eval ids
        if cfg.INFERENCE.EVAL == 'COCO':
            with open(eval_ids, 'r') as f:
                self.ids2path = json.load(f)
                self.eval_ids = np.array(list(self.ids2path.keys()))
        else:
            self.ids2path = None
            self.eval_ids = None  # set after loader

        # Build loader (uses ids_path basename to infer split)
        self.eval_loader = data_loader.load_val(eval_ids, gv_feat, att_feats, max_samples=self.max_samples)
        if self.eval_ids is None:
            # For Flickr8k, load the actual image IDs from the JSON file if available
            if eval_ids and os.path.exists(eval_ids):
                with open(eval_ids, 'r') as f:
                    ids_data = json.load(f)
                    # Convert string keys to integers for consistency
                    self.eval_ids = np.array([int(k) for k in ids_data.keys()])
                    print(f"Loaded {len(self.eval_ids)} evaluation IDs from {eval_ids}")
                    print(f"ID range: {self.eval_ids.min()} to {self.eval_ids.max()}")
            else:
                # Fallback to sequential IDs when no file is provided (HuggingFace mode)
                self.eval_ids = np.arange(len(self.eval_loader.dataset))
                print(f"Using sequential IDs: 0 to {len(self.eval_ids)-1} (HuggingFace mode)")

        # Apply max_samples limit if specified (should already be handled by load_val)
        if self.max_samples is not None and self.max_samples > 0:
            original_size = len(self.eval_ids)
            if original_size > self.max_samples:
                self.eval_ids = self.eval_ids[:self.max_samples]
                print(f"Evaluation: Limited to {len(self.eval_ids)} samples (from {original_size})")

        # Create evaluator instance
        if cfg.INFERENCE.EVAL == 'COCO':
            self.evaler = evaluation.create('COCO', eval_annfile)
        elif cfg.INFERENCE.EVAL == 'FLICKR8K_HF':
            # Use HF-based evaluator that doesn't need annotation files
            # Infer split from eval_ids path, or default to validation
            if eval_ids and os.path.exists(eval_ids):
                basename = os.path.basename(str(eval_ids)).lower()
                if 'val' in basename or 'valid' in basename:
                    split = 'validation'
                elif 'test' in basename:
                    split = 'test'
                else:
                    split = 'train'
            else:
                # Default to validation when no eval_ids path is provided
                split = 'validation'
            self.evaler = evaluation.create('FLICKR8K_HF', split)
        else:
            # For original Flickr8k, use the annotation file
            self.evaler = evaluation.create('FLICKR8K', eval_annfile)

    def make_kwargs(self, indices, ids, gv_feat, att_feats, att_mask):
        kwargs = {}
        kwargs[cfg.PARAM.INDICES] = indices
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs['BEAM_SIZE'] = cfg.INFERENCE.BEAM_SIZE
        kwargs['GREEDY_DECODE'] = cfg.INFERENCE.GREEDY_DECODE
        return kwargs

    def __call__(self, model, rname):
        model.eval()

        results = []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        global_idx = 0
        # output_indices = {0, 5, 10, 15, 20, 25}
        with torch.no_grad():
            for _, (indices, gv_feat, att_feats, att_mask) in enumerate(tqdm.tqdm(self.eval_loader, desc=rname)):
                ids = self.eval_ids[indices]
                gv_feat = gv_feat.to(device)
                att_feats = att_feats.to(device)
                att_mask = att_mask.to(device)
                kwargs = self.make_kwargs(indices, ids, gv_feat, att_feats, att_mask)
                m = getattr(model, 'module', model)
                if kwargs['BEAM_SIZE'] > 1:
                    seq, _ = m.decode_beam(**kwargs)
                else:
                    seq, _ = m.decode(**kwargs)

                sents = utils.decode_sequence(self.vocab, seq.data)
                for sid, sent in enumerate(sents):
                    result = {cfg.INFERENCE.ID_KEY: int(ids[sid]), cfg.INFERENCE.CAP_KEY: sent}
                    results.append(result)
                    if global_idx in range(5):
                        image_id = int(ids[sid])
                        print("\n" + "="*60)
                        print(f"Eval Sample {global_idx}")
                        print(f"Image ID: {image_id}")
                        print(f"Generated: {sent}")
                        # Ground truth
                        gt_captions = []
                        if hasattr(self.evaler, 'id_to_captions') and image_id in self.evaler.id_to_captions:
                            gt_captions = self.evaler.id_to_captions[image_id]
                        elif hasattr(self.evaler, 'coco_data'):
                            for ann in self.evaler.coco_data.get('annotations', []):
                                if ann['image_id'] == image_id:
                                    gt_captions.append(ann['caption'])
                        if gt_captions:
                            print("Ground Truth:")
                            for i, gt_cap in enumerate(gt_captions[:3]):
                                print(f"  {i+1}: {gt_cap}")
                        print("-" * 40)
                    global_idx += 1

        # Evaluate
        eval_res = self.evaler.eval(results)

        result_folder = os.path.join(cfg.ROOT_DIR, 'result')
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        json.dump(results, open(os.path.join(result_folder, 'result_' + rname + '.json'), 'w'))

        model.train()
        return eval_res
