import os
import numpy as np
import json
from lib.config import cfg
from coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from coco_caption.pycocoevalcap.bleu.bleu import Bleu
from coco_caption.pycocoevalcap.rouge.rouge import Rouge
from coco_caption.pycocoevalcap.cider.cider import Cider

try:
    from coco_caption.pycocoevalcap.meteor.meteor import Meteor
    _HAS_METEOR = True
except Exception:
    _HAS_METEOR = False

try:
    from coco_caption.pycocoevalcap.ciderR.ciderR import CiderR
    _HAS_CIDERR = True
except Exception:
    _HAS_CIDERR = False

try:
    from coco_caption.pycocoevalcap.spice.spice import Spice
    _HAS_SPICE = True
except Exception:
    _HAS_SPICE = False

# Remove the HuggingFace dataset import to avoid RLock issues
# from datasets.flickr8k_hf import load_flickr8k


class Flickr8kEvaler(object):
    """
    Flickr8k evaluator using pycocoevalcap metrics without COCO jsons.

    Expects results as list of dicts: {'image_id': int, 'caption': str}.
    Ground truth references are loaded from the HF Flickr8k split inferred from
    the basename of ids_path (contains 'val'/'valid' -> validation, 'test' -> test, else train).
    """

    def __init__(self, annfile):
        super().__init__()
        self.annfile = annfile
        
        # Load annotations from the preprocessed COCO-format file
        with open(annfile, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build a mapping from image_id to captions for quick lookup
        self.id_to_captions = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.id_to_captions:
                self.id_to_captions[img_id] = []
            self.id_to_captions[img_id].append(ann['caption'])
        
        self.tokenizer = PTBTokenizer()

    def _build_refs(self, img_ids):
        gts = {}
        for img_id in img_ids:
            # Use the preloaded COCO-format annotations
            caps = self.id_to_captions.get(img_id, [])
            if not caps:
                caps = ['.']  # fallback caption
            gts[img_id] = [{'caption': c} for c in caps]
        return gts

    def _build_res(self, results):
        res = {}
        for r in results:
            iid = int(r['image_id'])
            res.setdefault(iid, [])
            res[iid].append({'caption': r['caption']})
        return res

    def eval(self, results):
        img_ids = sorted({int(r['image_id']) for r in results})
        gts = self._build_refs(img_ids)
        res = self._build_res(results)

        # Tokenize
        gts_tok = self.tokenizer.tokenize(gts)
        res_tok = self.tokenizer.tokenize(res)

        scores = {}
        requested_metrics = cfg.SCORER.TYPES if hasattr(cfg.SCORER, 'TYPES') else ['Bleu_4']
        
        print(f"Computing evaluation metrics: {requested_metrics}")
        
        # BLEU metrics (1-4)
        bleu_metrics = [m for m in requested_metrics if m.startswith('Bleu_')]
        if bleu_metrics:
            bleu = Bleu(4)
            bscore, _ = bleu.compute_score(gts_tok, res_tok)
            bleu_names = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]
            for i, k in enumerate(bleu_names):
                if k in requested_metrics:
                    scores[k] = float(bscore[i])
                    print(f"  {k}: {scores[k]:.4f}")

        # ROUGE_L
        if 'ROUGE_L' in requested_metrics:
            try:
                rouge = Rouge()
                rscore, _ = rouge.compute_score(gts_tok, res_tok)
                scores['ROUGE_L'] = float(rscore)
                print(f"  ROUGE_L: {scores['ROUGE_L']:.4f}")
            except Exception as e:
                print(f"  ROUGE_L: Error - {e}")
                scores['ROUGE_L'] = 0.0

        # CIDEr
        if 'CIDEr' in requested_metrics or 'Cider' in requested_metrics:
            try:
                cider = Cider()
                cscore, _ = cider.compute_score(gts_tok, res_tok)
                scores['CIDEr'] = float(cscore)
                print(f"  CIDEr: {scores['CIDEr']:.4f}")
            except Exception as e:
                print(f"  CIDEr: Error - {e}")
                scores['CIDEr'] = 0.0

        # CIDEr-R
        if 'CIDEr-R' in requested_metrics and _HAS_CIDERR:
            try:
                ciderr = CiderR()
                cr, _ = ciderr.compute_score(gts_tok, res_tok)
                scores['CIDEr-R'] = float(cr)
                print(f"  CIDEr-R: {scores['CIDEr-R']:.4f}")
            except Exception as e:
                print(f"  CIDEr-R: Error - {e}")
                scores['CIDEr-R'] = 0.0

        # METEOR
        if 'METEOR' in requested_metrics and _HAS_METEOR:
            try:
                meteor = Meteor()
                m, _ = meteor.compute_score(gts_tok, res_tok)
                scores['METEOR'] = float(m)
                print(f"  METEOR: {scores['METEOR']:.4f}")
            except Exception as e:
                print(f"  METEOR: Error - {e}")
                scores['METEOR'] = 0.0

        # SPICE
        if 'SPICE' in requested_metrics and _HAS_SPICE:
            try:
                spice = Spice()
                s, _ = spice.compute_score(gts_tok, res_tok)
                scores['SPICE'] = float(s)
                print(f"  SPICE: {scores['SPICE']:.4f}")
            except Exception as e:
                print(f"  SPICE: Error - {e}")
                scores['SPICE'] = 0.0

        return scores

    # keep an API compatible with COCOEvaler
    def eval_no_spice(self, results):
        out = self.eval(results)
        out.pop('SPICE', None)
        return out

