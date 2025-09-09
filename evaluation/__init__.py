from evaluation.coco_evaler import COCOEvaler
from evaluation.flickr8k_evaler import Flickr8kEvaler
from evaluation.flickr8k_hf_evaler import Flickr8kHFEvaler

__factory = {
    'COCO': COCOEvaler,
    'FLICKR8K': Flickr8kEvaler,
    'FLICKR8K_HF': Flickr8kHFEvaler,
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown Evaler:", name)
    return __factory[name](*args, **kwargs)
