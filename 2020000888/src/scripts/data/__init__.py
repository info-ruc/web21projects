"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""
from .data import (TxtTokLmdb, DetectFeatLmdb,
                   ImageLmdbGroup, ConcatDatasetWithLens)
from .sampler import TokenBucketSampler
from .loader import PrefetchLoader, MetaLoader
from .vqa import VqaDataset, VqaEvalDataset, vqa_collate, vqa_eval_collate
from .ve import VeDataset, VeEvalDataset, ve_collate, ve_eval_collate
from .nlvr2 import (Nlvr2PairedDataset, Nlvr2PairedEvalDataset,
                    Nlvr2TripletDataset, Nlvr2TripletEvalDataset,
                    nlvr2_paired_collate, nlvr2_paired_eval_collate,
                    nlvr2_triplet_collate, nlvr2_triplet_eval_collate)
from .itm import (TokenBucketSamplerForItm, ItmDataset,
                  itm_collate, itm_ot_collate,
                  ItmRankDataset, ItmValDataset, ItmEvalDataset, ItmACCRankDataset,
                  ItmTestImg2TxtDataset, ItmTestDataset,
                  ItmRankDatasetHardNegFromImage,
                  ItmRankDatasetHardNegFromText,
                  itm_rank_collate, itm_val_collate, itm_eval_collate,
                  itm_rank_hn_collate,
                  EnItmDataset, enitm_collate, ItmNCEDataset, itm_nce_collate, itm_rank_batch_collate)
from .mlm import MlmDataset, mlm_collate
from .cem import CEMDataset, cem_collate, CEMNCEDataset, cem_nce_collate
from .mrm import MrfrDataset, MrcDataset, mrfr_collate, mrc_collate
from .vcr import (VcrTxtTokLmdb, VcrDataset, VcrEvalDataset,
                  vcr_collate, vcr_eval_collate)
from .caption import (CaptionDataset, CaptionGenDataset, caption_collate, CaptionEvalDataset, caption_eval_collate,
                      CaptionMMAutoGenDataset, auto_caption_collate)