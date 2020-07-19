__author__ = 'tylin'
import re

from .tokenizer.ptbtokenizer import PTBTokenizer
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
from .spice.spice import Spice


class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}

        self.eval = {}
        self.evalImgs = []
        self.imgToEval = {}

        # Stitches back @-@ and BPE tokens
        self.cleanup_fn = lambda s: re.sub(
            '\s*@-@\s*', '-', s.replace("@@ ", "").replace("@@", ""))

        self.scorer_classes = {
            'bleu': (Bleu, ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            'meteor': (Meteor, "METEOR"),
            'rouge': (Rouge, "ROUGE_L"),
            'cider': (Cider, "CIDEr"),
            'spice': (Spice, "SPICE"),
        }

    def postprocess(self, caps):
        for cap in caps:
            cap['caption'] = self.cleanup_fn(cap['caption'])
        return caps

    def evaluate(self, verbose=True, metrics=None):
        imgIds = self.params['image_id']
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.postprocess(self.coco.imgToAnns[imgId])
            res[imgId] = self.postprocess(self.cocoRes.imgToAnns[imgId])

        # =================================================
        # Set up tokenizer and tokenize
        # =================================================
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        if metrics is None:
            # Use all scorers
            metrics = self.scorer_classes.keys()

        scorers = [self.scorer_classes[k] for k in metrics]
        scorers = [(klass(), strs) for (klass, strs) in scorers]

        # =================================================
        # Compute scores
        # =================================================
        score_dict = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, list(gts.keys()), m)
                    if verbose:
                        print("{}: {:.3f}".format(m, sc))
                    score_dict[m] = sc
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, list(gts.keys()), method)
                if verbose:
                    print("{}: {:.3f}".format(method, score))
                score_dict[method] = score
        self.setEvalImgs()

        return score_dict

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if imgId not in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval_ for imgId, eval_ in self.imgToEval.items()]
