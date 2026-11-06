import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from radius_sweep import measure_pair, prediction_kl, run_sweep
from summarize_radius_sweep import summarize


class RadiusSweepTests(unittest.TestCase):
    def setUp(self):
        self.threads=torch.get_num_threads(); torch.set_num_threads(2)
        torch.manual_seed(14)
        self.model=nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Dropout(.4), nn.Linear(5, 4)).double()
        self.params=list(self.model.parameters())
        self.a=[(torch.randn(2,3,dtype=torch.double), torch.tensor([0,1]))]
        self.b=[(torch.randn(2,3,dtype=torch.double), torch.tensor([2,3]))]
        self.loss=lambda x,y: F.cross_entropy(self.model(x),y)
        for p in self.params: p.grad=torch.randn_like(p)

    def tearDown(self): torch.set_num_threads(self.threads)

    def test_metrics_match_independent_reference_and_state_is_preserved(self):
        before=copy.deepcopy(self.model.state_dict())
        grads=[p.grad.clone() for p in self.params]
        rng=torch.get_rng_state().clone()
        rows=measure_pair(self.model,self.params,self.a,self.b,self.loss,self.model,[0,.1])
        for k,v in self.model.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        for p,g in zip(self.params,grads): torch.testing.assert_close(p.grad,g,rtol=0,atol=0)
        self.assertTrue(all(m.training for m in self.model.modules()))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for field in ('a_loss_change','b_loss_change','b_prediction_kl','relative_b_gradient_change'):
            self.assertEqual(rows[0][field],0)
        ref=copy.deepcopy(self.model).eval()
        rp=list(ref.parameters())
        ga=torch.autograd.grad(F.cross_entropy(ref(self.a[0][0]),self.a[0][1]),rp)
        norm=torch.cat([g.flatten() for g in ga]).norm()
        before_b=ref(self.b[0][0]).detach()
        before_loss=float(F.cross_entropy(before_b,self.b[0][1]))
        with torch.no_grad():
            for p,g in zip(rp,ga): p.add_(g,alpha=-.1/float(norm))
        after_b=ref(self.b[0][0]).detach()
        expected_loss=float(F.cross_entropy(after_b,self.b[0][1]))-before_loss
        expected_kl=float((before_b.log_softmax(-1).exp()*(before_b.log_softmax(-1)-after_b.log_softmax(-1))).sum(-1).mean())
        self.assertAlmostEqual(rows[1]['b_loss_change'],expected_loss,places=6)
        self.assertAlmostEqual(rows[1]['b_prediction_kl'],expected_kl,places=6)

    def test_exception_restores_parameters_and_dropout(self):
        original=[p.detach().clone() for p in self.params]
        def broken(x): raise RuntimeError('forced prediction failure')
        with self.assertRaisesRegex(RuntimeError,'forced prediction failure'):
            measure_pair(self.model,self.params,self.a,self.b,self.loss,broken,[.1])
        for p,w in zip(self.params,original): torch.testing.assert_close(p,w,rtol=0,atol=0)
        self.assertTrue(all(m.training for m in self.model.modules()))

    def test_chunked_kl_matches_full_distribution_with_mask(self):
        a=torch.randn(2,3,11); b=torch.randn_like(a); y=torch.tensor([[1,-1,2],[3,2,1]])
        value,n=prediction_kl(a,b,y,chunk=2)
        mask=y!=-1
        lp=a[mask].log_softmax(-1); lq=b[mask].log_softmax(-1)
        self.assertEqual(n,5)
        self.assertAlmostEqual(value,float((lp.exp()*(lp-lq)).sum(-1).mean()),places=6)

    def test_streamed_offsets_hashes_and_report(self):
        with tempfile.TemporaryDirectory() as folder:
            batches=iter([(x,y,1) for x,y in (self.b+self.a+self.a+self.b+self.b+self.a)])
            run_sweep(self.model,self.params,batches,self.loss,self.model,folder,2,1,1,[0,.1],1,{})
            path=Path(folder)/'radius_pairs.jsonl'
            rows=[json.loads(s) for s in path.read_text().splitlines()]
            self.assertEqual([r['pair'] for r in rows],[1,1,2,2])
            self.assertEqual(rows[0]['a_sha256'],rows[1]['a_sha256'])
            self.assertEqual(rows[0]['a_sha256'],rows[2]['b_sha256'])
            summary=summarize(path)
            self.assertEqual(summary[0]['pairs'],2)
            self.assertTrue((Path(folder)/'radius_report.html').exists())


if __name__=='__main__': unittest.main()
