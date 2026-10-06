import os, sys, types, unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapter"))
import replay_guard as G


class Obj:
    def __init__(self, **kw):
        self.return_logprob = kw.get("rl", False)
        self.logprob_start_len = kw.get("start", -1)
        self.return_hidden_states = kw.get("hs", False)


class T(unittest.TestCase):
    def test_rules(self):
        ids = list(range(100))
        G.check(Obj(rl=True, start=-1), ids, True)          # output logprobs only
        G.check(Obj(rl=True, start=None), ids, True)
        G.check(Obj(rl=True, start=100), ids, True)         # start at the prompt length
        G.check(Obj(rl=False, start=0), ids, True)          # logprob_start_len ignored without logprobs
        G.check(Obj(hs="last"), ids, True)
        G.check(Obj(rl=True, start=0), ids, False)          # bounded replay off: stock behaviour
        for o in (Obj(rl=True, start=0), Obj(rl=True, start=99), Obj(hs=True)):
            with self.assertRaises(ValueError):
                G.check(o, ids, True)

    def test_install_wraps_validate(self):
        calls = []

        class TM:
            def _validate_one_request(self, obj, input_ids):
                calls.append(obj)

        class GenerateReqInput(Obj):
            pass

        feats = types.SimpleNamespace(enable_decoder_swa_bounded_replay=True)
        mod = types.SimpleNamespace(TokenizerManager=TM, get_exec=lambda: types.SimpleNamespace(features=feats))
        G.install(mod)
        G.install(mod)  # idempotent
        tm = TM()
        tm._validate_one_request(GenerateReqInput(rl=True, start=-1), [1, 2, 3])
        with self.assertRaises(ValueError):
            tm._validate_one_request(GenerateReqInput(rl=True, start=0), [1, 2, 3])
        self.assertEqual(len(calls), 1)
        feats.enable_decoder_swa_bounded_replay = False
        tm._validate_one_request(GenerateReqInput(rl=True, start=0), [1, 2, 3])
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
