"""`exploration.affinity_table`: the hand table, the model, or neither.

The review asked for the affinity prior to come from the LLM for every class
rather than a hand-written table for some and the model for the rest. These pin
the three settings and that the default is untouched.
"""
from types import SimpleNamespace

from osg.graph.priors import CONTAINER_AFFINITY, affinity_scores
from osg.llm.affinity import SoleAffinity
from osg.pipeline.beliefs import build_affinity_prior


class _Model:
    def __init__(self, answer):
        self.answer = answer
        self.asked = []
        self.n_calls = self.n_errors = 0
        self.grounded = None

    def __call__(self, target):
        self.asked.append(target)
        return self.answer

    def ground(self, present):
        self.grounded = set(present)


def test_by_default_the_table_wins_where_it_has_the_class():
    model = _Model(["bed"])
    scores = affinity_scores("bowl", source=model)
    assert next(iter(scores)) == CONTAINER_AFFINITY["bowl"][0]
    assert model.asked == []


def test_a_sole_source_ranks_a_class_the_table_has():
    model = _Model(["cabinet", "shelf"])
    scores = affinity_scores("bowl", source=SoleAffinity(model))
    assert list(scores) == ["cabinet", "shelf"] and model.asked == ["bowl"]


def test_no_table_and_no_model_is_a_flat_prior():
    assert affinity_scores("bowl", source=SoleAffinity(None)) == {}


def test_the_wrapper_passes_grounding_and_counters_through():
    model = _Model(["table"])
    model.n_calls, model.n_errors = 3, 1
    sole = SoleAffinity(model)
    sole.ground({"table", "cabinet"})
    assert model.grounded == {"table", "cabinet"}
    assert (sole.n_calls, sole.n_errors) == (3, 1)
    flat = SoleAffinity(None)
    flat.ground({"table"})
    assert (flat.n_calls, flat.n_errors) == (0, 0)


def _cfg(llm, table):
    return SimpleNamespace(
        exploration=SimpleNamespace(affinity_llm=llm, affinity_table=table, affinity_cache=""),
        llm=SimpleNamespace(api_key="", base_url="", text_model="", timeout_s=1.0,
                            max_image_px=0, send_response_format=False),
    )


def test_the_four_settings_build_what_they_say():
    from osg.llm.affinity import AffinityProvider

    assert build_affinity_prior(_cfg(False, True)) is None          # shipped default
    flat = build_affinity_prior(_cfg(False, False))
    assert isinstance(flat, SoleAffinity) and flat.provider is None
    hybrid = build_affinity_prior(_cfg(True, True))
    assert isinstance(hybrid, AffinityProvider)
    model_only = build_affinity_prior(_cfg(True, False))
    assert isinstance(model_only, SoleAffinity)
    assert isinstance(model_only.provider, AffinityProvider)
