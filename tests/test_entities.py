from core.entities import CLASS, DRUG, ENZYME


def ids(linker, text):
    return [(e.canonical, e.kind) for e in linker.link(text)]


def test_brand_names_normalise_to_generics(linker):
    assert ids(linker, "Can I take Viagra with Nitrostat?") == [
        ("sildenafil", DRUG), ("nitroglycerin", DRUG),
    ]


def test_longest_match_wins(linker):
    # "isosorbide mononitrate" must not be read as the class "nitrate".
    assert ids(linker, "isosorbide mononitrate tablets") == [("isosorbide mononitrate", DRUG)]


def test_no_match_inside_longer_words(linker):
    assert ids(linker, "prewarfarinization and nitrated compounds") == []


def test_enzyme_spellings_are_unified(linker):
    for text in ("CYP3A4", "CYP 3A4", "cytochrome P450 3A4", "CYP3A", "CYP3A4/5"):
        assert ids(linker, f"metabolised by {text}.") == [("CYP3A4", ENZYME)], text
    assert ids(linker, "a P-glycoprotein substrate") == [("P-gp", ENZYME)]


def test_classes_and_lay_terms(linker):
    assert ids(linker, "MAOIs and blood thinners") == [
        ("MAO inhibitors", CLASS), ("anticoagulants", CLASS),
    ]
    assert "nitrates" in linker.classes_of["nitroglycerin"]


def test_query_normalisation_adds_canonical_names(linker):
    assert (
        linker.normalize_query("Is Coumadin safe with an MAOI?")
        == "Is Coumadin (warfarin) safe with an MAOI (MAO inhibitors)?"
    )
    assert linker.normalize_query("warfarin and aspirin") == "warfarin and aspirin"
