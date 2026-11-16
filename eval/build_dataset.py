"""
Evaluation set definition.

Each answerable query lists its gold evidence as (label, terms[, section]) rules: a passage
is relevant if it comes from that drug's label and contains any of the terms (and lies in
the given section, if one is named). The rules are resolved against the chunked corpus at
evaluation time, so gold labels survive re-chunking and never depend on what any retriever
returned. Running this file validates every rule and writes eval/dataset.jsonl.

Categories
    direct        both drugs named by generic name; the interaction is stated in a label
    brand         same kind of question, asked with brand names
    drug_class    one side is a drug class or lay term ("blood thinners")
    mechanism     asks for the enzyme / transporter pathway behind an interaction
    single_drug   facts about one drug (boxed warning, contraindications, metabolism)

Unanswerable (the system should refuse)
    neg_fictional      invented drug names, alone or paired with a real drug
    neg_out_of_corpus  real drugs that have no label in the corpus and are never mentioned
    neg_off_topic      nothing to do with pharmacology
    neg_unsupported    real corpus drugs, but the labels do not contain the answer;
                       includes drug pairs that no label mentions together

Splits: every third query of each category is "dev" (used to calibrate the gate threshold
and to debug); the rest are "test" (reported).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.chunking import ClinicalChunk  # noqa: E402

Rule = Tuple  # (doc_id, [terms]) or (doc_id, [terms], section)

NIT = ["nitrate", "nitroglycerin"]
PDE = ["phosphodiesterase", "pde"]
NSAID = ["nsaid", "nonsteroidal", "non-steroidal"]

ANSWERABLE: Dict[str, List[Tuple[str, List[Rule]]]] = {
    "direct": [
        ("Is sildenafil contraindicated with nitroglycerin?", [("sildenafil", NIT), ("nitroglycerin", ["sildenafil"] + PDE)]),
        ("Why is clarithromycin contraindicated with simvastatin?", [("clarithromycin", ["simvastatin"]), ("simvastatin", ["clarithromycin"])]),
        ("What is the interaction between fluconazole and warfarin?", [("fluconazole", ["warfarin", "coumarin", "prothrombin"]), ("warfarin", ["fluconazole"])]),
        ("Can amiodarone be given with warfarin?", [("amiodarone", ["warfarin"]), ("warfarin", ["amiodarone"])]),
        ("Does rifampin reduce the effect of apixaban?", [("apixaban", ["rifampin"])]),
        ("What happens when ketoconazole is given with rivaroxaban?", [("rivaroxaban", ["ketoconazole"])]),
        ("Does verapamil increase digoxin levels?", [("digoxin", ["verapamil"]), ("verapamil", ["digoxin"])]),
        ("Can clopidogrel be taken with omeprazole?", [("clopidogrel", ["omeprazole"]), ("omeprazole", ["clopidogrel"])]),
        ("What is the effect of gemfibrozil on repaglinide?", [("repaglinide", ["gemfibrozil"]), ("gemfibrozil", ["repaglinide"])]),
        ("Is tizanidine contraindicated with ciprofloxacin?", [("tizanidine", ["ciprofloxacin"]), ("ciprofloxacin", ["tizanidine"])]),
        ("Can lithium be combined with hydrochlorothiazide?", [("lithium", ["thiazide", "diuretic"]), ("hydrochlorothiazide", ["lithium"])]),
        ("Does carbamazepine lower lamotrigine levels?", [("lamotrigine", ["carbamazepine"])]),
        ("Does spironolactone interact with lisinopril?", [("spironolactone", ["ace inhibitor", "angiotensin", "lisinopril"]), ("lisinopril", ["spironolactone", "potassium-sparing", "potassium sparing"])]),
        ("What is the risk of combining cyclosporine with simvastatin?", [("simvastatin", ["cyclosporine"]), ("cyclosporine", ["simvastatin", "statin"])]),
        ("Can allopurinol be given with azathioprine?", [("azathioprine", ["allopurinol"]), ("allopurinol", ["azathioprine"])]),
        ("Does fluoxetine affect metoprolol?", [("metoprolol", ["fluoxetine", "cyp2d6"])]),
        ("Is colchicine safe with clarithromycin?", [("colchicine", ["clarithromycin"]), ("clarithromycin", ["colchicine"])]),
        ("Does itraconazole increase sildenafil exposure?", [("sildenafil", ["itraconazole"])]),
        ("Can theophylline be taken with ciprofloxacin?", [("theophylline", ["ciprofloxacin"]), ("ciprofloxacin", ["theophylline"])]),
        ("What happens if phenytoin is taken with voriconazole?", [("voriconazole", ["phenytoin"]), ("phenytoin", ["voriconazole"])]),
        ("Can tacrolimus be combined with fluconazole?", [("tacrolimus", ["fluconazole"]), ("fluconazole", ["tacrolimus"])]),
        ("Is ranolazine contraindicated with ketoconazole?", [("ranolazine", ["ketoconazole"])]),
        ("Does paroxetine reduce the effectiveness of tamoxifen?", [("tamoxifen", ["paroxetine"]), ("paroxetine", ["tamoxifen"])]),
        ("Can dofetilide be used with verapamil?", [("dofetilide", ["verapamil"])]),
        ("Is riociguat contraindicated with sildenafil?", [("riociguat", ["sildenafil"] + PDE), ("sildenafil", ["riociguat"])]),
        ("Does diltiazem interact with ivabradine?", [("ivabradine", ["diltiazem"])]),
        ("Can methadone be taken with rifampin?", [("methadone", ["rifampin"])]),
        ("Is eplerenone contraindicated with ketoconazole?", [("eplerenone", ["ketoconazole"])]),
        ("Can linezolid be used with sertraline?", [("linezolid", ["sertraline", "serotonin"]), ("sertraline", ["linezolid"])]),
    ],
    "brand": [
        ("Can I take Viagra with Nitrostat?", [("sildenafil", NIT), ("nitroglycerin", ["sildenafil"] + PDE)]),
        ("Is Biaxin safe with Zocor?", [("clarithromycin", ["simvastatin"]), ("simvastatin", ["clarithromycin"])]),
        ("Does Diflucan interact with Coumadin?", [("fluconazole", ["warfarin", "coumarin", "prothrombin"]), ("warfarin", ["fluconazole"])]),
        ("Can Plavix and Prilosec be taken together?", [("clopidogrel", ["omeprazole"]), ("omeprazole", ["clopidogrel"])]),
        ("Is it okay to use Cialis while on Imdur?", [("tadalafil", NIT)]),
        ("Does Lanoxin interact with Pacerone?", [("digoxin", ["amiodarone"]), ("amiodarone", ["digoxin"])]),
        ("Can Xarelto be combined with Norvir?", [("rivaroxaban", ["ritonavir"])]),
        ("Is Tegretol a problem with Lamictal?", [("lamotrigine", ["carbamazepine"])]),
        ("Zanaflex and Cipro together?", [("tizanidine", ["ciprofloxacin"]), ("ciprofloxacin", ["tizanidine"])]),
        ("Can I take Ultram with Nardil?", [("tramadol", ["mao", "phenelzine"])]),
        ("Does Sporanox raise Halcion levels?", [("triazolam", ["itraconazole"]), ("itraconazole", ["triazolam"])]),
        ("Is Lipitor affected by Sporanox?", [("atorvastatin", ["itraconazole"]), ("itraconazole", ["atorvastatin"])]),
        ("Is Multaq safe with Lanoxin?", [("dronedarone", ["digoxin"]), ("digoxin", ["dronedarone"])]),
        ("Eliquis and Rifadin interaction", [("apixaban", ["rifampin"])]),
        ("Can Prograf be given with Vfend?", [("tacrolimus", ["voriconazole"]), ("voriconazole", ["tacrolimus"])]),
        ("Can Lithobid be taken with Motrin?", [("lithium", NSAID + ["ibuprofen"]), ("ibuprofen", ["lithium"])]),
        ("Is Zyvox safe with Zoloft?", [("linezolid", ["sertraline", "serotonin"]), ("sertraline", ["linezolid"])]),
        ("Is Colcrys dangerous with Neoral?", [("colchicine", ["cyclosporine"]), ("cyclosporine", ["colchicine"])]),
    ],
    "drug_class": [
        ("Why should patients on nitrates avoid tadalafil?", [("tadalafil", NIT)]),
        ("Can an SSRI be combined with tramadol?", [("tramadol", ["ssri", "serotonin reuptake", "serotonergic"])]),
        ("Are NSAIDs a problem for patients taking warfarin?", [("warfarin", NSAID)]),
        ("Is it dangerous to take an opioid with a benzodiazepine?", [(d, ["benzodiazepine"]) for d in ("oxycodone", "morphine", "fentanyl", "methadone")] + [(d, ["opioid"]) for d in ("alprazolam", "diazepam", "lorazepam", "clonazepam")]),
        ("Can sumatriptan be taken with an MAO inhibitor?", [("sumatriptan", ["mao"])]),
        ("Do ACE inhibitors interact with lithium?", [("lithium", ["ace inhibitor", "angiotensin"]), ("lisinopril", ["lithium"]), ("enalapril", ["lithium"])]),
        ("Why is methotrexate risky with NSAIDs?", [("methotrexate", NSAID)]),
        ("Can beta-blockers be combined with verapamil?", [("verapamil", ["beta-blocker", "beta blocker", "beta-adrenergic"])]),
        ("Is it safe to use potassium-sparing diuretics with lisinopril?", [("lisinopril", ["potassium-sparing", "potassium sparing"])]),
        ("Do proton pump inhibitors affect atazanavir?", [("atazanavir", ["proton pump", "proton-pump"])]),
        ("Should statins be avoided with gemfibrozil?", [("gemfibrozil", ["statin", "hmg-coa"]), ("simvastatin", ["gemfibrozil"])]),
        ("Can macrolide antibiotics be used with colchicine?", [("colchicine", ["macrolide", "clarithromycin", "erythromycin"])]),
        ("Are azole antifungals a concern with tacrolimus?", [("tacrolimus", ["azole", "antifungal"])]),
        ("Is linezolid safe with SSRIs?", [("linezolid", ["serotonin", "ssri"])]),
        ("Can blood thinners be taken with Motrin?", [("ibuprofen", ["warfarin", "anticoagulant"]), ("warfarin", NSAID + ["ibuprofen"])]),
        ("Can a patient on Isordil take Levitra?", [("vardenafil", NIT), ("isosorbide_dinitrate", ["vardenafil"] + PDE)]),
        ("Should fluoroquinolones be avoided in patients on tizanidine?", [("tizanidine", ["fluoroquinolone", "ciprofloxacin"])]),
        ("Is it a problem to use alpha-blockers with Viagra?", [("sildenafil", ["alpha-blocker", "alpha blocker"])]),
    ],
    "mechanism": [
        ("How does clarithromycin affect simvastatin clearance through liver enzymes?", [("simvastatin", ["cyp3a4"]), ("clarithromycin", ["simvastatin", "hmg-coa"])]),
        ("Which enzyme pathway explains why rifampin lowers apixaban exposure?", [("apixaban", ["rifampin", "inducer"])]),
        ("Through which enzyme does fluconazole raise warfarin levels?", [("warfarin", ["cyp2c9"]), ("fluconazole", ["cyp2c9"])]),
        ("Why would ketoconazole raise lurasidone concentrations?", [("lurasidone", ["ketoconazole", "cyp3a4"])]),
        ("What transporter is involved in the interaction between digoxin and cyclosporine?", [("digoxin", ["p-glycoprotein", "p-gp"]), ("cyclosporine", ["digoxin"])]),
        ("Why do strong CYP2D6 inhibitors such as paroxetine matter for atomoxetine?", [("atomoxetine", ["cyp2d6", "paroxetine"])]),
        ("Explain why carbamazepine can make ticagrelor less effective.", [("ticagrelor", ["carbamazepine", "inducer"])]),
        ("What is the mechanism behind the interaction of erythromycin with buspirone?", [("buspirone", ["erythromycin", "cyp3a4"])]),
        ("How do CYP3A4 inducers such as phenytoin affect everolimus?", [("everolimus", ["phenytoin", "inducer"])]),
        ("Why is tramadol exposure altered by fluoxetine?", [("tramadol", ["cyp2d6", "fluoxetine"])]),
        ("What is the mechanism of the interaction between ritonavir and midazolam?", [("midazolam", ["ritonavir", "protease inhibitor", "cyp3a4"]), ("ritonavir", ["midazolam"])]),
        ("Which metabolic enzyme links voriconazole and sirolimus?", [("sirolimus", ["voriconazole", "cyp3a4"]), ("voriconazole", ["sirolimus"])]),
        ("Why does bupropion increase metoprolol concentrations?", [("bupropion", ["metoprolol", "cyp2d6"]), ("metoprolol", ["cyp2d6"])]),
        ("How does diltiazem change exposure to triazolam?", [("triazolam", ["diltiazem", "cyp3a"]), ("diltiazem", ["triazolam"])]),
        ("What pathway makes phenobarbital reduce dolutegravir levels?", [("dolutegravir", ["phenobarbital"])]),
        ("Why does amiodarone increase flecainide plasma levels?", [("flecainide", ["amiodarone"]), ("amiodarone", ["flecainide"])]),
        ("Explain how fluvoxamine affects tizanidine through CYP1A2.", [("tizanidine", ["fluvoxamine", "cyp1a2"])]),
        ("How does cyclosporine alter rosuvastatin exposure?", [("rosuvastatin", ["cyclosporine"])]),
    ],
    "single_drug": [
        ("What is the boxed warning for warfarin?", [("warfarin", [""], "BOXED WARNING")]),
        ("Which enzyme is mainly responsible for the metabolism of S-warfarin?", [("warfarin", ["cyp2c9"])]),
        ("What are the contraindications for metformin?", [("metformin", [""], "CONTRAINDICATIONS")]),
        ("How is sildenafil metabolized?", [("sildenafil", ["cyp3a4"])]),
        ("What does the boxed warning on clozapine say?", [("clozapine", [""], "BOXED WARNING")]),
        ("What is the mechanism of action of apixaban?", [("apixaban", [""], "MECHANISM OF ACTION")]),
        ("What is dabigatran indicated for?", [("dabigatran", [""], "INDICATIONS AND USAGE")]),
        ("Who should not take dronedarone?", [("dronedarone", [""], "CONTRAINDICATIONS")]),
        ("How does empagliflozin work?", [("empagliflozin", [""], "MECHANISM OF ACTION")]),
        ("Is there a boxed warning for amiodarone?", [("amiodarone", [""], "BOXED WARNING")]),
        ("What are the contraindications of Tikosyn?", [("dofetilide", [""], "CONTRAINDICATIONS")]),
        ("Which transporter is digoxin a substrate of?", [("digoxin", ["p-glycoprotein", "p-gp"])]),
    ],
}

UNANSWERABLE: Dict[str, List[str]] = {
    "neg_fictional": [
        "Can Xanodrol be prescribed alongside Gloxipen?",
        "What is the interaction between Zynorapid and warfarin?",
        "Is Plaxirastine safe to take with simvastatin?",
        "Does Torvamil interact with metoprolol?",
        "Can Nexobrutinib be combined with Zephyroline?",
        "What are the contraindications of Floraxan?",
        "Is Quentarix metabolized by CYP3A4?",
        "Does Virobacterol raise digoxin levels?",
        "Can I take Omnipril together with lisinopril?",
        "What is the boxed warning for Chronoflex?",
        "What is the interaction between phymocline and veltoradine?",
        "Is morvastatin contraindicated with clarithromycin?",
        "Does dexalozole inhibit CYP2C9?",
        "How should the dose of Kryptex be adjusted with rifampin?",
    ],
    "neg_out_of_corpus": [
        "Does pembrolizumab interact with warfarin?",
        "Can adalimumab be given with methotrexate?",
        "What are the contraindications of nivolumab?",
        "Is lenalidomide metabolized by CYP3A4?",
        "Does apremilast interact with rifampin?",
        "What is the boxed warning for ustekinumab?",
        "Can dupilumab be combined with prednisone?",
        "Does ocrelizumab interact with fluoxetine?",
        "How is remdesivir metabolized?",
        "Is bevacizumab safe with carvedilol?",
    ],
    "neg_off_topic": [
        "What is the capital of France?",
        "Write a Python function that reverses a linked list.",
        "Who won the football world cup in 2018?",
        "What will the weather be like in Mumbai tomorrow?",
        "How do I change a flat tyre on a bicycle?",
        "Explain the rules of chess castling.",
        "What is the boiling point of water at sea level?",
        "Recommend a good laptop for programming.",
        "Translate 'good morning' into Spanish.",
        "What is the square root of 144?",
    ],
    "neg_unsupported": [
        "What is the retail price of atorvastatin in the United States?",
        "Is lisinopril safe for dogs?",
        "Who discovered warfarin and in which year?",
        "Does metformin interact with moon rocks?",
        "Which company's share price rose after apixaban was approved?",
        "Can sertraline make someone better at chess?",
        "How many people in Canada take amlodipine?",
        "In what year did simvastatin lose patent protection?",
        "Is omeprazole suitable for a vegan diet?",
        "Which pharmacy near me has clopidogrel in stock?",
    ],
}

# Drug pairs for which no label in the corpus mentions the other drug or its class.
# Validated below; a pair that fails validation is dropped rather than kept as a negative.
NO_EVIDENCE_PAIRS: List[Tuple[str, str]] = [
    ("amoxicillin", "levothyroxine"),
    ("cephalexin", "sertraline"),
    ("montelukast", "tamsulosin"),
    ("finasteride", "metformin"),
    ("oseltamivir", "donepezil"),
    ("acyclovir", "mirabegron"),
    ("memantine", "cephalexin"),
    ("famotidine", "finasteride"),
    ("valacyclovir", "alendronate"),
    ("nitrofurantoin", "montelukast"),
]


def gold_chunk_ids(rules: Sequence[Rule], chunks: Sequence[ClinicalChunk]) -> List[str]:
    """Resolves gold rules to passage ids."""
    gold: List[str] = []
    for rule in rules:
        doc_id, terms = rule[0], rule[1]
        section = rule[2] if len(rule) > 2 else None
        for chunk in chunks:
            if chunk.doc_id != doc_id or (section and chunk.section != section):
                continue
            text = chunk.content.lower()
            if any(term in text for term in terms):
                gold.append(chunk.chunk_id)
    return list(dict.fromkeys(gold))


def main() -> None:
    from core.entities import EntityLinker
    from core.pipeline import load_chunks
    from core.sparse import BiomedicalBM25, tokenize

    chunks = load_chunks(ROOT / "data/index/chunks.jsonl")
    by_doc: Dict[str, List[ClinicalChunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)
    linker = EntityLinker()
    bm25 = BiomedicalBM25()
    bm25.index_chunks(chunks)

    rows, problems = [], []

    def split_for(index: int) -> str:
        return "dev" if index % 3 == 0 else "test"

    for category, items in ANSWERABLE.items():
        for i, (query, rules) in enumerate(items):
            for rule in rules:
                if rule[0] not in by_doc:
                    problems.append(f"[{category}] unknown label '{rule[0]}': {query}")
                elif not gold_chunk_ids([rule], by_doc[rule[0]]):
                    problems.append(f"[{category}] rule matches nothing {rule}: {query}")
            rules = [r for r in rules if r[0] in by_doc and gold_chunk_ids([r], by_doc[r[0]])]
            if not rules:
                continue
            rows.append({
                "category": category, "query": query, "answerable": True, "split": split_for(i),
                "gold": [
                    {"doc": r[0], "terms": r[1], **({"section": r[2]} if len(r) > 2 else {})}
                    for r in rules
                ],
            })

    for category, queries in UNANSWERABLE.items():
        for i, query in enumerate(queries):
            if category == "neg_out_of_corpus":
                drug = query.split()[1] if query.split()[0] in ("Does", "Can", "Is") else query.rstrip("?").split()[-1]
                candidates = [w for w in query.rstrip("?").split() if w.endswith(("mab", "mide", "last", "vir"))]
                for word in candidates or [drug]:
                    if any(t in bm25.vocabulary for t in tokenize(word)):
                        problems.append(f"[{category}] '{word}' occurs in the corpus: {query}")
            rows.append({"category": category, "query": query, "answerable": False, "split": split_for(i), "gold": []})

    kept_pairs = 0
    for a, b in NO_EVIDENCE_PAIRS:
        def mentions(label: str, other: str) -> bool:
            names = [other] + [a.lower() for c in linker.classes_of.get(other, []) for a in [c]]
            return any(
                e.canonical in names
                for chunk in by_doc.get(label.replace(" ", "_"), [])
                for e in linker.link(chunk.content)
            )
        if a.replace(" ", "_") not in by_doc or b.replace(" ", "_") not in by_doc:
            problems.append(f"[neg_unsupported] label missing for pair {a}/{b}")
        elif mentions(a, b) or mentions(b, a):
            problems.append(f"[neg_unsupported] pair IS mentioned in a label, dropped: {a}/{b}")
        else:
            rows.append({
                "category": "neg_unsupported",
                "query": f"Is there a documented interaction between {a} and {b}?",
                "answerable": False, "split": split_for(kept_pairs + 1), "gold": [],
            })
            kept_pairs += 1

    for n, row in enumerate(rows, start=1):
        row["id"] = f"q{n:03d}"
    out = ROOT / "eval/dataset.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({k: row[k] for k in ("id", "category", "split", "query", "answerable", "gold")}) + "\n")

    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    gold_sizes = [len(gold_chunk_ids([(g["doc"], g["terms"], g.get("section")) if g.get("section") else (g["doc"], g["terms"]) for g in r["gold"]], chunks)) for r in rows if r["answerable"]]
    print(f"Wrote {len(rows)} queries to {out}")
    print("per category:", counts)
    print("split:", {s: sum(r['split'] == s for r in rows) for s in ('dev', 'test')})
    print(f"gold passages per answerable query: min={min(gold_sizes)} median={sorted(gold_sizes)[len(gold_sizes)//2]} max={max(gold_sizes)}")
    for problem in problems:
        print("PROBLEM:", problem)


if __name__ == "__main__":
    main()
