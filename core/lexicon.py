"""
Curated pharmacology lexicon.

Three small hand-maintained vocabularies drive both ingestion and entity linking:

* ``DRUGS``         generic name -> well-known US brand names. This is the ingestion
                    target list (one openFDA label is fetched per generic) and the
                    brand -> generic normalisation table ("Coumadin" -> warfarin).
* ``DRUG_CLASSES``  class id -> surface forms used in label text + member generics.
                    Labels usually state interactions at class level ("contraindicated
                    with organic nitrates"), so class nodes are what make multi-hop
                    graph retrieval work.
* ``ENZYMES``       metabolic enzymes / transporters and their surface forms.

Everything else in the knowledge graph is extracted from label text; nothing about
individual drug-drug interactions is hard-coded here.
"""

from __future__ import annotations

from typing import Dict, List

# --------------------------------------------------------------------------- #
# Generic name -> brand names
# --------------------------------------------------------------------------- #
DRUGS: Dict[str, List[str]] = {
    # Lipids
    "atorvastatin": ["Lipitor"],
    "simvastatin": ["Zocor"],
    "rosuvastatin": ["Crestor"],
    "pravastatin": ["Pravachol"],
    "lovastatin": ["Mevacor", "Altoprev"],
    "pitavastatin": ["Livalo"],
    "fluvastatin": ["Lescol"],
    "ezetimibe": ["Zetia"],
    "fenofibrate": ["Tricor"],
    "gemfibrozil": ["Lopid"],
    # ACE inhibitors / ARBs
    "lisinopril": ["Zestril", "Prinivil"],
    "enalapril": ["Vasotec"],
    "ramipril": ["Altace"],
    "benazepril": ["Lotensin"],
    "captopril": ["Capoten"],
    "quinapril": ["Accupril"],
    "losartan": ["Cozaar"],
    "valsartan": ["Diovan"],
    "irbesartan": ["Avapro"],
    "olmesartan": ["Benicar"],
    "telmisartan": ["Micardis"],
    "candesartan": ["Atacand"],
    # Calcium channel blockers
    "amlodipine": ["Norvasc"],
    "nifedipine": ["Procardia", "Adalat"],
    "diltiazem": ["Cardizem"],
    "verapamil": ["Calan", "Verelan"],
    "felodipine": ["Plendil"],
    # Beta blockers
    "metoprolol": ["Lopressor", "Toprol-XL"],
    "atenolol": ["Tenormin"],
    "propranolol": ["Inderal"],
    "carvedilol": ["Coreg"],
    "bisoprolol": ["Zebeta"],
    "nebivolol": ["Bystolic"],
    "labetalol": ["Trandate"],
    # Diuretics
    "hydrochlorothiazide": ["Microzide"],
    "chlorthalidone": ["Thalitone"],
    "furosemide": ["Lasix"],
    "bumetanide": ["Bumex"],
    "torsemide": ["Demadex"],
    "spironolactone": ["Aldactone"],
    "eplerenone": ["Inspra"],
    "triamterene": ["Dyrenium"],
    "amiloride": ["Midamor"],
    # Other cardiovascular
    "clonidine": ["Catapres"],
    "hydralazine": ["Apresoline"],
    "isosorbide mononitrate": ["Imdur"],
    "isosorbide dinitrate": ["Isordil"],
    "nitroglycerin": ["Nitrostat", "Nitro-Dur"],
    "digoxin": ["Lanoxin"],
    "amiodarone": ["Pacerone", "Cordarone"],
    "dronedarone": ["Multaq"],
    "sotalol": ["Betapace"],
    "flecainide": ["Tambocor"],
    "propafenone": ["Rythmol"],
    "dofetilide": ["Tikosyn"],
    "ranolazine": ["Ranexa"],
    "ivabradine": ["Corlanor"],
    # Anticoagulants / antiplatelets
    "warfarin": ["Coumadin", "Jantoven"],
    "apixaban": ["Eliquis"],
    "rivaroxaban": ["Xarelto"],
    "dabigatran": ["Pradaxa"],
    "edoxaban": ["Savaysa"],
    "enoxaparin": ["Lovenox"],
    "clopidogrel": ["Plavix"],
    "prasugrel": ["Effient"],
    "ticagrelor": ["Brilinta"],
    "cilostazol": ["Pletal"],
    "dipyridamole": ["Persantine"],
    # PDE5 inhibitors / pulmonary hypertension
    "sildenafil": ["Viagra", "Revatio"],
    "tadalafil": ["Cialis", "Adcirca"],
    "vardenafil": ["Levitra"],
    "riociguat": ["Adempas"],
    "bosentan": ["Tracleer"],
    "ambrisentan": ["Letairis"],
    # Diabetes
    "metformin": ["Glucophage"],
    "glipizide": ["Glucotrol"],
    "glyburide": ["Glynase"],
    "glimepiride": ["Amaryl"],
    "pioglitazone": ["Actos"],
    "sitagliptin": ["Januvia"],
    "saxagliptin": ["Onglyza"],
    "linagliptin": ["Tradjenta"],
    "empagliflozin": ["Jardiance"],
    "dapagliflozin": ["Farxiga"],
    "canagliflozin": ["Invokana"],
    "liraglutide": ["Victoza", "Saxenda"],
    "semaglutide": ["Ozempic", "Wegovy", "Rybelsus"],
    "dulaglutide": ["Trulicity"],
    "repaglinide": ["Prandin"],
    "nateglinide": ["Starlix"],
    # Antibacterials
    "clarithromycin": ["Biaxin"],
    "erythromycin": ["Ery-Tab", "Erythrocin"],
    "azithromycin": ["Zithromax"],
    "ciprofloxacin": ["Cipro"],
    "levofloxacin": ["Levaquin"],
    "moxifloxacin": ["Avelox"],
    "amoxicillin": ["Amoxil"],
    "doxycycline": ["Vibramycin"],
    "minocycline": ["Minocin"],
    "cephalexin": ["Keflex"],
    "nitrofurantoin": ["Macrobid"],
    "metronidazole": ["Flagyl"],
    "linezolid": ["Zyvox"],
    "rifampin": ["Rifadin"],
    "rifabutin": ["Mycobutin"],
    "isoniazid": [],
    "clindamycin": ["Cleocin"],
    "trimethoprim": ["Primsol"],
    # Antifungals
    "fluconazole": ["Diflucan"],
    "itraconazole": ["Sporanox"],
    "ketoconazole": ["Nizoral"],
    "voriconazole": ["Vfend"],
    "posaconazole": ["Noxafil"],
    "terbinafine": ["Lamisil"],
    # Antivirals
    "acyclovir": ["Zovirax"],
    "valacyclovir": ["Valtrex"],
    "oseltamivir": ["Tamiflu"],
    "ritonavir": ["Norvir"],
    "atazanavir": ["Reyataz"],
    "darunavir": ["Prezista"],
    "efavirenz": ["Sustiva"],
    "dolutegravir": ["Tivicay"],
    "raltegravir": ["Isentress"],
    "hydroxychloroquine": ["Plaquenil"],
    # Antidepressants
    "fluoxetine": ["Prozac"],
    "sertraline": ["Zoloft"],
    "paroxetine": ["Paxil"],
    "citalopram": ["Celexa"],
    "escitalopram": ["Lexapro"],
    "fluvoxamine": ["Luvox"],
    "venlafaxine": ["Effexor"],
    "desvenlafaxine": ["Pristiq"],
    "duloxetine": ["Cymbalta"],
    "bupropion": ["Wellbutrin", "Zyban"],
    "mirtazapine": ["Remeron"],
    "trazodone": ["Desyrel"],
    "amitriptyline": ["Elavil"],
    "nortriptyline": ["Pamelor"],
    "imipramine": ["Tofranil"],
    "phenelzine": ["Nardil"],
    "tranylcypromine": ["Parnate"],
    "selegiline": ["Emsam", "Eldepryl"],
    # Antipsychotics / mood stabilisers
    "lithium": ["Lithobid"],
    "quetiapine": ["Seroquel"],
    "olanzapine": ["Zyprexa"],
    "risperidone": ["Risperdal"],
    "aripiprazole": ["Abilify"],
    "ziprasidone": ["Geodon"],
    "haloperidol": ["Haldol"],
    "clozapine": ["Clozaril"],
    "lurasidone": ["Latuda"],
    # Sedatives / anxiolytics
    "alprazolam": ["Xanax"],
    "lorazepam": ["Ativan"],
    "clonazepam": ["Klonopin"],
    "diazepam": ["Valium"],
    "midazolam": ["Versed"],
    "triazolam": ["Halcion"],
    "zolpidem": ["Ambien"],
    "eszopiclone": ["Lunesta"],
    "buspirone": ["Buspar"],
    # Anticonvulsants
    "carbamazepine": ["Tegretol"],
    "phenytoin": ["Dilantin"],
    "valproic acid": ["Depakene"],
    "divalproex": ["Depakote"],
    "lamotrigine": ["Lamictal"],
    "levetiracetam": ["Keppra"],
    "topiramate": ["Topamax"],
    "oxcarbazepine": ["Trileptal"],
    "gabapentin": ["Neurontin"],
    "pregabalin": ["Lyrica"],
    "phenobarbital": [],
    # Other CNS
    "methylphenidate": ["Ritalin", "Concerta"],
    "atomoxetine": ["Strattera"],
    "donepezil": ["Aricept"],
    "memantine": ["Namenda"],
    "sumatriptan": ["Imitrex"],
    "rizatriptan": ["Maxalt"],
    "ropinirole": ["Requip"],
    "pramipexole": ["Mirapex"],
    "tizanidine": ["Zanaflex"],
    "cyclobenzaprine": ["Flexeril"],
    "baclofen": ["Lioresal"],
    # Opioids and related
    "tramadol": ["Ultram"],
    "oxycodone": ["OxyContin", "Roxicodone"],
    "hydrocodone": ["Hysingla"],
    "morphine": ["MS Contin"],
    "fentanyl": ["Duragesic"],
    "methadone": ["Dolophine"],
    "buprenorphine": ["Subutex", "Butrans"],
    "tapentadol": ["Nucynta"],
    "naltrexone": ["Vivitrol"],
    # NSAIDs / gout
    "ibuprofen": ["Motrin"],
    "naproxen": ["Naprosyn"],
    "celecoxib": ["Celebrex"],
    "meloxicam": ["Mobic"],
    "diclofenac": ["Voltaren"],
    "indomethacin": ["Indocin"],
    "ketorolac": ["Toradol"],
    "etodolac": ["Lodine"],
    "colchicine": ["Colcrys"],
    "allopurinol": ["Zyloprim"],
    "febuxostat": ["Uloric"],
    # Gastrointestinal
    "omeprazole": ["Prilosec"],
    "esomeprazole": ["Nexium"],
    "pantoprazole": ["Protonix"],
    "lansoprazole": ["Prevacid"],
    "rabeprazole": ["Aciphex"],
    "famotidine": ["Pepcid"],
    "cimetidine": ["Tagamet"],
    "ondansetron": ["Zofran"],
    "metoclopramide": ["Reglan"],
    # Endocrine / steroids / urology
    "levothyroxine": ["Synthroid"],
    "methimazole": ["Tapazole"],
    "prednisone": ["Rayos", "Deltasone"],
    "prednisolone": ["Orapred"],
    "methylprednisolone": ["Medrol"],
    "dexamethasone": ["Decadron"],
    "estradiol": ["Estrace"],
    "tamoxifen": ["Soltamox", "Nolvadex"],
    "anastrozole": ["Arimidex"],
    "letrozole": ["Femara"],
    "finasteride": ["Proscar", "Propecia"],
    "dutasteride": ["Avodart"],
    "tamsulosin": ["Flomax"],
    "alfuzosin": ["Uroxatral"],
    "doxazosin": ["Cardura"],
    "terazosin": ["Hytrin"],
    "oxybutynin": ["Ditropan"],
    "tolterodine": ["Detrol"],
    "mirabegron": ["Myrbetriq"],
    "alendronate": ["Fosamax"],
    # Immunosuppressants / oncology
    "cyclosporine": ["Neoral", "Sandimmune"],
    "tacrolimus": ["Prograf"],
    "sirolimus": ["Rapamune"],
    "everolimus": ["Afinitor", "Zortress"],
    "mycophenolate mofetil": ["CellCept"],
    "methotrexate": ["Trexall"],
    "azathioprine": ["Imuran"],
    "imatinib": ["Gleevec"],
    "dasatinib": ["Sprycel"],
    "nilotinib": ["Tasigna"],
    "erlotinib": ["Tarceva"],
    "ibrutinib": ["Imbruvica"],
    "tofacitinib": ["Xeljanz"],
    # Respiratory / allergy / misc
    "montelukast": ["Singulair"],
    "theophylline": ["Theo-24"],
    "hydroxyzine": ["Vistaril"],
    "promethazine": ["Phenergan"],
    "potassium chloride": ["Klor-Con"],
    "varenicline": ["Chantix"],
    "disulfiram": ["Antabuse"],
}

# --------------------------------------------------------------------------- #
# Drug classes: surface forms found in label text, and member generics
# --------------------------------------------------------------------------- #
DRUG_CLASSES: Dict[str, Dict[str, List[str]]] = {
    "nitrates": {
        "aliases": ["nitrates", "organic nitrates", "organic nitrate", "nitrate"],
        "members": ["nitroglycerin", "isosorbide mononitrate", "isosorbide dinitrate"],
    },
    "PDE5 inhibitors": {
        "aliases": [
            "PDE5 inhibitors", "PDE-5 inhibitors", "PDE5 inhibitor", "PDE-5 inhibitor",
            "phosphodiesterase type 5 inhibitors", "phosphodiesterase-5 inhibitors",
            "phosphodiesterase 5 inhibitors", "phosphodiesterase type 5 (PDE5) inhibitors",
            "phosphodiesterase-5 (PDE-5) inhibitors", "phosphodiesterase inhibitors",
        ],
        "members": ["sildenafil", "tadalafil", "vardenafil"],
    },
    "MAO inhibitors": {
        "aliases": [
            "MAOIs", "MAOI", "MAO inhibitors", "MAO inhibitor",
            "monoamine oxidase inhibitors", "monoamine oxidase inhibitor",
            "monoamine oxidase inhibitors (MAOIs)",
        ],
        "members": ["phenelzine", "tranylcypromine", "selegiline", "linezolid"],
    },
    "SSRIs": {
        "aliases": [
            "SSRIs", "SSRI", "selective serotonin reuptake inhibitors",
            "selective serotonin reuptake inhibitor",
        ],
        "members": ["fluoxetine", "sertraline", "paroxetine", "citalopram", "escitalopram", "fluvoxamine"],
    },
    "SNRIs": {
        "aliases": [
            "SNRIs", "SNRI", "serotonin norepinephrine reuptake inhibitors",
            "serotonin-norepinephrine reuptake inhibitors",
        ],
        "members": ["venlafaxine", "desvenlafaxine", "duloxetine"],
    },
    "tricyclic antidepressants": {
        "aliases": ["tricyclic antidepressants", "tricyclic antidepressant", "TCAs", "tricyclics"],
        "members": ["amitriptyline", "nortriptyline", "imipramine"],
    },
    "NSAIDs": {
        "aliases": [
            "NSAIDs", "NSAID", "nonsteroidal anti-inflammatory drugs",
            "non-steroidal anti-inflammatory drugs", "nonsteroidal antiinflammatory drugs",
            "nonsteroidal anti-inflammatory agents",
        ],
        "members": [
            "ibuprofen", "naproxen", "celecoxib", "meloxicam", "diclofenac",
            "indomethacin", "ketorolac", "etodolac",
        ],
    },
    "statins": {
        "aliases": [
            "statins", "statin", "HMG-CoA reductase inhibitors", "HMG-CoA reductase inhibitor",
        ],
        "members": [
            "atorvastatin", "simvastatin", "rosuvastatin", "pravastatin",
            "lovastatin", "pitavastatin", "fluvastatin",
        ],
    },
    "macrolide antibiotics": {
        "aliases": ["macrolide antibiotics", "macrolides", "macrolide"],
        "members": ["clarithromycin", "erythromycin", "azithromycin"],
    },
    "azole antifungals": {
        "aliases": ["azole antifungals", "azole antifungal", "azoles", "triazole antifungals", "antifungals"],
        "members": ["fluconazole", "itraconazole", "ketoconazole", "voriconazole", "posaconazole"],
    },
    "fluoroquinolones": {
        "aliases": ["fluoroquinolones", "fluoroquinolone", "quinolones", "quinolone antibiotics"],
        "members": ["ciprofloxacin", "levofloxacin", "moxifloxacin"],
    },
    "anticoagulants": {
        "aliases": ["anticoagulants", "anticoagulant", "oral anticoagulants", "blood thinners", "blood thinner"],
        "members": ["warfarin", "apixaban", "rivaroxaban", "dabigatran", "edoxaban", "enoxaparin"],
    },
    "antiplatelet agents": {
        "aliases": [
            "antiplatelet agents", "antiplatelet drugs", "antiplatelets",
            "platelet aggregation inhibitors", "platelet inhibitors",
        ],
        "members": ["clopidogrel", "prasugrel", "ticagrelor", "cilostazol", "dipyridamole"],
    },
    "beta-blockers": {
        "aliases": [
            "beta-blockers", "beta blockers", "beta-blocker", "beta-adrenergic blockers",
            "beta-adrenergic blocking agents", "beta-adrenergic antagonists",
        ],
        "members": [
            "metoprolol", "atenolol", "propranolol", "carvedilol",
            "bisoprolol", "nebivolol", "labetalol", "sotalol",
        ],
    },
    "ACE inhibitors": {
        "aliases": [
            "ACE inhibitors", "ACE inhibitor", "angiotensin converting enzyme inhibitors",
            "angiotensin-converting enzyme inhibitors",
        ],
        "members": ["lisinopril", "enalapril", "ramipril", "benazepril", "captopril", "quinapril"],
    },
    "angiotensin receptor blockers": {
        "aliases": [
            "angiotensin receptor blockers", "angiotensin II receptor blockers", "ARBs",
            "angiotensin II receptor antagonists",
        ],
        "members": ["losartan", "valsartan", "irbesartan", "olmesartan", "telmisartan", "candesartan"],
    },
    "calcium channel blockers": {
        "aliases": ["calcium channel blockers", "calcium channel blocker", "calcium antagonists"],
        "members": ["amlodipine", "nifedipine", "diltiazem", "verapamil", "felodipine"],
    },
    "potassium-sparing diuretics": {
        "aliases": ["potassium-sparing diuretics", "potassium sparing diuretics", "potassium-sparing diuretic"],
        "members": ["spironolactone", "eplerenone", "triamterene", "amiloride"],
    },
    "loop diuretics": {
        "aliases": ["loop diuretics", "loop diuretic"],
        "members": ["furosemide", "bumetanide", "torsemide"],
    },
    "thiazide diuretics": {
        "aliases": ["thiazide diuretics", "thiazides", "thiazide diuretic"],
        "members": ["hydrochlorothiazide", "chlorthalidone"],
    },
    "opioids": {
        "aliases": ["opioids", "opioid analgesics", "opioid agonists", "opioid"],
        "members": [
            "tramadol", "oxycodone", "hydrocodone", "morphine", "fentanyl",
            "methadone", "buprenorphine", "tapentadol",
        ],
    },
    "benzodiazepines": {
        "aliases": ["benzodiazepines", "benzodiazepine"],
        "members": ["alprazolam", "lorazepam", "clonazepam", "diazepam", "midazolam", "triazolam"],
    },
    "alpha-blockers": {
        "aliases": [
            "alpha-blockers", "alpha blockers", "alpha-blocker", "alpha-adrenergic blockers",
            "alpha-adrenergic blocking agents", "alpha-adrenergic antagonists",
        ],
        "members": ["tamsulosin", "alfuzosin", "doxazosin", "terazosin"],
    },
    "triptans": {
        "aliases": ["triptans", "triptan", "5-HT1 agonists", "5-HT1B/1D agonists"],
        "members": ["sumatriptan", "rizatriptan"],
    },
    "proton pump inhibitors": {
        "aliases": ["proton pump inhibitors", "proton pump inhibitor", "PPIs"],
        "members": ["omeprazole", "esomeprazole", "pantoprazole", "lansoprazole", "rabeprazole"],
    },
    "sulfonylureas": {
        "aliases": ["sulfonylureas", "sulfonylurea"],
        "members": ["glipizide", "glyburide", "glimepiride"],
    },
    "HIV protease inhibitors": {
        "aliases": ["HIV protease inhibitors", "protease inhibitors", "HIV protease inhibitor"],
        "members": ["ritonavir", "atazanavir", "darunavir"],
    },
    "corticosteroids": {
        "aliases": ["corticosteroids", "corticosteroid", "glucocorticoids"],
        "members": ["prednisone", "prednisolone", "methylprednisolone", "dexamethasone"],
    },
    "antipsychotics": {
        "aliases": ["antipsychotics", "antipsychotic drugs", "antipsychotic agents", "neuroleptics"],
        "members": [
            "quetiapine", "olanzapine", "risperidone", "aripiprazole",
            "ziprasidone", "haloperidol", "clozapine", "lurasidone",
        ],
    },
}

# --------------------------------------------------------------------------- #
# Enzymes and transporters: canonical id -> extra surface forms.
# CYP isoforms written as "CYP3A4", "CYP 3A4" or "cytochrome P450 3A4" are handled
# by a regular expression in core/entities.py; only irregular aliases live here.
# The CYP3A subfamily ("CYP3A", "CYP3A4/5") is folded into CYP3A4 because labels
# use the two interchangeably when describing inhibitors and inducers.
# --------------------------------------------------------------------------- #
ENZYMES: Dict[str, List[str]] = {
    "CYP3A4": ["CYP3A", "CYP3A4/5", "CYP3A5"],
    "CYP2D6": [],
    "CYP2C9": [],
    "CYP2C19": [],
    "CYP2C8": [],
    "CYP1A2": [],
    "CYP2B6": [],
    "CYP2E1": [],
    "P-gp": ["P-glycoprotein", "P glycoprotein", "Pgp", "P-gp"],
    "OATP1B1": ["OATP1B1/1B3", "OATP1B"],
    "OATP1B3": [],
    "BCRP": ["breast cancer resistance protein"],
    "UGT1A1": [],
}
