"""Complete RoboMME method inventory (main paper Table 2 and §5.1).

Registry describes actual inference variants, including Qwen vs oracle. A method
name never silently substitutes another backbone or a randomly initialized model.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Method:
    label: str
    category: str
    backend: str
    native: str
    subgoals: str = "native"
    rimem: bool = False
    history_control: str | None = None


METHODS = {}
for rep, short in (("framesamp", "FS"), ("tokendrop", "TD")):
    for fusion, label in (("context", "Ctx"), ("modul", "Mod"), ("expert", "Exp")):
        name = f"{rep}-{fusion}"
        METHODS[name] = Method(
            f"{short}+{label}", "perceptual", "mme", f"perceptual-{rep}-{fusion}"
        )
    name = f"oft-{rep}-context"
    METHODS[name] = Method(
        f"{short}+Ctx (OFT)", "perceptual", "official", f"oft-perceptual-{rep}-context"
    )
METHODS["retrieval-fs-mod"] = Method(
    "FS+Mod (Retrieval)", "perceptual", "mme", "perceptual-framesamp-modul"
)
for kind, short in (("simple", "SimpleSG"), ("grounded", "GroundSG")):
    for predictor in ("qwen", "oracle"):
        METHODS[f"{kind}sg-{predictor}"] = Method(
            f"{short}+{predictor.title()}",
            "symbolic",
            "mme",
            f"symbolic-{kind}-subgoal",
            predictor,
        )
    METHODS[f"oft-{kind}sg-qwen"] = Method(
        f"{short}+Qwen (OFT)",
        "symbolic",
        "official",
        f"oft-symbolic-{kind}-subgoal",
        "qwen",
    )
for fusion, short in (("context", "Ctx"), ("expert", "Exp")):
    METHODS[f"ttt-{fusion}"] = Method(
        f"TTT+{short}", "recurrent", "mme", f"recurrent-ttt-{fusion}"
    )
for name, label, cat, native in (
    ("simplememvla", "SimpleMemVLA", "perceptual", "simplememvla-robomme"),
    ("sam2act-plus", "SAM2Act+", "perceptual", "sam2act-plus"),
    ("hamlet", "HAMLET", "perceptual", "hamlet"),
    ("memoryvla", "MemoryVLA", "perceptual-cognitive", "memoryvla-robomme"),
    ("memer", "MemER", "perceptual-cognitive", "memer"),
    ("ponderpounce", "PonderPounce", "perceptual-cognitive", "ponderpounce"),
    ("pi05", "π0.5", "memory-free", "pi05-no-memory"),
    ("sam2act", "SAM2Act", "memory-free", "sam2act"),
    ("diffusion-policy", "Diffusion Policy", "memory-free", "dp-unet-obs1-text-200k"),
    ("smolvla", "SmolVLA", "memory-free", "smolvla-robomme-repro"),
    ("openvla-oft", "OpenVLA-OFT", "memory-free", "oft-no-memory"),
):
    backend = (
        "mme"
        if name in ("pi05", "memer")
        else "simplememvla"
        if name == "simplememvla"
        else "official"
    )
    METHODS[name] = Method(
        label, cat, backend, native, "qwen" if name == "memer" else "native"
    )
for base in ("framesamp-modul", "tokendrop-modul", "simplememvla"):
    m = METHODS[base]
    METHODS[base + "+ri-mem"] = Method(
        m.label + "+RI-Mem", m.category, m.backend, m.native, m.subgoals, True
    )

# Supplement Table 6: 11 configurations (including the original policy).
ABLATIONS = (
    "original",
    "query_only",
    "reweight_only",
    "no_gate",
    "no_relevance",
    "no_irrelevance",
    "no_observability",
    "no_validity",
    "fixed_weights",
    "random_weights",
    "full",
)

MAIN_METHODS = tuple(METHODS)
for base in ("framesamp-modul", "tokendrop-modul", "simplememvla"):
    m = METHODS[base]
    for control in ("no-history", "shuffled-history", "oracle-mem"):
        METHODS[base + "+" + control] = Method(
            m.label + " / " + control,
            m.category,
            m.backend,
            m.native,
            m.subgoals,
            False,
            control,
        )
