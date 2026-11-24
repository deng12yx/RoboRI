"""Model variants from main-paper Table 2 and Section 5.1."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Method:
    label: str
    category: str
    backend: str
    native: str
    subgoals: str = "native"
    rimem: bool = False


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
        else "ponderpounce"
        if name == "ponderpounce"
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

# Table 2 reports 26 baselines; Section 5.1 additionally names these two.
ADDITIONAL_BASELINES = ("smolvla", "openvla-oft")
PAPER_BASELINES = tuple(
    name
    for name, method in METHODS.items()
    if not method.rimem and name not in ADDITIONAL_BASELINES
)
RIMEM_BACKBONES = ("framesamp-modul", "tokendrop-modul", "simplememvla")
MAIN_METHODS = PAPER_BASELINES + tuple(base + "+ri-mem" for base in RIMEM_BACKBONES)
